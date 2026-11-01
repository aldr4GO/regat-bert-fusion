import numpy as np
import torch

from src.datasets.common import norm_title
from src.graph import build_adjacency, to_edge_index
from src.graph_prep import title_links


def test_norm_title():
    assert norm_title("Polish-Russian War (film)") == "polish-russian war"
    assert norm_title("  Ed   Wood ") == "ed wood"


def test_title_links_symmetric_and_whole_phrase():
    titles = ["Ed Wood", "Ed Wood (film)", "Tim Burton", "Wo"]
    texts = ["Ed Wood was a director.", "A film by Tim Burton about Ed Wood.", "Tim Burton directed Edwardian films.",
             "Wo is short."]
    A = title_links(titles, texts, 3)
    assert (A == A.T).all() and not A.diagonal().any()
    assert A[1, 0] and A[1, 2]                 # text 1 mentions "ed wood" and "tim burton"
    assert not A[2, 0]                         # "Edwardian" is not the whole phrase "ed wood"
    assert not A[:, 3].any()                   # title shorter than 3 characters ignored


def _X(B=4, K=12, d=16, seed=0):
    return torch.nn.functional.normalize(torch.randn(B, K, d, generator=torch.Generator().manual_seed(seed)), dim=-1)


def test_masked_graphs():
    X = _X()
    mask = torch.ones(4, 12, dtype=torch.bool)
    mask[1, 7:] = False
    mask[3, 2:] = False
    title = torch.zeros(4, 12, 12, dtype=torch.bool)
    title[0, 0, 1] = title[0, 1, 0] = True
    title[1, 0, 10] = title[1, 10, 0] = True      # touches a padded node -> must be dropped
    for mode in ("knn_emb", "none", "random", "title_link", "hybrid"):
        A = build_adjacency(X, 3, 0.0, mode, generator=torch.Generator().manual_seed(0), mask=mask, title_adj=title)
        assert torch.equal(A, A.transpose(1, 2)) and A.diagonal(dim1=1, dim2=2).all()
        pad_pairs = ~(mask.unsqueeze(-1) & mask.unsqueeze(1)) & ~torch.eye(12, dtype=torch.bool)
        assert not A[pad_pairs].any(), mode     # padded nodes only have their self-loop
        ei = to_edge_index(A)
        assert torch.equal(ei[0] // 12, ei[1] // 12)
    knn = build_adjacency(X, 3, 0.0, "knn_emb", mask=mask)
    rnd = build_adjacency(X, 3, 0.0, "random", generator=torch.Generator().manual_seed(0), mask=mask)
    assert torch.equal(knn.sum(-1).sort(-1).values, rnd.sum(-1).sort(-1).values)
    A = build_adjacency(X, 3, 0.0, "title_link", mask=mask, title_adj=title)
    assert A[0, 0, 1] and not A[1, 0, 10]


def test_closed_pool_k3_degree():
    A = build_adjacency(_X(2, 10), 3, -1.0, "knn_emb")
    deg = A.sum(-1) - 1
    assert (deg >= 3).all() and (deg <= 9).all()
