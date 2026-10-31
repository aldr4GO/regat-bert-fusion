import pytest
import torch

from src.gat import PassageGAT, contextual_scores
from src.graph import build_adjacency, to_edge_index


@pytest.fixture
def X():
    return torch.nn.functional.normalize(torch.randn(6, 100, 32, generator=torch.Generator().manual_seed(0)), dim=-1)


@pytest.mark.parametrize("mode", ["knn_emb", "no_graph", "random_graph"])
def test_graph_properties(X, mode):
    A = build_adjacency(X, 8, 0.0 if mode != "no_graph" else 0.3, mode, generator=torch.Generator().manual_seed(0))
    assert torch.equal(A, A.transpose(1, 2))                       # undirected
    assert A.diagonal(dim1=1, dim2=2).all()                         # self-loops
    assert (A.sum(-1) >= 1).all()                                   # degree >= 1
    ei = to_edge_index(A)
    assert torch.equal(ei[0] // 100, ei[1] // 100)                  # no cross-query edges
    if mode == "no_graph":
        assert A.sum() == 6 * 100


def test_knn_degree_and_threshold(X):
    A = build_adjacency(X, 8, 0.0)
    assert (A.sum(-1) >= 9).all()                                   # >= k_nn + self
    A_hi = build_adjacency(X, 8, 0.99)                              # random vectors: no edge that similar
    assert A_hi.sum() == 6 * 100                                    # only self-loops survive


def test_random_graph_same_degrees(X):
    knn = build_adjacency(X, 8, 0.1, "knn_emb")
    rnd = build_adjacency(X, 8, 0.1, "random_graph", generator=torch.Generator().manual_seed(0))
    assert torch.equal(knn.sum(-1).sort(-1).values, rnd.sum(-1).sort(-1).values)   # same degree sequence
    assert knn.sum() == rnd.sum()
    assert not torch.equal(knn, rnd)


def test_gat_forward_shapes_and_norm(X):
    gat = PassageGAT(32, 8, 2, 0.1).eval()
    A = build_adjacency(X, 8, 0.3)
    s = contextual_scores(gat, X, torch.nn.functional.normalize(torch.randn(6, 32), dim=-1), A)
    assert s.shape == (6, 100) and (s.abs() <= 1 + 1e-5).all()
    h, att = gat(X.reshape(-1, 32), to_edge_index(A), return_attention=True)
    assert torch.allclose(h.norm(dim=-1), torch.ones(600), atol=1e-5)
