"""Per-query candidate graphs (Sec. 6.2). The query is never a node.

Graph types (multi-hop extension Sec. 3.3): knn_emb (default), none / no_graph (self-loops only),
random / random_graph (kNN graph with shuffled node labels: same degrees, random neighbours),
title_link (paragraph i mentions the title of j or vice versa; precomputed, see src/graph_prep.py) and
hybrid (knn_emb | title_link). `mask` [N, K] marks valid candidates of variable-size pools: padded nodes get
no edges except their self-loop.

Graphs are represented as dense boolean adjacency [N, K, K] (K = 100, so 10k entries per query);
`to_edge_index` turns a batch into a PyG edge_index over B*K nodes with no cross-query edges.
"""
import torch
import torch.nn.functional as F


def _knn_mask(X, k_nn, min_sim, mask=None):
    """X: [B, K, d] L2-normalised. Directed kNN mask (i -> its k nearest others), cos >= min_sim."""
    B, K, _ = X.shape
    sim = X @ X.transpose(1, 2)
    eye = torch.eye(K, dtype=torch.bool, device=X.device)
    sim = sim.masked_fill(eye, float("-inf"))
    if mask is not None:
        sim = sim.masked_fill(~mask.unsqueeze(1), float("-inf"))     # never point to a padded node
    k = min(k_nn, K - 1)
    A = torch.zeros(B, K, K, dtype=torch.bool, device=X.device)
    if k <= 0:
        return A
    vals, idx = sim.topk(k, dim=-1)
    keep = vals >= min_sim
    A.scatter_(-1, idx, keep)
    if mask is not None:
        A &= mask.unsqueeze(-1)                                        # padded nodes have no out-edges
    return A


def _permute_nodes(A, generator, mask=None):
    """Random node-label permutation per query: A'[i, j] = A[pi(i), pi(j)].

    Keeps the degree sequence and the number of edges exactly, but neighbours become random w.r.t. content.
    With a mask, only valid nodes are permuted (padding is trailing, so padded positions map to themselves).
    """
    B, K, _ = A.shape
    r = torch.rand(B, K, generator=generator, device="cpu")
    if mask is not None:
        r = r.masked_fill(~mask.cpu(), 2.0)
    perm = r.argsort(-1).to(A.device)
    rows = torch.gather(A, 1, perm.unsqueeze(-1).expand(B, K, K))
    return torch.gather(rows, 2, perm.unsqueeze(1).expand(B, K, K))


ALIASES = {"none": "no_graph", "random": "random_graph"}


def build_adjacency(X, k_nn=8, min_sim=0.3, mode="knn_emb", generator=None, chunk=2048, mask=None,
                    title_adj=None):
    """X: [N, K, d] normalised passage embeddings -> bool adjacency [N, K, K] (symmetric, self-loops).

    mask: [N, K] valid candidates (None = all valid); title_adj: [N, K, K] bool for title_link / hybrid.
    """
    mode = ALIASES.get(mode, mode)
    if mask is not None and bool(mask.all()):
        mask = None
    outs = []
    for i in range(0, X.shape[0], chunk):
        x = F.normalize(X[i:i + chunk].float(), dim=-1)
        m = None if mask is None else mask[i:i + chunk].to(x.device)
        B, K, _ = x.shape
        if mode == "no_graph":
            A = torch.zeros(B, K, K, dtype=torch.bool, device=x.device)
        elif mode == "knn_emb":
            A = _knn_mask(x, k_nn, min_sim, m)
        elif mode == "random_graph":
            # the symmetric kNN graph with node labels shuffled: same degrees, random neighbours
            A = _knn_mask(x, k_nn, min_sim, m)
            A = _permute_nodes(A | A.transpose(1, 2), generator, m)
        elif mode in ("title_link", "hybrid"):
            assert title_adj is not None, f"{mode} needs the precomputed title-link adjacency"
            A = title_adj[i:i + chunk].to(x.device).clone()
            if mode == "hybrid":
                A |= _knn_mask(x, k_nn, min_sim, m)
            if m is not None:
                A &= m.unsqueeze(-1) & m.unsqueeze(1)
        else:
            raise ValueError(f"unknown graph mode {mode}")
        A = A | A.transpose(1, 2)
        A |= torch.eye(K, dtype=torch.bool, device=x.device)
        outs.append(A)
    return torch.cat(outs)


def to_edge_index(A):
    """A: [B, K, K] bool -> edge_index [2, E] over B*K nodes (source j -> target i for A[b, i, j])."""
    B, K, _ = A.shape
    b, i, j = A.nonzero(as_tuple=True)
    return torch.stack([b * K + j, b * K + i])
