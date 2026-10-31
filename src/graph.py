"""Per-query candidate graphs (Sec. 6.2). The query is never a node.

Graphs are represented as dense boolean adjacency [N, K, K] (K = 100, so 10k entries per query);
`to_edge_index` turns a batch into a PyG edge_index over B*K nodes with no cross-query edges.
"""
import torch
import torch.nn.functional as F


def _knn_mask(X, k_nn, min_sim):
    """X: [B, K, d] L2-normalised. Directed kNN mask (i -> its k nearest others), cos >= min_sim."""
    B, K, _ = X.shape
    sim = X @ X.transpose(1, 2)
    eye = torch.eye(K, dtype=torch.bool, device=X.device)
    sim = sim.masked_fill(eye, float("-inf"))
    k = min(k_nn, K - 1)
    vals, idx = sim.topk(k, dim=-1)
    keep = vals >= min_sim
    A = torch.zeros(B, K, K, dtype=torch.bool, device=X.device)
    A.scatter_(-1, idx, keep)
    return A


def _permute_nodes(A, generator):
    """Random node-label permutation per query: A'[i, j] = A[pi(i), pi(j)].

    Keeps the degree sequence and the number of edges exactly, but neighbours become random w.r.t. content.
    """
    B, K, _ = A.shape
    perm = torch.rand(B, K, generator=generator, device="cpu").argsort(-1).to(A.device)
    rows = torch.gather(A, 1, perm.unsqueeze(-1).expand(B, K, K))
    return torch.gather(rows, 2, perm.unsqueeze(1).expand(B, K, K))


def build_adjacency(X, k_nn=8, min_sim=0.3, mode="knn_emb", generator=None, chunk=2048):
    """X: [N, K, d] normalised passage embeddings -> bool adjacency [N, K, K] (symmetric, self-loops)."""
    outs = []
    for i in range(0, X.shape[0], chunk):
        x = F.normalize(X[i:i + chunk].float(), dim=-1)
        B, K, _ = x.shape
        if mode == "no_graph":
            A = torch.zeros(B, K, K, dtype=torch.bool, device=x.device)
        elif mode == "knn_emb":
            A = _knn_mask(x, k_nn, min_sim)
        elif mode == "random_graph":
            # the symmetric kNN graph with node labels shuffled: same degrees, random neighbours
            A = _knn_mask(x, k_nn, min_sim)
            A = _permute_nodes(A | A.transpose(1, 2), generator)
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
