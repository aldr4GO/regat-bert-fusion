"""GAT refinement of passage embeddings (Sec. 6.3)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class PassageGAT(nn.Module):
    def __init__(self, d, hidden=128, heads=4, dropout=0.1):
        super().__init__()
        # self-loops are already part of the graph (graph.build_adjacency), so PyG must not add them again
        self.conv1 = GATv2Conv(d, hidden, heads=heads, concat=True, dropout=dropout, add_self_loops=False)
        self.conv2 = GATv2Conv(hidden * heads, d, heads=1, concat=False, dropout=dropout, add_self_loops=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, edge_index, return_attention=False):
        """x: [n, d] node features (p_emb), returns L2-normalised refined embeddings [n, d]."""
        if return_attention:
            h1, att1 = self.conv1(x, edge_index, return_attention_weights=True)
            h2, att2 = self.conv2(F.elu(h1), edge_index, return_attention_weights=True)
        else:
            h2 = self.conv2(F.elu(self.conv1(x, edge_index)), edge_index)
        h = F.normalize(self.norm(x + h2), dim=-1)      # residual -> starts near the bi-encoder
        return (h, (att1, att2)) if return_attention else h


def contextual_scores(gat, P, Q, A):
    """P: [B, K, d] passage emb, Q: [B, d] query emb, A: [B, K, K] adjacency -> s_ctx [B, K]."""
    from src.graph import to_edge_index
    B, K, d = P.shape
    h = gat(P.reshape(B * K, d), to_edge_index(A)).view(B, K, d)
    return torch.einsum("bkd,bd->bk", h, Q)
