"""Fast vectorised ranking metrics over [N, K] candidate score matrices.

Used for early stopping / grid search / oracle. Final reported numbers come from ir_measures
(src/evaluate.py); tests/test_metrics.py checks the two agree. Ties are broken by candidate order.
IDCG is computed from the full qrels (relevant passages outside the top-K still count), which is what
trec_eval does, so queries without a relevant candidate score 0.
"""
import torch


def ideal_gains(qrels_list, k):
    """qrels_list: list of {pid: rel}; returns [N, k] top-k gains of the ideal ranking."""
    out = torch.zeros(len(qrels_list), k)
    for i, qr in enumerate(qrels_list):
        g = sorted((v for v in qr.values() if v > 0), reverse=True)[:k]
        if g:
            out[i, :len(g)] = torch.tensor(g, dtype=torch.float)
    return out


def _discount(k, device):
    return 1.0 / torch.log2(torch.arange(2, k + 2, device=device, dtype=torch.float))


def ndcg_at_k(scores, gains, ideal, k=10):
    """scores, gains: [..., K]; ideal: [..., k]. Returns [...] NDCG@k (linear gain, like trec_eval)."""
    order = torch.sort(scores, dim=-1, descending=True, stable=True).indices[..., :k]
    g = torch.gather(gains.expand_as(scores), -1, order)
    disc = _discount(order.shape[-1], scores.device)
    dcg = (g * disc).sum(-1)
    idcg = (ideal[..., :k] * _discount(ideal[..., :k].shape[-1], scores.device)).sum(-1)
    return torch.where(idcg > 0, dcg / idcg.clamp_min(1e-12), torch.zeros_like(dcg))


def rr_at_k(scores, binary, k=10):
    order = torch.sort(scores, dim=-1, descending=True, stable=True).indices[..., :k]
    b = torch.gather(binary.expand_as(scores).float(), -1, order)
    ranks = torch.arange(1, order.shape[-1] + 1, device=scores.device, dtype=torch.float)
    rr = b / ranks
    return rr.max(-1).values
