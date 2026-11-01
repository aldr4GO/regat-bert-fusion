"""Score-statistic features for alpha (Sec. 7.3). Computed from RAW scores, never from labels.

Mask-aware for variable candidate counts (multi-hop Sec. 3.2); with mask=None / all valid the original code
path runs unchanged (golden regression).
"""
import torch

from src.normalize import _all_valid, average_rank_desc, masked_mean_std, zscore

FEATURE_NAMES = [f"{s}_{f}" for s in ("sem", "ctx") for f in ("mean", "std", "max", "gap12", "entropy")] + \
                ["spearman_sem_ctx", "top10_overlap"]


def _single(x, eps):
    top2 = x.topk(2, dim=-1).values
    p = torch.softmax(zscore(x, eps), dim=-1)
    ent = -(p * torch.log(p.clamp_min(eps))).sum(-1)
    return [x.mean(-1), x.std(-1, unbiased=False), top2[..., 0], top2[..., 0] - top2[..., 1], ent]


def _single_masked(x, mask, eps):
    n = mask.sum(-1)
    mean, std = masked_mean_std(x, mask)
    xm = x.masked_fill(~mask, float("-inf"))
    k = min(2, x.shape[-1])
    top = xm.topk(k, dim=-1).values
    first = top[..., 0]
    second = top[..., 1] if k == 2 else first
    gap = torch.where(n >= 2, first - second, torch.zeros_like(first))
    logits = zscore(x, eps, mask).masked_fill(~mask, float("-inf"))
    p = torch.softmax(logits, dim=-1)
    ent = -(p * torch.log(p.clamp_min(eps))).sum(-1)
    return [mean.squeeze(-1), std.squeeze(-1), first, gap, ent]


def spearman(a, b, eps=1e-8, mask=None):
    if _all_valid(mask):
        ra, rb = average_rank_desc(a), average_rank_desc(b)
        ra = ra - ra.mean(-1, keepdim=True)
        rb = rb - rb.mean(-1, keepdim=True)
        return (ra * rb).sum(-1) / (ra.pow(2).sum(-1).sqrt() * rb.pow(2).sum(-1).sqrt() + eps)
    m = mask.float()
    n = m.sum(-1, keepdim=True).clamp_min(1)
    ra, rb = average_rank_desc(a, mask=mask), average_rank_desc(b, mask=mask)
    ra = (ra - (ra * m).sum(-1, keepdim=True) / n) * m
    rb = (rb - (rb * m).sum(-1, keepdim=True) / n) * m
    return (ra * rb).sum(-1) / (ra.pow(2).sum(-1).sqrt() * rb.pow(2).sum(-1).sqrt() + eps)


def top_overlap(a, b, k, mask=None):
    if _all_valid(mask) and k <= a.shape[-1]:
        ia, ib = a.topk(k, dim=-1).indices, b.topk(k, dim=-1).indices
        return (ia.unsqueeze(-1) == ib.unsqueeze(-2)).any(-1).float().sum(-1) / k
    mask = torch.ones_like(a, dtype=torch.bool) if mask is None else mask
    m_eff = mask.sum(-1).clamp(max=k).clamp_min(1)                     # m = min(k, #valid)

    def in_top(x):
        order = x.masked_fill(~mask, float("-inf")).sort(dim=-1, descending=True, stable=True).indices
        pos = order.argsort(-1)
        return (pos < m_eff.unsqueeze(-1)) & mask

    return (in_top(a) & in_top(b)).float().sum(-1) / m_eff


def score_features(sem_raw, ctx_raw, eps=1e-8, top_k=10, mask=None):
    """[N, K] x2 -> [N, 12]."""
    if _all_valid(mask) and sem_raw.shape[-1] >= 2:
        f = _single(sem_raw, eps) + _single(ctx_raw, eps)
    else:
        mask = torch.ones_like(sem_raw, dtype=torch.bool) if mask is None else mask
        f = _single_masked(sem_raw, mask, eps) + _single_masked(ctx_raw, mask, eps)
    f += [spearman(sem_raw, ctx_raw, eps, mask), top_overlap(sem_raw, ctx_raw, top_k, mask)]
    return torch.stack(f, -1)


class FeatureScaler:
    """Zero-mean / unit-variance standardisation fitted on fusion_train only."""

    def __init__(self, mean=None, std=None, eps=1e-8):
        self.mean, self.std, self.eps = mean, std, eps

    def fit(self, F):
        self.mean, self.std = F.mean(0), F.std(0, unbiased=False)
        return self

    def __call__(self, F):
        return (F - self.mean) / (self.std + self.eps)

    def state(self):
        return {"mean": self.mean, "std": self.std, "names": FEATURE_NAMES}
