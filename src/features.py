"""Score-statistic features for alpha (Sec. 7.3). Computed from RAW scores, never from labels."""
import torch

from src.normalize import average_rank_desc, zscore

FEATURE_NAMES = [f"{s}_{f}" for s in ("sem", "ctx") for f in ("mean", "std", "max", "gap12", "entropy")] + \
                ["spearman_sem_ctx", "top10_overlap"]


def _single(x, eps):
    top2 = x.topk(2, dim=-1).values
    p = torch.softmax(zscore(x, eps), dim=-1)
    ent = -(p * torch.log(p.clamp_min(eps))).sum(-1)
    return [x.mean(-1), x.std(-1, unbiased=False), top2[..., 0], top2[..., 0] - top2[..., 1], ent]


def spearman(a, b, eps=1e-8):
    ra, rb = average_rank_desc(a), average_rank_desc(b)
    ra = ra - ra.mean(-1, keepdim=True)
    rb = rb - rb.mean(-1, keepdim=True)
    return (ra * rb).sum(-1) / (ra.pow(2).sum(-1).sqrt() * rb.pow(2).sum(-1).sqrt() + eps)


def top_overlap(a, b, k):
    ia, ib = a.topk(k, dim=-1).indices, b.topk(k, dim=-1).indices
    return (ia.unsqueeze(-1) == ib.unsqueeze(-2)).any(-1).float().sum(-1) / k


def score_features(sem_raw, ctx_raw, eps=1e-8, top_k=10):
    """[N, K] x2 -> [N, 12]."""
    f = _single(sem_raw, eps) + _single(ctx_raw, eps) + \
        [spearman(sem_raw, ctx_raw, eps), top_overlap(sem_raw, ctx_raw, top_k)]
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
