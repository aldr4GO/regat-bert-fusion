"""Axis 1: per-query score normalisation (Sec. 7.1). Every function maps [..., K] -> [..., K].

Variable candidate counts (multi-hop extension, Sec. 3.2): `mask` [..., K] marks valid candidates; statistics
use valid entries only and padded positions return 0. With mask=None (or all valid) the original code runs
unchanged, so MS MARCO results are bit-identical (golden regression test).
"""
import torch


def _all_valid(mask):
    return mask is None or bool(mask.all())


def none(x, eps=1e-8, mask=None):
    return x if _all_valid(mask) else x.masked_fill(~mask, 0.0)


def minmax(x, eps=1e-8, mask=None):
    if _all_valid(mask):
        lo = x.min(-1, keepdim=True).values
        hi = x.max(-1, keepdim=True).values
        return (x - lo) / (hi - lo + eps)
    lo = x.masked_fill(~mask, float("inf")).min(-1, keepdim=True).values
    hi = x.masked_fill(~mask, float("-inf")).max(-1, keepdim=True).values
    return ((x - lo) / (hi - lo + eps)).masked_fill(~mask, 0.0)


def masked_mean_std(x, mask):
    m = mask.float()
    n = m.sum(-1, keepdim=True).clamp_min(1)
    mean = (x * m).sum(-1, keepdim=True) / n
    std = (((x - mean) * m).pow(2).sum(-1, keepdim=True) / n).sqrt()
    return mean, std


def zscore(x, eps=1e-8, mask=None):
    if _all_valid(mask):
        return (x - x.mean(-1, keepdim=True)) / (x.std(-1, unbiased=False, keepdim=True) + eps)
    mean, std = masked_mean_std(x, mask)
    return ((x - mean) / (std + eps)).masked_fill(~mask, 0.0)


def average_rank_desc(x, chunk=4096, mask=None):
    """Rank 1 = highest score; ties get the average of the ranks they span (valid entries only)."""
    flat = x.reshape(-1, x.shape[-1])
    fm = None if _all_valid(mask) else mask.reshape(-1, x.shape[-1])
    out = []
    for i in range(0, flat.shape[0], chunk):
        v = flat[i:i + chunk]
        greater = v.unsqueeze(-2) > v.unsqueeze(-1)          # [i, j] = x_j > x_i
        equal = v.unsqueeze(-2) == v.unsqueeze(-1)            # includes i itself
        if fm is not None:
            valid_j = fm[i:i + chunk].unsqueeze(-2)
            greater, equal = greater & valid_j, equal & valid_j
        out.append(greater.sum(-1).float() + (equal.sum(-1).float() + 1) / 2)
    return torch.cat(out).view_as(x)


def rank(x, eps=1e-8, mask=None):
    if _all_valid(mask):
        K = x.shape[-1]
        return 1 - (average_rank_desc(x) - 1) / (K - 1)
    n = mask.float().sum(-1, keepdim=True)
    r = 1 - (average_rank_desc(x, mask=mask) - 1) / (n - 1).clamp_min(1)
    return r.masked_fill(~mask, 0.0)


NORMALIZERS = {"none": none, "minmax": minmax, "zscore": zscore, "rank": rank}


def normalize(name, x, eps=1e-8, mask=None):
    return NORMALIZERS[name](x, eps, mask)
