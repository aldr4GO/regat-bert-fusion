"""Axis 1: per-query score normalisation (Sec. 7.1). Every function maps [..., K] -> [..., K]."""
import torch


def none(x, eps=1e-8):
    return x


def minmax(x, eps=1e-8):
    lo = x.min(-1, keepdim=True).values
    hi = x.max(-1, keepdim=True).values
    return (x - lo) / (hi - lo + eps)


def zscore(x, eps=1e-8):
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, unbiased=False, keepdim=True) + eps)


def average_rank_desc(x, chunk=4096):
    """Rank 1 = highest score; ties get the average of the ranks they span."""
    flat = x.reshape(-1, x.shape[-1])
    out = []
    for i in range(0, flat.shape[0], chunk):
        v = flat[i:i + chunk]
        greater = (v.unsqueeze(-2) > v.unsqueeze(-1)).sum(-1).float()     # #{j : x_j > x_i}
        equal = (v.unsqueeze(-2) == v.unsqueeze(-1)).sum(-1).float()      # includes i itself
        out.append(greater + (equal + 1) / 2)
    return torch.cat(out).view_as(x)


def rank(x, eps=1e-8):
    K = x.shape[-1]
    return 1 - (average_rank_desc(x) - 1) / (K - 1)


NORMALIZERS = {"none": none, "minmax": minmax, "zscore": zscore, "rank": rank}


def normalize(name, x, eps=1e-8):
    return NORMALIZERS[name](x, eps)
