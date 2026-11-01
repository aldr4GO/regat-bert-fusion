"""Bootstrap statistics shared by the diagnostics and hypothesis tests (resampling queries)."""
import numpy as np
from scipy.stats import rankdata


def _p_two_sided(boots):
    return float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))


def bootstrap(stat, arrays, B=10000, seed=0, chunk=500):
    """stat(*arrays_resampled) over B paired resamples of the rows. Returns (value, lo, hi, p for stat != 0)."""
    arrays = [np.asarray(a, dtype=np.float64) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    boots = []
    for i in range(0, B, chunk):
        idx = rng.integers(0, n, (min(chunk, B - i), n))
        boots.extend(stat(*[a[j] for a in arrays]) for j in idx)
    boots = np.asarray(boots)
    boots = boots[np.isfinite(boots)]
    v = float(stat(*arrays))
    if len(boots) == 0:
        return v, float("nan"), float("nan"), float("nan")
    return v, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), max(_p_two_sided(boots), 1.0 / B)


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def mean_ci(x, B=10000, seed=0):
    return bootstrap(np.mean, [x], B, seed)


def diff_of_means_ci(a, b, B=10000, seed=0, chunk=500):
    """mean(a) - mean(b) for two independent samples, each resampled within itself (e.g. two datasets)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = []
    for i in range(0, B, chunk):
        m = min(chunk, B - i)
        boots.append(a[rng.integers(0, len(a), (m, len(a)))].mean(1) - b[rng.integers(0, len(b), (m, len(b)))].mean(1))
    boots = np.concatenate(boots)
    v = float(a.mean() - b.mean())
    return v, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), max(_p_two_sided(boots), 1.0 / B)


def group_diff(g1, g2):
    """Statistic of paired (values, group codes): mean(values | code in g1) - mean(values | code in g2)."""
    def stat(v, g):
        a, b = v[np.isin(g, g1)], v[np.isin(g, g2)]
        return a.mean() - b.mean() if len(a) and len(b) else np.nan
    return stat


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as the input; NaNs stay NaN)."""
    p = np.asarray(pvals, dtype=np.float64)
    ok = np.where(np.isfinite(p))[0]
    adj = np.full_like(p, np.nan)
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj
