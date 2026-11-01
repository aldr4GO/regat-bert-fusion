"""Extra analyses for the paper (no training): alpha-response curves, per-bucket optimal fixed alpha and the
bucketed-alpha ablation. Everything uses the frozen caches; selection is on val, numbers on test.

 curves     mean NDCG@10 of alpha*sem + (1-alpha)*ctx over the alpha grid, every normaliser, val and test.
 buckets    per complexity bucket (hops / type / query length): argmax-alpha of the bucket's mean curve with a
            bootstrap 95% CI, and a tie-robust per-query preference (queries whose NDCG@10 varies with alpha:
            centre of the optimal-alpha set < 0.5 -> prefers the contextual score).
 bucketed   a query-dependent *fixed* alpha: one alpha per bucket chosen on val, applied on test; paired bootstrap
            against the single global alpha0 (also chosen on val). Buckets from gold labels (hops / type) or from
            query length (available at test time for every dataset).

Usage:  python -m src.report_analysis [--B 2000]
Writes: results/report/{alpha_curves,bucket_alpha,bucket_curves,bucketed_alpha}.csv
"""
import argparse

import numpy as np
import pandas as pd

from src.common import get_logger, load_config, results_path, set_env
from src.hypotheses import labels_for
from src.normalize import normalize
from src.oracle import alpha_grid, per_alpha_ndcg
from src.stats import bootstrap
from src.train_fusion import raw_split

log = get_logger("report_analysis")
KEYS = ["A", "B", "musique", "hotpotqa", "beir_hotpotqa", "twowiki"]
GOLD = {"musique": "hops", "hotpotqa": "type", "beir_hotpotqa": "type", "twowiki": "type"}


def nd_matrix(cfg, key, split, norm, grid):
    d = raw_split(cfg, key, split)
    eps = cfg["fusion"]["eps"]
    s, c = normalize(norm, d.scores["sem"], eps, d.mask), normalize(norm, d.scores["ctx"], eps, d.mask)
    return per_alpha_ndcg(s, c, d.gains, d.ideal, grid, cfg["fusion"]["eval"]["cutoff"], d.mask).double().numpy()


def boot_argmax(nd, grid, B, seed=0, chunk=100):
    rng = np.random.default_rng(seed)
    n, out = nd.shape[1], []
    for i in range(0, B, chunk):
        idx = rng.integers(0, n, (min(chunk, B - i), n))
        out.append(grid[nd[:, idx].mean(-1).argmax(0)])
    out = np.concatenate(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def preference(nd, grid):
    """Tie-robust per-query preference: centre of the optimal-alpha set, NaN when NDCG@10 is flat in alpha."""
    best = nd.max(0)
    flat = best - nd.min(0) < 1e-9
    opt = nd >= best - 1e-12
    centre = (opt * grid[:, None]).sum(0) / opt.sum(0)
    return np.where(flat, np.nan, centre)


def labels(cfg, key, split):
    d = raw_split(cfg, key, split)
    nw = d.qlen.numpy()
    out = {"n_words": np.where(nw <= 4, "short (<=4)", np.where(nw <= 7, "medium (5-7)", "long (>=8)"))}
    if key in GOLD and d.meta is not None:
        out[GOLD[key]] = np.array([str(m.get(GOLD[key])) for m in d.meta])
    if key == "A":
        out["wh"] = labels_for(cfg, key, split)["wh"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()
    cfg = load_config()
    set_env(cfg)
    grid_t = alpha_grid(cfg["fusion"]["alpha_grid_step"])
    grid = grid_t.double().numpy()
    pn = "minmax"
    curves, buckets, bcurves, bucketed = [], [], [], []
    for key in KEYS:
        if not results_path(cfg, key, "stage4_sanity.json").exists():
            continue
        ndm = {}
        for norm in cfg["fusion"]["normalizers"]:
            for split in ("val", "test"):
                nd = nd_matrix(cfg, key, split, norm, grid_t)
                if norm == pn:
                    ndm[split] = nd
                curves += [{"key": key, "split": split, "norm": norm, "alpha": a, "ndcg@10": v}
                           for a, v in zip(grid, nd.mean(1))]
        a0 = grid[ndm["val"].mean(1).argmax()]
        for split in ("val", "test"):
            nd = ndm[split]
            lab = labels(cfg, key, split)
            lab["all"] = np.array(["all"] * nd.shape[1])
            pref = preference(nd, grid)
            for name, v in lab.items():
                for b in sorted(set(v)):
                    m = v == b
                    if m.sum() < 30:
                        continue
                    sub = nd[:, m]
                    curve = sub.mean(1)
                    lo, hi = boot_argmax(sub, grid, args.B)
                    p = pref[m]
                    dec = p[np.isfinite(p)]
                    buckets.append({"key": key, "split": split, "label": name, "bucket": b, "n": int(m.sum()),
                                    "alpha0_global": a0, "best_alpha": grid[curve.argmax()], "ci_lo": lo,
                                    "ci_hi": hi, "ndcg_at_best": curve.max(),
                                    "ndcg_at_alpha0": curve[np.argmin(np.abs(grid - a0))],
                                    "n_decisive": len(dec), "mean_pref_alpha": dec.mean() if len(dec) else np.nan,
                                    "frac_pref_ctx": (dec < 0.5).mean() if len(dec) else np.nan,
                                    "frac_pref_sem": (dec > 0.5).mean() if len(dec) else np.nan})
                    bcurves += [{"key": key, "split": split, "label": name, "bucket": b, "alpha": a,
                                 "ndcg@10": c} for a, c in zip(grid, curve)]
        # bucketed fixed alpha: chosen per bucket on val, applied on test
        lv, lt = labels(cfg, key, "val"), labels(cfg, key, "test")
        i0 = int(np.argmin(np.abs(grid - a0)))
        for name in lv:
            if name not in lt:
                continue
            ia = np.full(ndm["test"].shape[1], i0)
            chosen = {}
            for b in sorted(set(lt[name])):
                mv = lv[name] == b
                if mv.sum() >= 30:
                    j = int(ndm["val"][:, mv].mean(1).argmax())
                    chosen[b] = grid[j]
                    ia[lt[name] == b] = j
            nt = ndm["test"]
            y_b = nt[ia, np.arange(nt.shape[1])]
            y_0 = nt[i0]
            v, lo, hi, p = bootstrap(lambda x, y: x.mean() - y.mean(), [y_b, y_0], B=10000)
            bucketed.append({"key": key, "label": name, "alpha0": a0,
                             "alphas": "; ".join(f"{b}={a:.2f}" for b, a in chosen.items()),
                             "test_ndcg_global": y_0.mean(), "test_ndcg_bucketed": y_b.mean(), "diff": v,
                             "ci_lo": lo, "ci_hi": hi, "p": p})
        log.info("%s done (alpha0=%.2f)", key, a0)
    out = results_path(cfg, "report", "x").parent
    pd.DataFrame(curves).to_csv(out / "alpha_curves.csv", index=False, float_format="%.6f")
    pd.DataFrame(buckets).to_csv(out / "bucket_alpha.csv", index=False, float_format="%.5f")
    pd.DataFrame(bcurves).to_csv(out / "bucket_curves.csv", index=False, float_format="%.6f")
    pd.DataFrame(bucketed).to_csv(out / "bucketed_alpha.csv", index=False, float_format="%.5f")
    log.info("bucketed alpha:\n%s", pd.DataFrame(bucketed).to_string(index=False))


if __name__ == "__main__":
    main()
