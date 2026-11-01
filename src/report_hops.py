"""Complexity-focused analyses for the paper (no training; analysis only, labels never enter a gate).

For MuSiQue (hop count, fine-grained id type) and 2Wiki (question type), on val and test:
 optima     best fixed alpha per bucket under every normaliser and every graph variant (bootstrap CI for min-max)
 signals    CE-only / GAT-only nDCG@10 per bucket
 pref       tie-robust per-query preferred alpha; Spearman(preferred alpha, hops) with a bootstrap CI
 gains      adaptive gate minus best fixed alpha per bucket, for every normaliser (paired bootstrap)
 datasets   val alpha0 vs mean number of gold passages per question (dataset level)

Usage:  python -m src.report_hops
Writes: results/report/hops_{optima,signals,pref,gains,datasets}.csv
"""
import json

import numpy as np
import pandas as pd

from src.common import get_logger, load_config, results_path, set_env
from src.evaluate import paired_bootstrap, per_query_mean
from src.oracle import alpha_grid
from src.report_analysis import boot_argmax, nd_matrix, preference
from src.report_paper import _groups, best_fixed
from src.stats import bootstrap, spearman
from src.train_fusion import raw_split

log = get_logger("report_hops")
NORMS = ("none", "minmax", "zscore", "rank")
SPEC = {"musique": ["hops", "type"], "twowiki": ["type"]}
VARIANTS = {"musique": ["none", "random", "title_link", "hybrid"], "twowiki": []}


def lab(cfg, key, split, field):
    d = raw_split(cfg, key, split)
    return np.array([str(m.get(field)) for m in d.meta])


def main():
    cfg = load_config()
    set_env(cfg)
    gt = alpha_grid(cfg["fusion"]["alpha_grid_step"])
    grid = gt.double().numpy()
    optima, signals, prefs, gains = [], [], [], []
    for ds, fields in SPEC.items():
        for key in [ds] + [f"{ds}@{v}" for v in VARIANTS[ds]]:
            graph = key.split("@")[1] if "@" in key else "knn_emb"
            for split in ("val", "test"):
                for norm in NORMS:
                    if graph != "knn_emb" and norm != "minmax":
                        continue
                    nd = nd_matrix(cfg, key, split, norm, gt)
                    for field in fields:
                        v = lab(cfg, key, split, field)
                        for b in sorted(set(v)):
                            m = v == b
                            if m.sum() < 30:
                                continue
                            sub = nd[:, m]
                            curve = sub.mean(1)
                            lo, hi = boot_argmax(sub, grid, 1000) if norm == "minmax" else (np.nan, np.nan)
                            optima.append({"key": key, "dataset": ds, "graph": graph, "split": split, "norm": norm,
                                           "field": field, "bucket": b, "n": int(m.sum()),
                                           "best_alpha": grid[curve.argmax()], "ci_lo": lo, "ci_hi": hi,
                                           "curve": json.dumps([round(float(c), 6) for c in curve])})
                            if norm == "minmax" and graph == "knn_emb":
                                signals.append({"dataset": ds, "split": split, "field": field, "bucket": b,
                                                "n": int(m.sum()), "ce": curve[-1], "gat": curve[0],
                                                "best": curve.max()})
                    if norm == "minmax" and graph == "knn_emb":
                        pref = preference(nd, grid)
                        for field in fields:
                            v = lab(cfg, key, split, field)
                            ok = np.isfinite(pref)
                            row = {"dataset": ds, "split": split, "field": field, "n_decisive": int(ok.sum())}
                            for b in sorted(set(v)):
                                m = (v == b) & ok
                                row[f"pref_mean_{b}"] = pref[m].mean()
                                row[f"frac_ctx_{b}"] = (pref[m] < 0.5).mean()
                            if field == "hops":
                                h = v.astype(float)
                                r, lo, hi, p = bootstrap(spearman, [pref[ok], h[ok]], B=2000)
                                row.update({"spearman": r, "ci_lo": lo, "ci_hi": hi, "p": p})
                            prefs.append(row)
        # adaptive gains per bucket (test), every normaliser, in-domain graph
        g = _groups(ds)
        for field in fields:
            v = lab(cfg, ds, "test", field)
            for norm in NORMS:
                base = per_query_mean(g[best_fixed(ds, norm)], "test", "ndcg@10")
                for dep in ("Q", "QS", "S"):
                    ck = f"{norm}_{dep}_listwise_lam0.01"
                    if ck not in g:
                        continue
                    a = per_query_mean(g[ck], "test", "ndcg@10")
                    al = per_query_mean(g[ck], "test", "alpha")
                    for b in sorted(set(v)):
                        m = v == b
                        if m.sum() < 30:
                            continue
                        d, lo, hi, p = paired_bootstrap(a[m], base[m])
                        gains.append({"dataset": ds, "field": field, "norm": norm, "dep": dep, "bucket": b,
                                      "n": int(m.sum()), "gain": d, "ci_lo": lo, "ci_hi": hi, "p": p,
                                      "alpha_mean": al[m].mean()})
    # dataset level: val alpha0 (min-max) vs gold passages per question
    rows = []
    for key in ("A", "B", "musique", "hotpotqa", "beir_hotpotqa", "twowiki"):
        d = raw_split(cfg, key, "val")
        gold = d.n_rel.double().mean().item()
        a0 = json.loads(results_path(cfg, key, "alpha0.json").read_text())["minmax"]["alpha0"]
        rows.append({"key": key, "gold_per_q": gold, "alpha0_minmax": a0})
    out = results_path(cfg, "report", "x").parent
    pd.DataFrame(optima).to_csv(out / "hops_optima.csv", index=False, float_format="%.5f")
    pd.DataFrame(signals).to_csv(out / "hops_signals.csv", index=False, float_format="%.5f")
    pd.DataFrame(prefs).to_csv(out / "hops_pref.csv", index=False, float_format="%.5f")
    pd.DataFrame(gains).to_csv(out / "hops_gains.csv", index=False, float_format="%.5f")
    pd.DataFrame(rows).to_csv(out / "hops_datasets.csv", index=False, float_format="%.4f")
    log.info("done")


if __name__ == "__main__":
    main()
