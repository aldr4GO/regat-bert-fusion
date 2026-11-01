"""Stage 6: metrics (ir_measures / pytrec_eval backend), run files, tables and significance.

Usage:  python -m src.evaluate --protocol A|B|<dataset>[@variant]
Writes: results/{key}/{main_table,main_table_long,significance,oracle_gap}.csv
Run files: results/{key}/runs/{name}.json (meta + aggregate metrics) and {name}.npz (per-query values)

Metrics (configs/multihop.yaml:metrics): nDCG@k, MRR@k, P@k, R@k for k in {1,3,5,10,50}, MAP, R@2 and
AllRecall@k (all relevant passages of the query inside the top k) for k in {2,5,10}. Binary metrics use
rel >= 2 on TREC DL and rel >= 1 elsewhere; nDCG uses graded labels. P@k divides by k (trec_eval), also when a
pool has fewer than k candidates.
"""
import argparse
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from src.common import get_logger, is_msmarco, load_config, load_json, path, results_path, save_json, set_env, test_sets
from src.metrics import all_recall_at_k
from src.store import binary_level

log = get_logger("evaluate")
HEADLINE = ("ndcg@10", "mrr@10", "map")


def metric_names(cfg):
    mc = cfg["multihop"]["metrics"]
    names = [f"{m}@{k}" for m in ("ndcg", "mrr", "p", "r") for k in mc["k"]]
    names += ["map"] + [f"r@{k}" for k in mc["extra_recall_k"] if f"r@{k}" not in names]
    return names + [f"allr@{k}" for k in mc["allrecall_k"]]


# ------------------------------------------------------------------ per-query metrics
def per_query(d, scores, cfg, with_recall=False):
    """d: SplitData, scores: [N, K]. Returns {metric: np.array[N]} in d.qids order (padded entries ignored)."""
    import ir_measures
    from ir_measures import AP, P, RR, R, nDCG
    lvl = binary_level(d.name)
    mc = cfg["multihop"]["metrics"]
    measures = {}
    for k in mc["k"]:
        measures[f"ndcg@{k}"] = nDCG @ k
        measures[f"mrr@{k}"] = RR(rel=lvl) @ k
        measures[f"p@{k}"] = P(rel=lvl) @ k
        measures[f"r@{k}"] = R(rel=lvl) @ k
    measures["map"] = AP(rel=lvl)
    for k in mc["extra_recall_k"]:
        measures[f"r@{k}"] = R(rel=lvl) @ k
    if with_recall:
        measures[f"r@{scores.shape[1]}"] = R(rel=lvl) @ scores.shape[1]
    # rank-based scores: ties broken by candidate (BM25) order exactly like src/metrics.py, never by docid
    mask = d.mask
    masked = scores.masked_fill(~mask, float("-inf"))
    order = torch.sort(masked, dim=-1, descending=True, stable=True).indices
    K = scores.shape[1]
    rank_score = torch.empty_like(order).scatter_(-1, order, torch.arange(K, 0, -1).expand_as(order)).tolist()
    run = {q: {p: float(v) for p, v in zip(pl, sl[:len(pl)])} for q, pl, sl in zip(d.qids, d.pids, rank_score)}
    qrels = {q: {p: int(r) for p, r in d.qrels[q].items()} for q in d.qids}
    vals = defaultdict(dict)
    inv = {str(m): k for k, m in measures.items()}
    for m in ir_measures.iter_calc(list(measures.values()), qrels, run):
        vals[inv[str(m.measure)]][m.query_id] = m.value
    # queries without any relevant judgement are absent from iter_calc output -> 0
    out = {k: np.array([vals[k].get(q, 0.0) for q in d.qids]) for k in measures}
    for k in mc["allrecall_k"]:
        out[f"allr@{k}"] = all_recall_at_k(scores, d.labels, d.n_rel, k, mask).double().numpy()
    return out


def save_run(cfg, protocol, name, meta, scores_by_set, alphas_by_set=None, with_recall=False):
    from src.train_fusion import raw_split
    keep = set(cfg["fusion"]["eval"]["per_query_metrics"])
    arrays, metrics = {}, {}
    for s, sc in scores_by_set.items():
        d = raw_split(cfg, protocol, s)
        pq = per_query(d, sc, cfg, with_recall)
        metrics[s] = {k: float(v.mean()) for k, v in pq.items()}
        for k, v in pq.items():
            if k in keep:
                arrays[f"{s}__{k}"] = v.astype(np.float32)
        if alphas_by_set is not None and s in alphas_by_set:
            arrays[f"{s}__alpha"] = torch.as_tensor(alphas_by_set[s]).float().numpy()
    p = results_path(cfg, protocol, "runs", f"{name}.json")
    np.savez_compressed(p.with_suffix(".npz"), **arrays)
    save_json({"name": name, **meta, "metrics": metrics}, p)
    return metrics


def reference_runs(cfg, protocol):
    """BM25 (in-pool BM25 for closed pools), s_bi, CE alone (alpha = 1), GAT-only (alpha = 0). No training."""
    from src.train_fusion import raw_split
    sets = ["val"] + test_sets(cfg, protocol)
    for ref, key in (("bm25", "bm25"), ("bi", "bi"), ("ce", "sem"), ("gat", "ctx")):
        scores = {s: raw_split(cfg, protocol, s).scores[key] for s in sets}
        save_run(cfg, protocol, f"{protocol}_ref_{ref}", {"protocol": protocol, "dep": f"ref_{ref}", "norm": "-"},
                 scores, with_recall=(ref == "bm25" and is_msmarco(protocol)))


# ------------------------------------------------------------------ aggregation
def config_key(name, protocol=None):
    """Strips key prefix and seed suffix: 'A_zscore_QS_listwise_lam0.01_s3' -> 'zscore_QS_listwise_lam0.01'."""
    rest = name[len(protocol) + 1:] if protocol and name.startswith(protocol + "_") else name.split("_", 1)[1]
    return re.sub(r"_s\d+$", "", rest)


def load_runs(cfg, protocol):
    runs = {}
    for p in sorted(results_path(cfg, protocol, "runs", "x").parent.glob("*.json")):  # x: dir is created
        meta = load_json(p)
        meta["arrays"] = p.with_suffix(".npz")
        runs[meta["name"]] = meta
    return runs


def group_runs(runs):
    groups = defaultdict(list)
    for name, r in runs.items():
        groups[config_key(name, r.get("protocol"))].append(r)
    return groups


def per_query_mean(group, set_name, metric):
    """Seed-averaged per-query values."""
    return np.mean([np.load(r["arrays"])[f"{set_name}__{metric}"].astype(np.float64) for r in group], axis=0)


def paired_bootstrap(a, b, B=10000, seed=0, chunk=500):
    """Mean difference a-b, 95% CI and two-sided p-value from a paired bootstrap over queries."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return bootstrap_mean(d, B, seed, chunk)


def bootstrap_mean(d, B=10000, seed=0, chunk=500):
    """Mean of d, 95% percentile CI, and two-sided p-value for mean != 0."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    rng = np.random.default_rng(seed)
    boots = np.concatenate([d[rng.integers(0, n, (min(chunk, B - i), n))].mean(1) for i in range(0, B, chunk)])
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), max(p, 1.0 / B)


def label(key, protocol="A"):
    if key.startswith("ref_"):
        bm25 = "BM25" if is_msmarco(protocol) or protocol.startswith("beir") else "BM25 (in-pool order)"
        return {"ref_bm25": bm25, "ref_bi": "SBERT (s_bi)", "ref_ce": "CE-BERT (alpha=1)",
                "ref_gat": "GAT-only (alpha=0)"}[key]
    return key


def row_order(cfg, key):
    refs = ["ref_bm25", "ref_bi", "ref_ce", "ref_gat"]
    if key in refs:
        return (0, refs.index(key), 0, key)
    norms = cfg["fusion"]["normalizers"]
    norm, rest = key.split("_", 1)
    deps = ["fixedgrid", "fixed_listwise", "Q_listwise", "QS_listwise", "S_listwise", "QSC_listwise", "paper",
            "fixed_bce", "QS_bce"]
    lam = cfg["fusion"]["train"]["lambda"]
    rank = next((i for i, d in enumerate(deps) if rest.startswith(d)), len(deps))
    extra = 0 if (rest == "fixedgrid" or rest.endswith(f"lam{lam:g}") or rest.startswith("paper")) else 1
    if rest == "oracle":
        return (2, norms.index(norm) if norm in norms else 9, 0, key)
    return (1, norms.index(norm) if norm in norms else 9, extra * 100 + rank, key)


def main_table(cfg, protocol, groups):
    sets = test_sets(cfg, protocol)
    names = metric_names(cfg)
    rows, long_rows = [], []
    for key in sorted(groups, key=lambda k: row_order(cfg, k)):
        g = groups[key]
        row = {"config": label(key, protocol), "n_seeds": len(g),
               "val_ndcg@10": np.mean([r["metrics"]["val"]["ndcg@10"] for r in g])}
        for s in sets:
            present = [m for m in names if all(m in r["metrics"].get(s, {}) for r in g)]
            for m in present:
                v = [r["metrics"][s][m] for r in g]
                row[f"{s}_{m}_mean"] = np.mean(v)
                row[f"{s}_{m}_std"] = np.std(v) if len(v) > 1 else 0.0
                long_rows.append({"config": row["config"], "set": s, "metric": m, "mean": np.mean(v),
                                  "std": row[f"{s}_{m}_std"], "n_seeds": len(g)})
        if key == "ref_bm25":
            for s in sets:
                for m, v in g[0]["metrics"][s].items():
                    if m.startswith("r@") and m not in names:
                        row[f"{s}_{m}"] = v
        if "oracle" in key:
            row["config"] += " (UPPER BOUND, not a method)"
        rows.append(row)
    df = pd.DataFrame(rows)
    fs_path = path(cfg, "results_dir", "bm25_first_stage.json")
    if is_msmarco(protocol) and fs_path.exists():
        for s, vals in load_json(fs_path)["first_stage_bm25"].items():
            for m, v in vals.items():
                df.loc[df.config == "BM25", f"{s}_first_stage_{m}"] = v
    ds_fs = results_path(cfg, protocol, "first_stage.json")
    if ds_fs.exists():
        for m, v in load_json(ds_fs).items():
            df.loc[df.config.str.startswith("BM25"), f"test_first_stage_{m}"] = v
    df.to_csv(results_path(cfg, protocol, "main_table.csv"), index=False, float_format="%.4f")
    pd.DataFrame(long_rows).to_csv(results_path(cfg, protocol, "main_table_long.csv"), index=False,
                                   float_format="%.4f")
    return df


def primary_norm(cfg):
    """Plan Sec. 4 DECISION: the normaliser whose val-best config wins on MS MARCO protocol A val.

    Fixed once in results/primary_norm.json (written on first use, normally by src/diagnose_msmarco.py)."""
    p = path(cfg, "results_dir_main", "primary_norm.json")
    if p.exists():
        return load_json(p)["primary_norm"]
    if cfg["multihop"]["primary_norm"] != "auto":
        return cfg["multihop"]["primary_norm"]
    main_cfg = {**cfg, "data": {**cfg["data"], "paths": {**cfg["data"]["paths"],
                                                          "results_dir": cfg["data"]["paths"]["results_dir_main"]}}}
    groups = group_runs(load_runs(main_cfg, "A"))
    lam_tag = f"lam{cfg['fusion']['train']['lambda']:g}"
    best = {}
    for n in cfg["fusion"]["normalizers"]:
        ks = [k for k in [f"{n}_fixedgrid", f"{n}_fixed_listwise_{lam_tag}"] +
              [f"{n}_{d}_listwise_{lam_tag}" for d in ("Q", "QS", "S")] if k in groups]
        if ks:
            best[n] = max(np.mean([r["metrics"]["val"]["ndcg@10"] for r in groups[k]]) for k in ks)
    assert best, "MS MARCO protocol A fusion results are needed to fix the primary normaliser"
    choice = max(best, key=best.get)
    save_json({"primary_norm": choice, "val_ndcg@10_by_norm": best,
               "rule": "normaliser whose best config (by val NDCG@10) wins on MS MARCO protocol A val"}, p)
    log.info("primary normaliser fixed to %s (%s)", choice, best)
    return choice


def best_fixed_key(groups, norm, lam_tag):
    """Best fixed config under a normaliser, chosen on val (grid search vs learned scalar)."""
    cands = [k for k in (f"{norm}_fixedgrid", f"{norm}_fixed_listwise_{lam_tag}") if k in groups]
    return max(cands, key=lambda k: np.mean([r["metrics"]["val"]["ndcg@10"] for r in groups[k]]))


def significance(cfg, protocol, groups):
    ec = cfg["fusion"]["eval"]
    lam_tag = f"lam{cfg['fusion']['train']['lambda']:g}"
    rows = []
    for norm in cfg["fusion"]["normalizers"]:
        if f"{norm}_fixedgrid" not in groups:
            continue
        base = best_fixed_key(groups, norm, lam_tag)
        for dep in ("Q", "QS", "S", "QSC"):
            for lt in sorted({lam_tag} | {f"lam{l:g}" for l in cfg["fusion"]["train"]["lambda_extra"]}):
                key = f"{norm}_{dep}_listwise_{lt}"
                if key not in groups:
                    continue
                for s in test_sets(cfg, protocol):
                    for m in ("ndcg@10", "mrr@10"):
                        a, b = per_query_mean(groups[key], s, m), per_query_mean(groups[base], s, m)
                        diff, lo, hi, p = paired_bootstrap(a, b, ec["bootstrap_resamples"], ec["bootstrap_seed"])
                        rows.append({"norm": norm, "config": key, "baseline": base, "set": s, "metric": m,
                                     "config_mean": a.mean(), "baseline_mean": b.mean(), "diff": diff,
                                     "ci95_lo": lo, "ci95_hi": hi, "p_value": p, "n_queries": len(a)})
    df = pd.DataFrame(rows)
    df.to_csv(results_path(cfg, protocol, "significance.csv"), index=False, float_format="%.5f")
    return df


def oracle_gap(cfg, protocol, groups):
    lam_tag = f"lam{cfg['fusion']['train']['lambda']:g}"
    rows = []
    for norm in cfg["fusion"]["normalizers"]:
        if f"{norm}_oracle" not in groups:
            continue
        base = best_fixed_key(groups, norm, lam_tag)
        for s in ["val"] + test_sets(cfg, protocol):
            mean = lambda k: np.mean([r["metrics"][s]["ndcg@10"] for r in groups[k]])
            row = {"norm": norm, "set": s, "oracle_ndcg@10": mean(f"{norm}_oracle"), "best_fixed": base,
                   "best_fixed_ndcg@10": mean(base)}
            row["oracle_minus_fixed"] = row["oracle_ndcg@10"] - row["best_fixed_ndcg@10"]
            for dep in ("Q", "QS", "S"):
                k = f"{norm}_{dep}_listwise_{lam_tag}"
                if k in groups:
                    row[f"{dep}_ndcg@10"] = mean(k)
                    row[f"{dep}_share_of_headroom"] = (mean(k) - row["best_fixed_ndcg@10"]) / max(
                        row["oracle_minus_fixed"], 1e-12)
            row["small_headroom_flag"] = row["oracle_minus_fixed"] < cfg["fusion"]["eval"]["oracle_flag_points"]
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(results_path(cfg, protocol, "oracle_gap.csv"), index=False, float_format="%.4f")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="A")
    args, _ = ap.parse_known_args(argv)
    cfg = load_config()
    set_env(cfg)
    groups = group_runs(load_runs(cfg, args.protocol))
    mt = main_table(cfg, args.protocol, groups)
    sig = significance(cfg, args.protocol, groups)
    og = oracle_gap(cfg, args.protocol, groups)
    log.info("main_table: %d rows; significance: %d rows; oracle_gap: %d rows", len(mt), len(sig), len(og))


if __name__ == "__main__":
    main()
