"""Step 0 (multi-hop plan Sec. 1): is "alpha > 0.5 because MS MARCO queries are simple" supported?

Reads the existing MS MARCO protocol A caches and runs (no new data):
 1. signal strength: CE-only vs GAT-only NDCG@10 / MRR@10 on val and test
 2. normalisation sensitivity: best fixed alpha under each normaliser
 3. does alpha adapt: std of the learned alpha over test queries, per variant / normaliser / seed
 4. regulariser and init: mean alpha with lambda = 0.01 vs 0, and mean alpha - alpha0
 5. alpha vs query properties (n_words, wh-word, numeric, #entities): mean learned and oracle alpha per bucket
    with bootstrap CIs, plus Spearman(alpha, n_words / n_entities)
 6. graph controls: no_graph / random_graph vs knn_emb on val (results/A/gat_controls.csv)
Decision rule: supported only if alpha (or oracle alpha) varies with the properties in the predicted direction
(more complex -> lower alpha), alpha adapts at all (std >= 0.05), and CE-only is not simply much stronger.

Usage:  python -m src.diagnose_msmarco
Writes: results/diagnostics_msmarco.csv, results/diagnostics_msmarco.json, results/A/figures/diag_*.png|pdf,
        results/primary_norm.json (the primary normaliser for all multi-hop tests, fixed here)
"""
import numpy as np
import pandas as pd

from src.common import get_logger, load_config, load_json, path, results_path, save_json, set_env
from src.evaluate import group_runs, load_runs, primary_norm
from src.query_meta import query_meta
from src.stats import bootstrap, mean_ci, spearman

log = get_logger("diagnose")
VARIANTS = ("Q", "QS", "S")


def seed_avg(group, s, k):
    return np.mean([np.load(r["arrays"])[f"{s}__{k}"].astype(np.float64) for r in group], axis=0)


def buckets(qm, qids):
    nw = np.array([qm[q]["n_words"] for q in qids])
    ne = np.array([qm[q]["n_entities"] for q in qids])
    return {
        "n_words": np.where(nw <= 4, "short (<=4)", np.where(nw <= 7, "medium (5-7)", "long (>=8)")),
        "wh": np.array([qm[q]["wh"] for q in qids]),
        "numeric": np.array(["numeric" if qm[q]["numeric"] else "non-numeric" for q in qids]),
        "n_entities": np.where(ne == 0, "0", np.where(ne == 1, "1", "2+")),
    }, nw, ne


def main():
    cfg = load_config()
    set_env(cfg)
    B, dc = cfg["multihop"]["bootstrap_resamples"], cfg["multihop"].get("diagnose", {})
    groups = group_runs(load_runs(cfg, "A"))
    lam = cfg["fusion"]["train"]["lambda"]
    lt = f"lam{lam:g}"
    pn = primary_norm(cfg)
    rows, dec = [], {"primary_norm": pn}
    add = lambda **r: rows.append(r)

    # 1. signal strength
    for s in ("val", "test"):
        for ref in ("ce", "gat", "bi", "bm25"):
            m = groups[f"ref_{ref}"][0]["metrics"][s]
            for k in ("ndcg@10", "mrr@10"):
                add(item="1_signal_strength", variant=ref, norm="-", bucket=s, stat=k, value=m[k])
    ce_gap = groups["ref_ce"][0]["metrics"]["val"]["ndcg@10"] - groups["ref_gat"][0]["metrics"]["val"]["ndcg@10"]
    dec["ce_minus_gat_val_ndcg@10"] = ce_gap
    dec["ce_much_stronger"] = bool(ce_gap > dc.get("ce_dominance_gap", 0.05))

    # 2. normalisation sensitivity
    a0 = load_json(results_path(cfg, "A", "alpha0.json"))
    for n, v in a0.items():
        add(item="2_alpha0_by_norm", variant="fixedgrid", norm=n, bucket="val", stat="alpha0", value=v["alpha0"])
    dec["alpha0_by_norm"] = {n: v["alpha0"] for n, v in a0.items()}
    dec["alpha0_range"] = float(max(dec["alpha0_by_norm"].values()) - min(dec["alpha0_by_norm"].values()))

    # 3./4. adaptation, regulariser, init
    stds = {}
    for n in cfg["fusion"]["normalizers"]:
        for v in VARIANTS:
            for tag in sorted({lt} | {f"lam{x:g}" for x in cfg["fusion"]["train"]["lambda_extra"]}):
                g = groups.get(f"{n}_{v}_listwise_{tag}")
                if not g:
                    continue
                for r in g:
                    a = np.load(r["arrays"])["test__alpha"]
                    add(item="3_alpha_std", variant=v, norm=n, bucket=tag, stat=f"std_seed{r['seed']}", value=float(a.std()))
                    add(item="4_alpha_mean", variant=v, norm=n, bucket=tag, stat=f"mean_seed{r['seed']}", value=float(a.mean()))
                    add(item="4_alpha_minus_alpha0", variant=v, norm=n, bucket=tag, stat=f"seed{r['seed']}",
                        value=float(a.mean() - r["alpha0"]))
                if tag == lt:
                    stds[(n, v)] = float(np.mean([np.load(r["arrays"])["test__alpha"].std() for r in g]))
    thr = dc.get("min_alpha_std", 0.05)
    dec["alpha_std_primary"] = {v: stds.get((pn, v)) for v in VARIANTS}
    dec["alpha_adapts"] = bool(any(x is not None and x >= thr for x in dec["alpha_std_primary"].values()))

    # 5. alpha vs query properties (primary normaliser; seed-averaged per query)
    test_qids = load_json(path(cfg, "data_dir", "splits_A.json"))["test"]
    qm = query_meta(cfg, "A")
    bk, nw, ne = buckets(qm, test_qids)
    series = {v: seed_avg(groups[f"{pn}_{v}_listwise_{lt}"], "test", "alpha") for v in VARIANTS
              if f"{pn}_{v}_listwise_{lt}" in groups}
    series["oracle"] = seed_avg(groups[f"{pn}_oracle"], "test", "alpha")
    direction = {}
    for name, a in series.items():
        for prop, lab in bk.items():
            for b in sorted(set(lab)):
                sel = a[lab == b]
                m, lo, hi, _ = mean_ci(sel, B)
                add(item="5_alpha_by_property", variant=name, norm=pn, bucket=f"{prop}={b}", stat="mean_alpha",
                    value=m, ci_lo=lo, ci_hi=hi, n=int(len(sel)))
        for prop, x in (("n_words", nw), ("n_entities", ne)):
            rho, lo, hi, p = bootstrap(spearman, [a, x], B)
            add(item="5_spearman", variant=name, norm=pn, bucket=prop, stat="spearman_rho", value=rho, ci_lo=lo,
                ci_hi=hi, p=p, n=int(len(a)))
            direction[f"{name}:{prop}"] = {"rho": rho, "ci": [lo, hi], "predicted_negative": bool(hi < 0)}
    dec["alpha_vs_properties"] = direction
    dec["properties_support"] = bool(any(v["predicted_negative"] for v in direction.values()))

    # 6. graph controls
    gc = results_path(cfg, "A", "gat_controls.csv")
    if gc.exists():
        g = pd.read_csv(gc).set_index("control")
        for c in g.index:
            add(item="6_graph_controls", variant=c, norm="-", bucket="val", stat="ndcg@10", value=float(g.loc[c, "val_ndcg@10"]))
        knn = float(g.loc["knn_k8_s0.3", "val_ndcg@10"]) if "knn_k8_s0.3" in g.index else None
        dec["graph_controls_val_ndcg@10"] = {c: float(g.loc[c, "val_ndcg@10"]) for c in g.index}
        if knn is not None and "random_graph" in g.index:
            dec["knn_minus_random"] = knn - float(g.loc["random_graph", "val_ndcg@10"])
            dec["knn_minus_no_graph"] = knn - float(g.loc["no_graph", "val_ndcg@10"])

    supported = dec["properties_support"] and dec["alpha_adapts"] and not dec["ce_much_stronger"]
    dec["simple_queries_explanation"] = "supported" if supported else "not supported on MS MARCO"
    dec["rule"] = ("supported only if alpha or oracle alpha decreases with query complexity (Spearman CI < 0 for "
                   "n_words or n_entities), the learned alpha adapts (test std >= min_alpha_std for some variant) "
                   "and CE-only is not much stronger than GAT-only (val NDCG@10 gap <= ce_dominance_gap)")
    df = pd.DataFrame(rows)
    df.to_csv(path(cfg, "results_dir", "diagnostics_msmarco.csv"), index=False, float_format="%.5f")
    save_json(dec, path(cfg, "results_dir", "diagnostics_msmarco.json"))
    plot(cfg, df, pn)
    log.info("Step 0 decision: %s", {k: dec[k] for k in ("simple_queries_explanation", "ce_minus_gat_val_ndcg@10",
                                                         "alpha_adapts", "properties_support", "primary_norm")})


def plot(cfg, df, pn):
    from src.analysis import SERIES, _save, plt
    d = df[df.item == "5_alpha_by_property"]
    props = ["n_words", "wh", "numeric", "n_entities"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0), sharey=True)
    colors = {**SERIES, "oracle": "#6250d6"}
    for ax, prop in zip(axes, props):
        sub = d[d.bucket.str.startswith(prop + "=")]
        bks = list(dict.fromkeys(sub.bucket))
        names = list(dict.fromkeys(sub.variant))
        w = 0.8 / max(len(names), 1)
        for i, name in enumerate(names):
            s = sub[sub.variant == name].set_index("bucket").reindex(bks)
            x = np.arange(len(bks)) + (i - (len(names) - 1) / 2) * w
            ax.errorbar(x, s.value, yerr=[s.value - s.ci_lo, s.ci_hi - s.value], fmt="o", ms=4, lw=1,
                        color=colors.get(name, "#52514e"), label=name)
        ax.set_xticks(range(len(bks)), [b.split("=", 1)[1] for b in bks], rotation=30, ha="right")
        ax.set_title(prop, loc="left")
    axes[0].set_ylabel(f"mean alpha on test (norm = {pn})")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("MS MARCO: learned and oracle alpha by query property (95% bootstrap CI)", x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, cfg, "A", "diag_alpha_by_property")


if __name__ == "__main__":
    main()
