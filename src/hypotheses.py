"""E3 hypothesis tests (multi-hop plan Sec. 0 and 4.3), bucket tables and cross-dataset figures.

All tests use the primary normaliser (results/primary_norm.json), the default lambda, test queries only, and
per-query values averaged over seeds. "best adaptive" = the variant among Q/QS/S with the best val NDCG@10;
"best fixed" = the better of grid-search and learned-scalar fixed alpha on val (same as significance.csv).
 H1  alpha decreases with complexity: Spearman(alpha, hops) on MuSiQue; mean alpha bridge - comparison on
     HotpotQA / BEIR HotpotQA. Predicted < 0.
 H2  adaptive gain (best adaptive - best fixed NDCG@10) is larger than on MS MARCO. Predicted > 0.
 H3  oracle headroom (oracle - best fixed NDCG@10) is larger than on MS MARCO. Predicted > 0.
 H4  control: H1-H3 statistics repeated for graph types none / random / title_link / hybrid (exploratory).
H1-H3 on the primary datasets are Holm-corrected together; everything else is labelled exploratory.

Usage:  python -m src.hypotheses
Writes: results/hypotheses.csv, results/bucket_tables/{dataset}_{label}.csv, results/figures_multihop/*,
        NOTES.md hypotheses section (full profile only)
"""
import re

import numpy as np
import pandas as pd

from src.common import ROOT, get_logger, load_config, load_json, path, results_path, set_env
from src.evaluate import best_fixed_key, group_runs, load_runs, primary_norm
from src.query_meta import query_meta
from src.stats import bootstrap, diff_of_means_ci, group_diff, holm, mean_ci, spearman

log = get_logger("hypotheses")
VARIANTS = ("Q", "QS", "S")
BRIDGE, COMPARISON = ["bridge"], ["comparison"]


# ------------------------------------------------------------------ per-query data of one key
def load_key(cfg, key, pn, lt):
    """Seed-averaged per-query test arrays for one experiment key, or None if its runs are missing."""
    if not results_path(cfg, key, "runs").exists():
        return None
    g = group_runs(load_runs(cfg, key))
    need = [f"{pn}_oracle", f"{pn}_fixedgrid"]
    if any(k not in g for k in need):
        return None
    avg = lambda k, m: np.mean([np.load(r["arrays"])[f"test__{m}"].astype(np.float64) for r in g[k]], axis=0)
    val = lambda k: np.mean([r["metrics"]["val"]["ndcg@10"] for r in g[k]])
    out = {"key": key, "fixed_key": best_fixed_key(g, pn, lt)}
    out["fixed"] = avg(out["fixed_key"], "ndcg@10")
    out["oracle"] = avg(f"{pn}_oracle", "ndcg@10")
    out["oracle_alpha"] = avg(f"{pn}_oracle", "alpha")
    out["alpha0"] = g[f"{pn}_fixedgrid"][0]["alpha0"]
    for ref in ("ce", "gat"):
        if f"ref_{ref}" in g:
            out[ref] = avg(f"ref_{ref}", "ndcg@10")
    avail = [v for v in VARIANTS if f"{pn}_{v}_listwise_{lt}" in g]
    for v in avail:
        out[f"alpha_{v}"] = avg(f"{pn}_{v}_listwise_{lt}", "alpha")
        out[f"ndcg_{v}"] = avg(f"{pn}_{v}_listwise_{lt}", "ndcg@10")
    if not avail:
        return None
    out["best"] = max(avail, key=lambda v: val(f"{pn}_{v}_listwise_{lt}"))
    out["adaptive"], out["alpha"] = out[f"ndcg_{out['best']}"], out[f"alpha_{out['best']}"]
    return out


def labels_for(cfg, key, split="test"):
    """Per-query complexity labels: dataset meta (hops/type/level) + Step 0 proxies, in run (split) order."""
    from src.common import base_key, data_cache_path, is_msmarco
    if is_msmarco(key):
        qids = load_json(path(cfg, "data_dir", "splits_A.json"))[split]
        meta = {q: {} for q in qids}
    else:
        qids = load_json(data_cache_path(cfg, "splits.json", key))[split]
        meta = load_json(data_cache_path(cfg, "meta.json", key))
    qm = query_meta(cfg, base_key(key))
    lab = {}
    for f in ("hops", "type", "level"):
        vals = [meta[q].get(f) for q in qids]
        if any(v is not None for v in vals):
            lab[f] = np.array([str(v) for v in vals])
    nw = np.array([qm[q]["n_words"] for q in qids])
    ne = np.array([qm[q]["n_entities"] for q in qids])
    lab["n_words"] = np.where(nw <= 4, "short (<=4)", np.where(nw <= 7, "medium (5-7)", "long (>=8)"))
    lab["wh"] = np.array([qm[q]["wh"] for q in qids])
    lab["numeric"] = np.array(["numeric" if qm[q]["numeric"] else "non-numeric" for q in qids])
    lab["n_entities"] = np.where(ne == 0, "0", np.where(ne == 1, "1", "2+"))
    return lab


# ------------------------------------------------------------------ tests
def h1(d, lab, B, which="alpha"):
    """Returns (statistic name, value, lo, hi, p) or None if the dataset has no usable complexity label."""
    a = d[which]
    if "hops" in lab and lab["hops"][0] != "None":
        hops = lab["hops"].astype(float)
        return ("spearman(alpha, hops)", *bootstrap(spearman, [a, hops], B))
    if "type" in lab and set(BRIDGE) <= set(lab["type"]) and set(COMPARISON) <= set(lab["type"]):
        code = np.where(lab["type"] == "bridge", 1.0, np.where(lab["type"] == "comparison", 0.0, -1.0))
        return ("mean alpha bridge - comparison", *bootstrap(group_diff([1.0], [0.0]), [a, code], B))
    return None


def run(cfg):
    hc, B = cfg["multihop"]["hypotheses"], cfg["multihop"]["bootstrap_resamples"]
    pn = primary_norm(cfg)
    lt = f"lam{cfg['fusion']['train']['lambda']:g}"
    ref = load_key(cfg, hc["reference"], pn, lt)
    assert ref is not None, "MS MARCO protocol A runs with the primary normaliser are required"
    primary = set(hc.get("primary_datasets", hc["datasets"]))
    rows = []

    def add(h, ds, graph, variant, stat, v, lo, hi, p, predicted, prim):
        verdict = "inconclusive"
        if np.isfinite(lo) and np.isfinite(hi):
            if (predicted == "<0" and hi < 0) or (predicted == ">0" and lo > 0):
                verdict = "in predicted direction"
            elif (predicted == "<0" and lo > 0) or (predicted == ">0" and hi < 0):
                verdict = "opposite direction"
        rows.append({"hypothesis": h, "dataset": ds, "graph": graph, "variant": variant, "statistic": stat,
                     "value": v, "ci95_lo": lo, "ci95_hi": hi, "p_raw": p, "predicted": predicted,
                     "primary": prim, "ci_verdict": verdict})

    def tests(key, ds, graph, prim):
        d = load_key(cfg, key, pn, lt)
        if d is None:
            return None
        lab = labels_for(cfg, key)
        r = h1(d, lab, B)
        if r is not None:
            add("H1", ds, graph, f"best={d['best']}", r[0], *r[1:], "<0", prim)
            for v in VARIANTS:
                if f"alpha_{v}" in d:
                    rv = h1(d, lab, B, f"alpha_{v}")
                    add("H1", ds, graph, v, rv[0], *rv[1:], "<0", False)
            ro = h1(d, lab, B, "oracle_alpha")
            add("H1", ds, graph, "oracle_alpha", ro[0], *ro[1:], "<0", False)
        g_ds, g_ref = d["adaptive"] - d["fixed"], ref["adaptive"] - ref["fixed"]
        add("H2", ds, graph, f"best={d['best']}", "gain(ds) - gain(MS MARCO)", *diff_of_means_ci(g_ds, g_ref, B),
            ">0", prim)
        add("H2", ds, graph, f"best={d['best']}", "gain(ds) = adaptive - best fixed", *mean_ci(g_ds, B), ">0", False)
        h_ds, h_ref = d["oracle"] - d["fixed"], ref["oracle"] - ref["fixed"]
        add("H3", ds, graph, "oracle", "headroom(ds) - headroom(MS MARCO)", *diff_of_means_ci(h_ds, h_ref, B),
            ">0", prim)
        add("H3", ds, graph, "oracle", "headroom(ds) = oracle - best fixed", *mean_ci(h_ds, B), ">0", False)
        return d, lab

    data = {}
    for ds in hc["datasets"]:
        out = tests(ds, ds, "knn_emb", ds in primary)
        if out is not None:
            data[ds] = out
    # MS MARCO reference numbers (gain / headroom on their own)
    add("H2", "A", "knn_emb", f"best={ref['best']}", "gain(MS MARCO) = adaptive - best fixed",
        *mean_ci(ref["adaptive"] - ref["fixed"], B), ">0", False)
    add("H3", "A", "knn_emb", "oracle", "headroom(MS MARCO) = oracle - best fixed",
        *mean_ci(ref["oracle"] - ref["fixed"], B), ">0", False)
    # H4: graph controls (exploratory)
    for ds in cfg["multihop"]["h4_datasets"]:
        for gt in cfg["multihop"]["graph_types"]:
            if gt != "knn_emb":
                tests(f"{ds}@{gt}", ds, gt, False)
    df = pd.DataFrame(rows)
    if len(df):
        df.loc[df.hypothesis.isin(["H1", "H2", "H3"]) & ~df.primary, "hypothesis"] = \
            df.loc[df.hypothesis.isin(["H1", "H2", "H3"]) & ~df.primary, "hypothesis"] + " (exploratory)"
        df.loc[df.graph != "knn_emb", "hypothesis"] = "H4 (" + df.loc[df.graph != "knn_emb", "hypothesis"] + ")"
        df["p_holm"] = np.nan
        pm = df.primary.values
        df.loc[pm, "p_holm"] = holm(df.loc[pm, "p_raw"].values)
        df["verdict"] = df.ci_verdict
        sig = pm & (df.p_holm < 0.05)
        df.loc[pm & ~sig & (df.ci_verdict != "inconclusive"), "verdict"] = "inconclusive after Holm"
        df.loc[sig & (df.ci_verdict == "in predicted direction"), "verdict"] = "supported"
        df.loc[sig & (df.ci_verdict == "opposite direction"), "verdict"] = "contradicted"
    df.to_csv(path(cfg, "results_dir", "hypotheses.csv"), index=False, float_format="%.5f")
    return df, data, ref


# ------------------------------------------------------------------ bucket tables
def bucket_tables(cfg, data, ref):
    B = cfg["multihop"]["bootstrap_resamples"]
    out_dir = path(cfg, "results_dir", "bucket_tables", "x").parent
    out_dir.mkdir(parents=True, exist_ok=True)
    allrows = []
    for ds, (d, lab) in {**data, "A": (ref, labels_for(cfg, "A"))}.items():
        for name, lv in lab.items():
            rows = []
            for b in sorted(set(lv), key=lambda x: (len(x), x)):
                sel = lv == b
                gain = d["adaptive"][sel] - d["fixed"][sel]
                m, lo, hi, p = mean_ci(gain, B)
                row = {"dataset": ds, "label": name, "bucket": b, "n": int(sel.sum())}
                for v in VARIANTS:
                    if f"alpha_{v}" in d:
                        row[f"alpha_{v}"] = d[f"alpha_{v}"][sel].mean()
                row.update({"best_adaptive": d["best"], "alpha_best_minus_alpha0": d["alpha"][sel].mean() - d["alpha0"],
                            "oracle_alpha": d["oracle_alpha"][sel].mean(), "ce_ndcg@10": d.get("ce", np.nan)[sel].mean()
                            if "ce" in d else np.nan, "gat_ndcg@10": d["gat"][sel].mean() if "gat" in d else np.nan,
                            "best_fixed_ndcg@10": d["fixed"][sel].mean(), "adaptive_ndcg@10": d["adaptive"][sel].mean(),
                            "oracle_ndcg@10": d["oracle"][sel].mean(), "gain": m, "gain_ci95_lo": lo,
                            "gain_ci95_hi": hi, "gain_p": p})
                rows.append(row)
            t = pd.DataFrame(rows)
            t.to_csv(out_dir / f"{ds}_{name}.csv", index=False, float_format="%.5f")
            allrows += rows
    return pd.DataFrame(allrows)


# ------------------------------------------------------------------ figures
def figures(cfg, data, ref, hyp, buckets):
    from src.analysis import INK2, SERIES, SURFACE, plt
    out = path(cfg, "results_dir", "figures_multihop", "x").parent
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    colors = {**SERIES, "oracle_alpha": "#6250d6"}
    # alpha / oracle alpha by complexity label (box plots), one row per dataset
    panels = [(ds, "hops" if "hops" in lab else "type") for ds, (d, lab) in data.items()
              if "hops" in lab or "type" in lab]
    if panels:
        fig, axes = plt.subplots(len(panels), 4, figsize=(13, 2.6 * len(panels)), squeeze=False)
        for r, (ds, lname) in enumerate(panels):
            d, lab = data[ds]
            lv = lab[lname]
            cats = sorted(set(lv), key=lambda x: (len(x), x))
            for c, name in enumerate(["alpha_Q", "alpha_QS", "alpha_S", "oracle_alpha"]):
                ax = axes[r, c]
                if name in d:
                    v = name.replace("alpha_", "") if name != "oracle_alpha" else name
                    ax.boxplot([d[name][lv == k] for k in cats], tick_labels=cats, showfliers=False,
                               patch_artist=True, boxprops={"facecolor": colors[v], "alpha": 0.6, "edgecolor": INK2},
                               medianprops={"color": INK2})
                    ax.axhline(d["alpha0"], color=INK2, lw=1, ls="--")
                ax.set_title(f"{ds}: {name.replace('_', ' ')}", loc="left")
                ax.tick_params(axis="x", rotation=30)
                if c == 0:
                    ax.set_ylabel(f"alpha by {lname}")
        fig.tight_layout()
        save(fig, "alpha_vs_complexity_box")
    # adaptive gain per bucket with CIs
    sub = buckets[buckets.label.isin(["hops", "type"])]
    if len(sub):
        fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(sub)), 3.2))
        xs = np.arange(len(sub))
        ax.bar(xs, sub["gain"], color=[SERIES["QS"] if l == "hops" else SERIES["Q"] for l in sub.label],
               edgecolor=SURFACE)
        ax.errorbar(xs, sub["gain"], yerr=[sub["gain"] - sub["gain_ci95_lo"], sub["gain_ci95_hi"] - sub["gain"]],
                    fmt="none", ecolor=INK2, lw=1)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(xs, [f"{a}\n{b}" for a, b in zip(sub.dataset, sub.bucket)], fontsize=7)
        ax.set_ylabel("adaptive - best fixed NDCG@10")
        ax.set_title("Adaptive gain per complexity bucket (test, 95% CI)", loc="left")
        save(fig, "bucket_gain_bars")
    # graph-type comparison (H4)
    h4 = hyp[hyp.statistic.isin(["gain(ds) = adaptive - best fixed", "headroom(ds) = oracle - best fixed"])]
    if len(h4) and h4.graph.nunique() > 1:
        dss = list(dict.fromkeys(h4.dataset))
        fig, axes = plt.subplots(1, len(dss), figsize=(5 * len(dss), 3), squeeze=False)
        for ax, ds in zip(axes[0], dss):
            s = h4[h4.dataset == ds]
            gts = list(dict.fromkeys(s.graph))
            for i, (st, col) in enumerate((("gain(ds) = adaptive - best fixed", SERIES["QS"]),
                                           ("headroom(ds) = oracle - best fixed", SERIES["fixed"]))):
                t = s[s.statistic == st].set_index("graph").reindex(gts)
                x = np.arange(len(gts)) + (i - 0.5) * 0.38
                ax.bar(x, t.value, width=0.36, color=col, edgecolor=SURFACE, label=st.split(" =")[0])
                ax.errorbar(x, t.value, yerr=[t.value - t.ci95_lo, t.ci95_hi - t.value], fmt="none", ecolor=INK2, lw=1)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xticks(range(len(gts)), gts, rotation=20)
            ax.set_title(f"{ds}: by graph type (H4)", loc="left")
            ax.set_ylabel("NDCG@10 difference")
        axes[0][-1].legend(fontsize=7)
        fig.tight_layout()
        save(fig, "graph_type_comparison")


def notes_block(cfg, hyp):
    if cfg["profile"] != "full":
        return
    lines = ["Auto-generated by `src/hypotheses.py` (primary normaliser "
             f"`{primary_norm(cfg)}`, test split, Holm over the primary H1-H3 tests).", ""]
    for _, r in hyp[hyp.primary].iterrows():
        lines.append(f"- **{r.hypothesis} / {r.dataset}** ({r.statistic}, {r.variant}): {r.value:+.4f} "
                     f"[{r.ci95_lo:+.4f}, {r.ci95_hi:+.4f}], p={r.p_raw:.3g}, Holm p={r.p_holm:.3g} -> **{r.verdict}**")
    h4 = hyp[hyp.hypothesis.str.startswith("H4")]
    if len(h4):
        lines.append("- **H4 (exploratory controls):** see `results/hypotheses.csv` rows `H4 (...)`; the effects "
                     "should vanish for `none` / `random` if they come from the graph.")
    text = "\n".join(lines) + "\n"
    notes = ROOT / "NOTES.md"
    s = notes.read_text()
    b, e = "<!-- HYPOTHESES:BEGIN -->", "<!-- HYPOTHESES:END -->"
    if b in s:
        notes.write_text(re.sub(re.escape(b) + ".*?" + re.escape(e), b + "\n" + text + e, s, flags=re.S))


def main():
    cfg = load_config()
    set_env(cfg)
    hyp, data, ref = run(cfg)
    bk = bucket_tables(cfg, data, ref)
    figures(cfg, data, ref, hyp, bk)
    notes_block(cfg, hyp)
    cols = ["hypothesis", "dataset", "graph", "variant", "statistic", "value", "ci95_lo", "ci95_hi", "p_raw",
            "p_holm", "verdict"]
    log.info("hypotheses:\n%s", hyp[cols].to_string(index=False) if len(hyp) else "(no results yet)")


if __name__ == "__main__":
    main()
