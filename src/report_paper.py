"""Builds the LaTeX tables and figures of the paper (paper/) from results/ (no training, no test-time selection).

Run `python -m src.report_analysis` first (alpha curves, bucket optima). Then:
    python -m src.report_paper
Writes: paper/tables/*.tex, paper/figures/*.pdf, paper/numbers.json (numbers quoted in the text).
"""
import glob
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import ROOT, load_config  # noqa: E402
from src.evaluate import group_runs, paired_bootstrap, per_query_mean  # noqa: E402
from src.stats import spearman  # noqa: E402

RES = ROOT / "results"
OUT = ROOT / "paper"
TAB, FIG = OUT / "tables", OUT / "figures"
PN = "minmax"
LAM = "lam0.01"
KEYS = ["A", "B", "musique", "hotpotqa", "beir_hotpotqa", "twowiki"]
NAME = {"A": "MS MARCO (A)", "B": "MS MARCO (B)", "musique": "MuSiQue", "hotpotqa": "HotpotQA",
        "beir_hotpotqa": "BEIR-HotpotQA", "twowiki": "2Wiki"}
SHORT = {"A": "MSM-A", "B": "MSM-B", "musique": "MuSiQue", "hotpotqa": "HotpotQA", "beir_hotpotqa": "BEIR-HQA",
         "twowiki": "2Wiki"}
# validated reference palette (dataviz skill), fixed slot order
COLOR = {"A": "#2a78d6", "B": "#6250d6", "musique": "#eb6834", "hotpotqa": "#1baf7a", "beir_hotpotqa": "#eda100",
         "twowiki": "#e87ba4"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BUCKET_NAME = {"2": "2-hop", "3": "3-hop", "4": "4-hop", "bridge_comparison": "bridge-comp.",
               "short (<=4)": "short ($\\le$4 w)", "medium (5-7)": "medium (5--7 w)", "long (>=8)": "long ($\\ge$8 w)"}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "lines.linewidth": 1.5, "pdf.fonttype": 42, "font.family": "serif"})

NUM = {}


def tex(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def f100(x, d=2):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.{d}f}"


def save_tab(name, body, fit=False):
    if fit:
        body = body.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}", 1)
        body = body.replace("\\end{tabular}", "\\end{tabular}}", 1)
    (TAB / f"{name}.tex").write_text(body)


def save_fig(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main_table(key):
    return pd.read_csv(RES / key / "main_table.csv").set_index("config")


_G = {}


def _groups(key):
    if key not in _G:
        runs = {}
        for p in sorted((RES / key / "runs").glob("*.json")):
            m = json.loads(p.read_text())
            m["arrays"] = p.with_suffix(".npz")
            runs[m["name"]] = m
        _G[key] = group_runs(runs)
    return _G[key]


def best_fixed(key, norm=PN):
    g = _groups(key)
    c = [k for k in (f"{norm}_fixedgrid", f"{norm}_fixed_listwise_{LAM}") if k in g]
    return max(c, key=lambda k: np.mean([r["metrics"]["val"]["ndcg@10"] for r in g[k]]))


def sig_vs_fixed(key, cfg_key, metric="ndcg@10", norm=PN):
    g = _groups(key)
    if cfg_key not in g:
        return None
    a = per_query_mean(g[cfg_key], "test", metric)
    b = per_query_mean(g[best_fixed(key, norm)], "test", metric)
    return paired_bootstrap(a, b)


def mark(sig):
    if sig is None:
        return ""
    d, lo, hi, p = sig
    if p < 0.05 and lo > 0:
        return "$^{\\blacktriangle}$"
    if p < 0.05 and hi < 0:
        return "$^{\\triangledown}$"
    return ""


# ------------------------------------------------------------------ tables
ROWS_MAIN = [("BM25", "BM25 (first stage / in-pool)"), ("SBERT (s_bi)", "SBERT ($s_{bi}$)"),
             ("CE-BERT (alpha=1)", "CE only ($\\alpha{=}1$)"), ("GAT-only (alpha=0)", "GAT only ($\\alpha{=}0$)"),
             (f"{PN}_paper_bce_lam0", "ReGAT-BERT-style ($a,b$; BCE)"),
             (f"{PN}_fixedgrid", "Fixed $\\alpha$ (grid on val)"),
             (f"{PN}_fixed_listwise_{LAM}", "Fixed $\\alpha$ (learned)"),
             (f"{PN}_Q_listwise_{LAM}", "Adaptive $\\alpha(q)$ \\textsc{Q}"),
             (f"{PN}_QS_listwise_{LAM}", "Adaptive $\\alpha(q,s)$ \\textsc{QS}"),
             (f"{PN}_S_listwise_{LAM}", "Adaptive $\\alpha(s)$ \\textsc{S}"),
             (f"{PN}_QSC_listwise_{LAM}", "Adaptive \\textsc{QS+C}"),
             (f"{PN}_oracle (UPPER BOUND, not a method)", "\\textit{Oracle per-query $\\alpha$ (bound)}")]


def row_key(t, k):
    if k == "BM25":
        return [c for c in t.index if c.startswith("BM25")][0]
    return k if k in t.index else None


def tab_main():
    cols = KEYS
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{SHORT[k]}}}" for k in cols)
    cm = " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(cols)))
    sub = " & ".join(["nDCG & MRR"] * len(cols))
    lines = []
    T = {k: main_table(k) for k in cols}
    for ck, label in ROWS_MAIN:
        cells = []
        for k in cols:
            rk = row_key(T[k], ck)
            for m in ("ndcg@10", "mrr@10"):
                if rk is None:
                    cells.append("--")
                    continue
                v = T[k].loc[rk, f"test_{m}_mean"]
                s = ""
                if "_Q" in ck or "_S_" in ck or "paper" in ck:
                    s = mark(sig_vs_fixed(k, ck, m))
                cells.append(f100(v) + s)
        if ck.startswith(f"{PN}_paper") or ck.startswith(f"{PN}_Q_") or ck.startswith(f"{PN}_oracle"):
            lines.append("\\midrule")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    body = (
        "\\begin{table*}[t]\n\\centering\\small\\setlength{\\tabcolsep}{3.2pt}\n"
        f"\\begin{{tabular}}{{l{'rr' * len(cols)}}}\n\\toprule\n & {head} \\\\\n{cm}\n & {sub} \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Main results on the test sets (nDCG@10 and MRR@10, $\\times 100$, mean over 5 seeds) under the "
        "primary normaliser (min--max, fixed on MS MARCO val). $^{\\blacktriangle}$/$^{\\triangledown}$: "
        "significantly better/worse than the best fixed $\\alpha$ (paired bootstrap over queries, 10k resamples, "
        "$p<0.05$, uncorrected). The oracle row picks $\\alpha$ per query on the test labels and is an upper bound, "
        "not a method. \\gate{QS+C} was added with the multi-hop extension and not run on MS MARCO (--). Full metrics at $k\\in\\{1,3,5,10,50\\}$ are in Appendix~\\ref{app:full}.}\n"
        "\\label{tab:main}\n\\end{table*}\n")
    save_tab("main", body, fit=True)


def tab_datasets():
    rows = []
    split_a = json.loads((ROOT / "data" / "splits_A.json").read_text())
    split_b = json.loads((ROOT / "data" / "splits_B.json").read_text())
    for k in KEYS:
        if k in ("A", "B"):
            s = split_a if k == "A" else split_b
            n = [len(s[x]) for x in ("gat_train", "fusion_train", "val", "test")]
            rows.append([NAME[k], "open (BM25)", "100", *[f"{x:,}" for x in n], "--", "proxies only"])
            continue
        ps = json.loads((RES / k / "prep_stats.json").read_text())
        n = [ps["split_sizes"][x] for x in ("gat_train", "fusion_train", "val", "test")]
        K = ps["test"]["mean_K"]
        lab = {"musique": "hops 2/3/4", "hotpotqa": "bridge/comparison", "beir_hotpotqa": "bridge/comparison",
               "twowiki": "4 types"}[k]
        setting = "open (BM25)" if k == "beir_hotpotqa" else "closed pool"
        rows.append([NAME[k], setting, f"{K:.0f}", *[f"{x:,}" for x in n], f"{ps['test']['mean_gold_in_pool']:.2f}",
                     lab])
    body = ("\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{llrrrrrrl}\n\\toprule\n"
            "Dataset & Setting & $K$ & GAT train & Fusion train & Val & Test & Gold/pool & Complexity labels \\\\\n"
            "\\midrule\n" + "\n".join(" & ".join(r) + " \\\\" for r in rows) +
            "\n\\bottomrule\n\\end{tabular}\n\\caption{Datasets and splits (queries). Val is always carved from the "
            "training data; test is the official dev set (MS MARCO A: dev/small; MuSiQue, HotpotQA, 2Wiki: dev; BEIR: "
            "test). MS MARCO B splits dev/small itself, so the cross-encoder never saw any of its queries. Gold/pool = "
            "mean number of relevant passages among the $K$ test candidates.}\n\\label{tab:data}\n\\end{table*}\n")
    save_tab("datasets", body, fit=True)


def alpha_curve_stats():
    cur = pd.read_csv(RES / "report" / "alpha_curves.csv")
    out = {}
    for k in KEYS:
        r = {}
        for norm in ("none", "minmax", "zscore", "rank"):
            v = cur[(cur.key == k) & (cur.norm == norm) & (cur.split == "val")]
            r[f"a0_{norm}"] = float(v.alpha.values[v["ndcg@10"].values.argmax()])
        for split in ("val", "test"):
            v = cur[(cur.key == k) & (cur.norm == PN) & (cur.split == split)]
            r[f"argmax_{split}"] = float(v.alpha.values[v["ndcg@10"].values.argmax()])
            r[f"ce_{split}"] = float(v[v.alpha == 1.0]["ndcg@10"].iloc[0])
            r[f"gat_{split}"] = float(v[v.alpha == 0.0]["ndcg@10"].iloc[0])
            r[f"best_{split}"] = float(v["ndcg@10"].max())
            r[f"half_{split}"] = float(v[np.isclose(v.alpha, 0.5)]["ndcg@10"].iloc[0])
        out[k] = r
    return out


def tab_dataset_alpha(st):
    lines = []
    for k in KEYS:
        r = st[k]
        side = lambda a: "ctx" if a < 0.5 else ("sem" if a > 0.5 else "=")
        lines.append(" & ".join([
            NAME[k], f100(r["ce_test"], 1), f100(r["gat_test"], 1), f"{100 * (r['ce_test'] - r['gat_test']):+.1f}",
            *[f"{r[f'a0_{n}']:.2f}" for n in ("none", "minmax", "zscore", "rank")],
            f"\\textbf{{{r['argmax_val']:.2f}}}", f"{r['argmax_test']:.2f}",
            f"{side(r['argmax_val'])}/{side(r['argmax_test'])}"]) + " \\\\")
    body = ("\\begin{table*}[t]\n\\centering\\small\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular}{lrrrrrrrrrc}\n\\toprule\n"
            " & \\multicolumn{3}{c}{Test nDCG@10 ($\\times$100)} & \\multicolumn{4}{c}{Best fixed $\\alpha$ on val, "
            "by normaliser} & \\multicolumn{2}{c}{min--max $\\alpha^*$} & \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\\cmidrule(lr){9-10}\n"
            "Dataset & CE & GAT & CE$-$GAT & none & min--max & z-score & rank & val & test & favours \\\\\n"
            "\\midrule\n" + "\n".join(lines) +
            "\n\\bottomrule\n\\end{tabular}\n\\caption{Dataset-level test of the hypothesis. $\\alpha$ weights the "
            "cross-encoder: fused $=\\alpha\\, s_{sem} + (1-\\alpha)\\, s_{ctx}$. $\\alpha^*$ is the arg-max of mean "
            "nDCG@10 over the grid $\\{0,0.05,\\dots,1\\}$; the val value (bold) is the one used by the fixed baseline, "
            "the test value is for analysis only. ``favours'': ctx if $\\alpha^*<0.5$, sem if $\\alpha^*>0.5$ "
            "(val/test). With raw scores (none) $\\alpha$ mostly reflects the scale ratio of the two scores "
            "and is not interpretable.}\n\\label{tab:dsalpha}\n\\end{table*}\n")
    save_tab("dataset_alpha", body)


def tab_adaptive_grid():
    lines = []
    for norm in ("none", "minmax", "zscore", "rank"):
        for dep in ("Q", "QS", "S", "QSC"):
            cells, any_ = [], False
            for k in KEYS:
                if f"{norm}_{dep}_listwise_{LAM}" not in _groups(k):
                    cells.append("--")
                    continue
                any_ = True
                sg = sig_vs_fixed(k, f"{norm}_{dep}_listwise_{LAM}", "ndcg@10", norm)
                m = mark(sg)
                cell = f"{100 * sg[0]:+.2f}{m}"
                if "blacktriangle" in m:
                    cell = f"\\textbf{{{cell}}}"
                NUM.setdefault("grid", {})[f"{k}/{norm}/{dep}"] = list(sg)
                cells.append(cell)
            if any_:
                lab = (f"{norm}" if dep == "Q" else "") + f" & \\textsc{{{dep.replace('QSC', 'QS+C')}}}"
                lines.append(lab + " & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines = lines[:-1]
    n_pos = sum(ln.count("blacktriangle") for ln in lines)
    NUM["n_negative_grid_cells"] = sum(ln.count("triangledown") for ln in lines)
    NUM["n_grid_cells"] = len(NUM["grid"])
    body = ("\\begin{table}[t]\n\\centering\\small\\setlength{\\tabcolsep}{6pt}\n"
            f"\\begin{{tabular}}{{ll{'r' * len(KEYS)}}}\n\\toprule\n"
            "Norm. & Gate & " + " & ".join(SHORT[k] for k in KEYS) + " \\\\\n\\midrule\n" + "\n".join(lines) +
            "\n\\bottomrule\n\\end{tabular}\n\\caption{Every adaptive gate against the best fixed $\\alpha$ under the same "
            "normaliser: test $\\Delta$nDCG@10 $\\times$100 (paired bootstrap; $^{\\blacktriangle}$ significantly "
            "better, $^{\\triangledown}$ worse, $p<0.05$ uncorrected).}\n\\label{tab:grid}\n\\end{table}\n")
    save_tab("adaptive_grid", body.replace("{table}", "{table*}"))
    return n_pos


def gate_alpha():
    """Per key and variant: seed-averaged per-query alpha on test, and share of runs restored at epoch 0."""
    out = {}
    for k in KEYS:
        g = _groups(k)
        for dep in ("Q", "QS", "S", "QSC"):
            ck = f"{PN}_{dep}_listwise_{LAM}"
            if ck not in g:
                continue
            a = per_query_mean(g[ck], "test", "alpha")
            out[(k, dep)] = {"alpha": a, "mean": float(a.mean()), "std": float(a.std()),
                             "ep0": float(np.mean([r["best_epoch"] == 0 for r in g[ck]])),
                             "alpha0": float(g[ck][0]["alpha0"])}
    return out


def tab_gate(ga):
    tr = pd.read_csv(RES / "transfer.csv")
    lines = []
    for k in KEYS:
        cells = [NAME[k], f"{ga[(k, 'Q')]['alpha0']:.2f}"]
        for dep in ("Q", "QS", "S", "QSC"):
            d = ga.get((k, dep))
            cells.append("--" if d is None else f"{d['mean']:.3f} ({d['std']:.3f})")
        eps = [ga[(k, d)]["ep0"] for d in ("Q", "QS", "S", "QSC") if (k, d) in ga]
        cells.append(f"{100 * np.mean(eps):.0f}\\%")
        mx = tr[(tr.setting == "mixed") & (tr.test == k) & (tr.variant == "Q")]
        cells.append("--" if not len(mx) else f"{mx.alpha_mean.iloc[0]:.2f} ({mx.alpha_std_over_queries.iloc[0]:.2f})")
        lines.append(" & ".join(cells) + " \\\\")
    body = ("\\begin{table*}[t]\n\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
            " & & \\multicolumn{4}{c}{In-domain gate: mean (sd over test queries) of learned $\\alpha$} & Restored & Mixed gate \\\\\n"
            "\\cmidrule(lr){3-6}\nDataset & $\\alpha_0$ & \\textsc{Q} & \\textsc{QS} & \\textsc{S} & \\textsc{QS+C} & "
            "at init & \\textsc{Q} \\\\\n\\midrule\n" + "\n".join(lines) +
            "\n\\bottomrule\n\\end{tabular}\n\\caption{What the gates learn (min--max, $\\lambda=0.01$). The gates are "
            "initialised at $\\alpha_0$ and early-stopped on val nDCG@10; ``restored at init'' is the share of "
            "adaptive runs (all gates, 5 seeds) whose best val epoch was the initialisation, i.e.\\ the returned model "
            "is the fixed baseline. Mixed gate: one \\textsc{Q} gate trained on MS MARCO + MuSiQue + HotpotQA "
            "(\\S\\ref{sec:transfer}), its $\\alpha$ on each test set.}\n\\label{tab:gate}\n\\end{table*}\n")
    save_tab("gate", body)


def tab_buckets(ga):
    ba = pd.read_csv(RES / "report" / "bucket_alpha.csv")
    bc = pd.read_csv(RES / "report" / "bucket_curves.csv")
    spec = [("A", "n_words"), ("musique", "hops"), ("hotpotqa", "type"), ("beir_hotpotqa", "type"),
            ("twowiki", "type")]
    order = {"n_words": ["short (<=4)", "medium (5-7)", "long (>=8)"], "hops": ["2", "3", "4"],
             "type": ["comparison", "bridge", "bridge_comparison", "compositional", "inference"]}
    lab_meta = {"musique": "hops", "hotpotqa": "type", "beir_hotpotqa": "type", "twowiki": "type"}
    lines = []
    for k, lab in spec:
        g = _groups(k)
        bt = None
        if k in lab_meta:
            bt = pd.read_csv(RES / "bucket_tables" / f"{k}_{lab}.csv")
        first = True
        for b in order[lab]:
            t = ba[(ba.key == k) & (ba.label == lab) & (ba.bucket.astype(str) == b) & (ba.split == "test")]
            v = ba[(ba.key == k) & (ba.label == lab) & (ba.bucket.astype(str) == b) & (ba.split == "val")]
            if not len(t):
                continue
            t, v = t.iloc[0], (v.iloc[0] if len(v) else None)
            c = bc[(bc.key == k) & (bc.label == lab) & (bc.bucket.astype(str) == b) & (bc.split == "test")]
            ce = c[c.alpha == 1.0]["ndcg@10"].iloc[0]
            gat = c[c.alpha == 0.0]["ndcg@10"].iloc[0]
            gq = "--"
            if bt is not None:
                r = bt[bt.bucket.astype(str) == b]
                if len(r):
                    best = r.best_adaptive.iloc[0]
                    gq = f"{r[f'alpha_{best}'].iloc[0]:.2f}"
            name = BUCKET_NAME.get(b, b)
            lines.append(" & ".join([
                NAME[k] if first else "", tex(name) if "$" not in name else name, f"{int(t.n):,}", f100(ce, 1),
                f100(gat, 1), ("--" if v is None else f"{v.best_alpha:.2f}"),
                f"{t.best_alpha:.2f} [{t.ci_lo:.2f}, {t.ci_hi:.2f}]", f"{100 * t.frac_pref_ctx:.0f}", gq]) + " \\\\")
            first = False
        lines.append("\\midrule")
    lines = lines[:-1]
    body = ("\\begin{table*}[t]\n\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{llrrrrlrr}\n\\toprule\n"
            " & & & \\multicolumn{2}{c}{Test nDCG@10} & \\multicolumn{2}{c}{Best fixed $\\alpha$ in bucket} & "
            "\\% queries & Learned \\\\\n\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
            "Dataset & Bucket & $n$ (test) & CE & GAT & val & test [95\\% CI] & prefer ctx & $\\bar\\alpha$ (gate) \\\\\n"
            "\\midrule\n" + "\n".join(lines) +
            "\n\\bottomrule\n\\end{tabular}\n\\caption{Query-level test of the hypothesis (min--max). Best fixed "
            "$\\alpha$ = arg-max of the bucket's mean nDCG@10 curve; CI from 2{,}000 bootstrap resamples of the "
            "bucket's queries. ``\\% prefer ctx'': among queries whose nDCG@10 changes with $\\alpha$, the share whose "
            "optimal $\\alpha$-interval is centred below 0.5 (tie-robust; the remaining queries are indifferent). "
            "Learned $\\bar\\alpha$: mean $\\alpha$ of the best (on val) in-domain gate in the bucket. Analysis only: "
            "no label enters any gate.}\n\\label{tab:bucket}\n\\end{table*}\n")
    save_tab("buckets", body)


def tab_bucketed():
    b = pd.read_csv(RES / "report" / "bucketed_alpha.csv")
    lines = []
    for _, r in b.iterrows():
        m = ""
        if r.p < 0.05 and r.ci_lo > 0:
            m = "$^{\\blacktriangle}$"
        elif r.p < 0.05 and r.ci_hi < 0:
            m = "$^{\\triangledown}$"
        lab = {"n_words": "length", "hops": "hops (gold)", "type": "type (gold)", "wh": "wh-word"}[r.label]
        parts = [x.rsplit("=", 1) for x in str(r.alphas).split("; ")]
        al = "; ".join(BUCKET_NAME.get(b, tex(b)) + f" {a}" for b, a in parts)
        lines.append(f"{SHORT[r.key]} & {lab} & {r.alpha0:.2f} & {{\\scriptsize {al}}} & "
                     f"{100 * r['diff']:+.2f}{m} & [{100 * r.ci_lo:+.2f}, {100 * r.ci_hi:+.2f}] \\\\")
    body = ("\\begin{table*}[t]\n\\centering\\small\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular}{llrp{7.2cm}rr}\n\\toprule\n"
            "Data & Buckets & $\\alpha_0$ & Per-bucket $\\alpha$ (chosen on val) & $\\Delta$ & 95\\% CI \\\\\n\\midrule\n"
            + "\n".join(lines) +
            "\n\\bottomrule\n\\end{tabular}\n\\caption{Bucketed fixed $\\alpha$: one $\\alpha$ per bucket selected on val "
            "and applied on test, against the single global $\\alpha_0$ (test $\\Delta$nDCG@10 $\\times$100, paired "
            "bootstrap). Length buckets are available at test time; gold hop/type labels are not, so those rows "
            "measure how much a type-aware $\\alpha$ \\emph{could} give (analysis only).}\n\\label{tab:bucketed}\n\\end{table*}\n")
    save_tab("bucketed", body)


def tab_hyp():
    h = pd.read_csv(RES / "hypotheses.csv")
    h = h[h.primary == True]  # noqa: E712
    stat = {"spearman(alpha, hops)": "Spearman($\\alpha$, hops)",
            "mean alpha bridge - comparison": "$\\bar\\alpha_{\\text{bridge}}-\\bar\\alpha_{\\text{comp.}}$",
            "gain(ds) - gain(MS MARCO)": "gain $-$ gain$_{\\text{MSM}}$",
            "headroom(ds) - headroom(MS MARCO)": "headroom $-$ headroom$_{\\text{MSM}}$"}
    lines = []
    for _, r in h.iterrows():
        sc = 1 if r.hypothesis == "H1" else 100
        lines.append(f"{r.hypothesis} & {NAME[r.dataset]} & {tex(r.variant)} & {stat[r.statistic]} & "
                     f"{sc * r.value:+.3f} & [{sc * r.ci95_lo:+.3f}, {sc * r.ci95_hi:+.3f}] & {r.p_holm:.3f} & "
                     f"{r.predicted} & {r.verdict} \\\\")
    body = ("\\begin{table*}[t]\n\\centering\\small\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{llllrlrcl}\n\\toprule\n"
            "Hyp. & Dataset & Gate & Statistic & Value & 95\\% CI & $p_{\\text{Holm}}$ & Pred. & Verdict \\\\\n\\midrule\n"
            + "\n".join(lines) +
            "\n\\bottomrule\n\\end{tabular}\n\\caption{Pre-registered primary tests (written before any multi-hop run; "
            "Holm over the nine tests). H1 uses the best (on val) gate; H2/H3 values are nDCG@10 $\\times$100. "
            "H1 verdicts on MuSiQue and BEIR are computed on gates whose $\\alpha$ varies by less than $10^{-3}$ "
            "across queries (Table~\\ref{tab:gate}), so their sign is not interpretable.}\n\\label{tab:hyp}\n\\end{table*}\n")
    save_tab("hypotheses", body)


def tab_graph():
    es = pd.read_csv(RES / "edge_stats.csv")
    h = pd.read_csv(RES / "hypotheses.csv")
    lines = []
    for ds in ("musique", "hotpotqa"):
        first = True
        for gt in ("none", "random", "knn_emb", "title_link", "hybrid"):
            key = ds if gt == "knn_emb" else f"{ds}@{gt}"
            t = main_table(key)
            e = es[(es.dataset == ds) & (es.split == "test") & (es.graph_type == gt)].iloc[0]
            if gt == "knn_emb":
                hh = h[(h.dataset == ds) & (h.graph == "knn_emb") & h.hypothesis.str.startswith("H2 (") &
                       (h.statistic == "gain(ds) = adaptive - best fixed")]
                ho = h[(h.dataset == ds) & (h.graph == "knn_emb") & h.hypothesis.str.startswith("H3 (") &
                       (h.statistic == "headroom(ds) = oracle - best fixed")]
            else:
                hh = h[(h.dataset == ds) & (h.graph == gt) & h.hypothesis.str.contains("H2") &
                       (h.statistic == "gain(ds) = adaptive - best fixed")]
                ho = h[(h.dataset == ds) & (h.graph == gt) & h.hypothesis.str.contains("H3") &
                       (h.statistic == "headroom(ds) = oracle - best fixed")]
            gain = hh.iloc[0]
            m = "$^{\\blacktriangle}$" if gain.p_raw < 0.05 and gain.ci95_lo > 0 else ""
            a0 = json.loads((RES / key / "alpha0.json").read_text())[PN]["alpha0"]
            gname = {"none": "none (self-loops)", "random": "random (same degrees)", "knn_emb": "kNN emb. (default)",
                     "title_link": "title link", "hybrid": "kNN $\\cup$ title"}[gt]
            lines.append(" & ".join([NAME[ds] if first else "", gname, f"{e.mean_degree:.2f}",
                                     f"{100 * e.frac_gold_pairs_connected:.0f}",
                                     f100(t.loc["GAT-only (alpha=0)", "test_ndcg@10_mean"], 1),
                                     f"{a0:.2f}", f100(t.loc[f"{PN}_fixedgrid", "test_ndcg@10_mean"], 2),
                                     f"{100 * gain.value:+.2f}{m}", f"{100 * ho.iloc[0].value:.1f}"]) + " \\\\")
            first = False
        lines.append("\\midrule")
    lines = lines[:-1]
    body = ("\\begin{table*}[t]\n\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{llrrrrrrr}\n\\toprule\n"
            "Dataset & Graph & Degree & Gold pairs & GAT & $\\alpha_0$ & Fixed & $\\Delta$ adaptive & Headroom \\\\\n"
            " & & & linked (\\%) & nDCG@10 & & nDCG@10 & (best gate) & (oracle) \\\\\n\\midrule\n"
            + "\n".join(lines) +
            "\n\\bottomrule\n\\end{tabular}\n\\caption{Graph controls (H4, exploratory): a GAT retrained per graph type, "
            "everything else fixed. Degree excludes self-loops; ``gold pairs linked'' = share of supporting-paragraph "
            "pairs joined by an edge (test). nDCG@10 and $\\Delta$ are $\\times$100 on test; $\\Delta$ = best gate (on "
            "val) $-$ best fixed $\\alpha$, $^{\\blacktriangle}$ $p<0.05$.}\n\\label{tab:graph}\n\\end{table*}\n")
    save_tab("graph", body)


# rows left out of the paper table at the authors' request (still in results/transfer.csv)
TRANSFER_HIDE = {("zero_shot", "musique")}


def tab_transfer():
    tr = pd.read_csv(RES / "transfer.csv")
    lines = []
    sname = {"zero_shot": "zero-shot", "mixed": "mixed", "lodo": "leave-one-out"}
    for (s, train, test), d in tr.groupby(["setting", "train", "test"], sort=False):
        if (s, test) in TRANSFER_HIDE:
            continue
        d = d.set_index("variant")
        cells = [sname[s], tex(train.replace("A", "MSM").replace("hotpotqa", "HQA").replace("musique", "MuSiQue")),
                 NAME[test]]
        cells += [f100(d.loc[v, "ndcg10_mean"]) for v in ("fixed", "Q", "QS", "S")]
        cells += [f100(d.loc["fixed", "in_domain_ndcg10"]), f"{d.loc['fixed', 'alpha0']:.2f}",
                  f"{d.loc['Q', 'alpha_mean']:.2f} ({d.loc['Q', 'alpha_std_over_queries']:.2f})"]
        lines.append(" & ".join(cells) + " \\\\")
    body = ("\\begin{table*}[t]\n\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lllrrrrrrr}\n\\toprule\n"
            " & & & \\multicolumn{4}{c}{Transferred gate, test nDCG@10} & In-domain & & \\textsc{Q} gate \\\\\n"
            "\\cmidrule(lr){4-7}\nSetting & Train & Test & fixed & \\textsc{Q} & \\textsc{QS} & \\textsc{S} & fixed & "
            "$\\alpha_0$ & $\\bar\\alpha$ (sd) \\\\\n\\midrule\n" + "\n".join(lines) +
            "\n\\bottomrule\n\\end{tabular}\n\\caption{Transfer and mixed training of the gate (E4, min--max, "
            "$\\times$100, 5 seeds). Zero-shot uses the frozen MS MARCO GAT for $s_{ctx}$; mixed and leave-one-out use "
            "each dataset's own GAT. $\\alpha_0$ is the fixed $\\alpha$ the gate starts from (selected on the training "
            "sources' val).}\n\\label{tab:transfer}\n\\end{table*}\n")
    save_tab("transfer", body, fit=True)


FULL_ROWS = ROWS_MAIN[:-1] + [
    (f"{PN}_Q_listwise_lam0", "\\textsc{Q}, $\\lambda=0$"), (f"{PN}_QS_listwise_lam0", "\\textsc{QS}, $\\lambda=0$"),
    (f"{PN}_S_listwise_lam0", "\\textsc{S}, $\\lambda=0$"), ROWS_MAIN[-1]]


def tab_full(key):
    t = main_table(key)
    ks = [1, 3, 5, 10, 50]
    ms = [("ndcg", "nDCG"), ("mrr", "MRR"), ("p", "P"), ("r", "R")]
    lines = []
    rows = list(FULL_ROWS)
    for c in t.index:   # loss ablation rows (normaliser chosen by the pipeline)
        if "_bce_lam0.01" in c:
            n, dep = c.split("_")[0], c.split("_")[1]
            lab = "Fixed $\\alpha$" if dep == "fixed" else f"\\textsc{{{dep}}}"
            rows.insert(-1, (c, f"{lab}, BCE loss ({n})"))
    for ck, label in rows:
        rk = row_key(t, ck)
        if rk is None:
            continue
        cells = [f100(t.loc[rk, f"test_{m}@{k}_mean"], 1) for m, _ in ms for k in ks]
        cells.append(f100(t.loc[rk, "test_map_mean"], 1))
        if ck.startswith(f"{PN}_paper") or ck.startswith(f"{PN}_Q_listwise_{LAM}") or "oracle" in ck:
            lines.append("\\midrule")
        lines.append(label + " & " + " & ".join(cells) + " \\\\")
    head = " & ".join(f"\\multicolumn{{5}}{{c}}{{{n}@$k$}}" for _, n in ms)
    cm = " ".join(f"\\cmidrule(lr){{{2 + 5 * i}-{6 + 5 * i}}}" for i in range(4))
    sub = " & ".join([" & ".join(str(k) for k in ks)] * 4)
    note = ""
    if key in ("musique", "hotpotqa", "twowiki"):
        note = (" Closed pools: every gold passage is in the pool, so R@50 $=1$ and P@50 = \\#gold/50 for all rows "
                "(also R@10 $=1$ when $K=10$).")
    body = ("\\begin{table*}[t]\n\\centering\n\\resizebox{\\textwidth}{!}{%\n"
            f"\\begin{{tabular}}{{l{'r' * 21}}}\n\\toprule\n & {head} & \\\\\n{cm}\n & {sub} & MAP \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}}\n"
            f"\\caption{{{NAME[key]}: all test metrics ($\\times$100, mean over seeds), min--max normaliser unless "
            f"stated.{note}}}\n\\label{{tab:full_{key}}}\n\\end{{table*}}\n")
    save_tab(f"full_{key}", body)


def tab_trec():
    lines = []
    for key in ("A", "B"):
        t = main_table(key)
        first = True
        for ck, label in ROWS_MAIN:
            rk = row_key(t, ck)
            if rk is None or "QSC" in ck:
                continue
            cells = [f100(t.loc[rk, f"{s}_{m}_mean"], 1) for s in ("trec19", "trec20") for m in ("ndcg@10", "mrr@10", "map")]
            lines.append(f"{NAME[key] if first else ''} & {label} & " + " & ".join(cells) + " \\\\")
            first = False
        lines.append("\\midrule")
    body = ("\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{llrrrrrr}\n\\toprule\n"
            " & & \\multicolumn{3}{c}{TREC DL 2019} & \\multicolumn{3}{c}{TREC DL 2020} \\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n & & nDCG@10 & MRR@10 & MAP & nDCG@10 & MRR@10 & MAP \\\\\n\\midrule\n"
            + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n\\caption{Out-of-distribution TREC DL "
            "(graded nDCG; MRR and MAP with rel $\\ge 2$), $\\times$100. Not used for any decision.}\n"
            "\\label{tab:trec}\n\\end{table*}\n")
    save_tab("trec", body, fit=True)


def tab_controls():
    a = pd.read_csv(RES / "A" / "gat_controls.csv")
    lines = []
    for _, r in a.iterrows():
        lines.append(f"MS MARCO (A) & {tex(r.control)} & {r.mean_degree:.1f} & {f100(r['val_ndcg@10'])} & "
                     f"{f100(r['val_ndcg@10_s_bi'])} \\\\")
    lines.append("\\midrule")
    for ds in ("musique", "hotpotqa"):
        c = pd.read_csv(RES / ds / "gat_controls.csv")
        for _, r in c.iterrows():
            lines.append(f"{NAME[ds]} & {tex(r.control)} & {r.mean_degree:.1f} & {f100(r['val_ndcg@10'])} & "
                         f"{f100(r['val_ndcg@10_s_bi'])} \\\\")
    body = ("\\begin{table}[t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{llrrr}\n\\toprule\n"
            "Data & Graph & Deg. & GAT & $s_{bi}$ \\\\\n\\midrule\n" + "\n".join(lines) +
            "\n\\bottomrule\n\\end{tabular}\n\\caption{GAT graph sensitivity on val (nDCG@10 $\\times$100 of $s_{ctx}$ "
            "alone). knn\\_k$k$\\_s$m$: $k$ neighbours, min cosine $m$. MS MARCO degrees include the self-loop.}\n"
            "\\label{tab:controls}\n\\end{table}\n")
    save_tab("controls", body)


# ------------------------------------------------------------------ figures
def fig_curves(st):
    cur = pd.read_csv(RES / "report" / "alpha_curves.csv")
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 3.4), sharex=True)
    for ax, k in zip(axes.flat, KEYS):
        for split, ls, mfc in (("val", "--", "white"), ("test", "-", COLOR[k])):
            v = cur[(cur.key == k) & (cur.norm == PN) & (cur.split == split)]
            y = 100 * (v["ndcg@10"].values - v["ndcg@10"].max())
            ax.plot(v.alpha, y, ls, color=COLOR[k], label=split)
            i = y.argmax()
            ax.plot(v.alpha.values[i], 0, "o", ms=6, mfc=mfc, mec=COLOR[k], mew=1.5, zorder=3)
        ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
        ax.set_title(NAME[k], color=INK, loc="left")
        ax.set_ylim(-6, 0.6)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    for ax in axes[1]:
        ax.set_xlabel("fixed $\\alpha$ (weight on CE)")
    for ax in axes[:, 0]:
        ax.set_ylabel("nDCG@10 $-$ max (pts)")
    axes[0, 0].legend(frameon=False, loc="lower center")
    fig.tight_layout()
    save_fig(fig, "alpha_curves")


def fig_bucket_alpha(ga):
    ba = pd.read_csv(RES / "report" / "bucket_alpha.csv")
    spec = [("A", "n_words", ["short (<=4)", "medium (5-7)", "long (>=8)"]), ("musique", "hops", ["2", "3", "4"]),
            ("hotpotqa", "type", ["comparison", "bridge"]), ("beir_hotpotqa", "type", ["comparison", "bridge"]),
            ("twowiki", "type", ["comparison", "compositional", "inference", "bridge_comparison"])]
    rows = [(k, lab, b) for k, lab, bs in spec for b in bs]
    fig, ax = plt.subplots(figsize=(3.15, 3.9))
    ax.axvspan(0, 0.5, color="#f1efe8", zorder=0)
    y = np.arange(len(rows))[::-1]
    for yi, (k, lab, b) in zip(y, rows):
        t = ba[(ba.key == k) & (ba.label == lab) & (ba.bucket.astype(str) == b) & (ba.split == "test")].iloc[0]
        v = ba[(ba.key == k) & (ba.label == lab) & (ba.bucket.astype(str) == b) & (ba.split == "val")]
        ax.plot([t.ci_lo, t.ci_hi], [yi, yi], color=COLOR[k], lw=1.5, solid_capstyle="round")
        ax.plot(t.best_alpha, yi, "o", ms=6, color=COLOR[k], mec="white", mew=1, zorder=3)
        if len(v):
            ax.plot(v.iloc[0].best_alpha, yi, "o", ms=6, mfc="white", mec=COLOR[k], mew=1.2, zorder=2)
    plain = lambda b: BUCKET_NAME.get(b, b).replace("$\\le$", "≤").replace("$\\ge$", "≥")
    labels = [SHORT[k] + ": " + plain(b) for k, _, b in rows]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("best fixed $\\alpha$ within bucket (min--max)")
    ax.text(0.25, y[0] + 0.9, "favours GAT", ha="center", color=INK2, fontsize=6.5)
    ax.text(0.75, y[0] + 0.9, "favours CE", ha="center", color=INK2, fontsize=6.5)
    ax.set_ylim(-0.7, y[0] + 1.6)
    ax.grid(axis="y", visible=False)
    h1 = ax.plot([], [], "o", color=INK2, label="test (95% CI)")[0]
    h2 = ax.plot([], [], "o", mfc="white", mec=INK2, label="val")[0]
    ax.legend(handles=[h1, h2], frameon=False, loc="lower right", bbox_to_anchor=(1.0, -0.22), ncol=2)
    fig.tight_layout()
    save_fig(fig, "bucket_alpha")


def fig_gap_vs_alpha():
    ba = pd.read_csv(RES / "report" / "bucket_alpha.csv")
    bc = pd.read_csv(RES / "report" / "bucket_curves.csv")
    spec = {"A": "n_words", "B": "n_words", "musique": "hops", "hotpotqa": "type", "beir_hotpotqa": "type",
            "twowiki": "type"}
    xs, ys = [], []
    fig, ax = plt.subplots(figsize=(3.15, 2.5))
    for k, lab in spec.items():
        t = ba[(ba.key == k) & (ba.label == lab) & (ba.split == "test")]
        for _, r in t.iterrows():
            c = bc[(bc.key == k) & (bc.label == lab) & (bc.bucket.astype(str) == str(r.bucket)) & (bc.split == "test")]
            gap = 100 * (c[c.alpha == 1.0]["ndcg@10"].iloc[0] - c[c.alpha == 0.0]["ndcg@10"].iloc[0])
            xs.append(gap)
            ys.append(r.best_alpha)
            ax.plot(gap, r.best_alpha, "o", ms=6, color=COLOR[k], mec="white", mew=1, zorder=3)
        ax.plot([], [], "o", color=COLOR[k], label=SHORT[k])
    rho = spearman(np.array(xs), np.array(ys))
    NUM["gap_alpha_spearman"] = rho
    NUM["gap_alpha_n"] = len(xs)
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    ax.axvline(0, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("CE $-$ GAT nDCG@10 in bucket (pts)")
    ax.set_ylabel("best fixed $\\alpha$ (test)")
    ax.set_title(f"Spearman $\\rho$ = {rho:.2f} over {len(xs)} buckets", loc="left", color=INK)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    save_fig(fig, "gap_vs_alpha")


def fig_gate(ga):
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.0), sharey=True)
    tr_runs = sorted(glob.glob(str(RES / "transfer" / "runs" / "mixed_*_Q_s*.json")))
    for ax, dep in zip(axes, ("Q", "QS", "S", "mixed Q")):
        data, cols, labs = [], [], []
        for k in KEYS:
            if dep == "mixed Q":
                arr = [np.load(p.replace(".json", ".npz"))["test__alpha"] for p in tr_runs if f"_to_{k}_" in p]
                if not arr:
                    continue
                a = np.mean(arr, axis=0)
            else:
                if (k, dep) not in ga:
                    continue
                a = ga[(k, dep)]["alpha"]
            data.append(a)
            cols.append(COLOR[k])
            labs.append(SHORT[k])
        bp = ax.boxplot(data, vert=True, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "lw": 1}, whiskerprops={"color": INK2, "lw": 0.8},
                        capprops={"color": INK2, "lw": 0.8})
        for p, c in zip(bp["boxes"], cols):
            p.set_facecolor(c)
            p.set_edgecolor(c)
        ax.set_xticks(range(1, len(labs) + 1))
        ax.set_xticklabels(labs, rotation=60, ha="right")
        ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
        ax.set_title({"mixed Q": "mixed-data \\textsc{Q}".replace("\\textsc{Q}", "Q")}.get(dep, f"in-domain {dep}"),
                     loc="left", color=INK)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("learned $\\alpha$ (test queries)")
    fig.tight_layout()
    save_fig(fig, "gate_alpha")


def fig_graph():
    es = pd.read_csv(RES / "edge_stats.csv")
    gts = ["none", "random", "knn_emb", "title_link", "hybrid"]
    gl = ["none", "random", "kNN", "title", "kNN+title"]
    fig, axes = plt.subplots(1, 2, figsize=(3.15, 2.1), sharey=False)
    for i, ds in enumerate(("musique", "hotpotqa")):
        link = [100 * es[(es.dataset == ds) & (es.split == "test") & (es.graph_type == g)].frac_gold_pairs_connected.iloc[0]
                for g in gts]
        gat = [100 * main_table(ds if g == "knn_emb" else f"{ds}@{g}").loc["GAT-only (alpha=0)", "test_ndcg@10_mean"]
               for g in gts]
        ax = axes[i]
        ax.plot(link, gat, "-", color=COLOR[ds], lw=0.8, alpha=0.4)
        for x, yv, l in zip(link, gat, gl):
            ax.plot(x, yv, "o", ms=6, color=COLOR[ds], mec="white", mew=1, zorder=3)
            ax.annotate(l, (x, yv), xytext=(3, -8 if l in ("random",) else 3), textcoords="offset points",
                        fontsize=6.5, color=INK2)
        ax.set_title(NAME[ds], loc="left", color=INK)
        ax.set_xlabel("gold pairs linked (%)")
        ax.set_xlim(-5, 100)
    axes[0].set_ylabel("GAT-only nDCG@10")
    fig.tight_layout()
    save_fig(fig, "graph_types")


# ------------------------------------------------------------------ complexity-focused (report_hops.py outputs)
HOPC = {"2": "#2a78d6", "3": "#eb6834", "4": "#1baf7a"}
CE_C, GAT_C = "#2a78d6", "#eb6834"


def _hop_optima():
    return pd.read_csv(RES / "report" / "hops_optima.csv")


def fig_hops():
    o = _hop_optima()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.1))
    for ax, split in zip(axes, ("val", "test")):
        for h in ("2", "3", "4"):
            r = o[(o.key == "musique") & (o.split == split) & (o.norm == "minmax") & (o.field == "hops") &
                  (o.bucket.astype(str) == h)].iloc[0]
            c = np.array(json.loads(r.curve))
            g = np.linspace(0, 1, len(c))
            ax.plot(g, 100 * (c - c.max()), color=HOPC[h], label=f"{h}-hop")
            ax.plot(g[c.argmax()], 0, "o", ms=6, color=HOPC[h], mec="white", mew=1, zorder=3)
        ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
        ax.set_ylim(-8, 0.8)
        ax.set_xlabel("fixed $\\alpha$ (weight on CE)")
        ax.set_title(f"MuSiQue {split}", loc="left", color=INK)
    axes[0].set_ylabel("nDCG@10 $-$ max (pts)")
    axes[0].legend(frameon=False, loc="lower center")
    fig.tight_layout()
    save_fig(fig, "hops_curves")
    fig, ax = plt.subplots(figsize=(3.15, 2.7))
    ax.axhspan(0, 0.5, color="#f1efe8", zorder=0)
    styles = [("knn_emb", "minmax", "min--max", "o", "-"), ("knn_emb", "zscore", "z-score", "s", "-"),
              ("knn_emb", "rank", "rank", "^", "-"), ("none", "minmax", "no graph", "o", ":"),
              ("random", "minmax", "random graph", "s", ":"), ("title_link", "minmax", "title graph", "^", ":"),
              ("hybrid", "minmax", "kNN+title", "D", ":")]
    for graph, norm, lab, mk, ls in styles:
        r = o[(o.dataset == "musique") & (o.graph == graph) & (o.norm == norm) & (o.field == "hops") &
              (o.split == "test")].sort_values("bucket")
        col = {"min--max": "#6250d6", "z-score": "#eda100", "rank": "#e87ba4"}.get(lab, "#9a9890")
        ax.plot(r.bucket.astype(int), r.best_alpha, ls, marker=mk, ms=4.5, color=col, lw=1.2, label=lab)
    ax.set_xticks([2, 3, 4])
    ax.set_xlabel("hops")
    ax.set_ylabel("best fixed $\\alpha$ (test)")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3,
              handlelength=1.8, columnspacing=0.8)
    fig.tight_layout()
    save_fig(fig, "hops_robust")


def fig_signals():
    sg = pd.read_csv(RES / "report" / "hops_signals.csv")
    sg = sg[(sg.dataset == "musique") & (sg.field == "hops")]
    fig, ax = plt.subplots(figsize=(3.15, 2.1))
    for split, ls, mfc in (("val", "--", "white"), ("test", "-", None)):
        d = sg[sg.split == split].sort_values("bucket")
        for col, c, lab in (("ce", CE_C, "CE ($s_{sem}$)"), ("gat", GAT_C, "GAT ($s_{ctx}$)")):
            ax.plot(d.bucket.astype(int), 100 * d[col], ls, marker="o", ms=5, color=c,
                    mfc=mfc or c, mec=c, label=f"{lab}, {split}")
    ax.set_xticks([2, 3, 4])
    ax.set_xlabel("hops")
    ax.set_ylabel("nDCG@10 alone")
    ax.legend(frameon=False, fontsize=6.3, ncol=2, loc="lower left")
    fig.tight_layout()
    save_fig(fig, "hops_signals")


def tab_hops():
    o = _hop_optima()
    sg = pd.read_csv(RES / "report" / "hops_signals.csv")
    pf = pd.read_csv(RES / "report" / "hops_pref.csv")
    lines = []
    for h in ("2", "3", "4"):
        cells = [f"{h}-hop"]
        for split in ("val", "test"):
            s = sg[(sg.dataset == "musique") & (sg.field == "hops") & (sg.split == split) &
                   (sg.bucket.astype(str) == h)].iloc[0]
            cells += [f"{100 * s.ce:.1f}", f"{100 * s.gat:.1f}"]
        for split in ("val", "test"):
            for norm in ("minmax", "zscore", "rank"):
                r = o[(o.key == "musique") & (o.split == split) & (o.norm == norm) & (o.field == "hops") &
                      (o.bucket.astype(str) == h)].iloc[0]
                v = f"{r.best_alpha:.2f}"
                cells.append(f"\\textbf{{{v}}}" if r.best_alpha < 0.5 else v)
        for split in ("val", "test"):
            p = pf[(pf.dataset == "musique") & (pf.field == "hops") & (pf.split == split)].iloc[0]
            cells.append(f"{100 * p[f'frac_ctx_{h}']:.0f}")
        n = int(o[(o.key == "musique") & (o.split == "test") & (o.norm == "minmax") & (o.field == "hops") &
                  (o.bucket.astype(str) == h)].iloc[0].n)
        lines.append(" & ".join([cells[0], f"{n:,}"] + cells[1:]) + " \\\\")
    sp = {s: pf[(pf.dataset == "musique") & (pf.field == "hops") & (pf.split == s)].iloc[0] for s in ("val", "test")}
    NUM["hop_spearman"] = {s: [float(sp[s].spearman), float(sp[s].ci_lo), float(sp[s].ci_hi), float(sp[s].p)]
                           for s in sp}
    body = ("\\begin{table*}[t]\n\\centering\\small\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular}{lrrrrrrrrrrrrr}\n\\toprule\n"
            " & & \\multicolumn{4}{c}{nDCG@10 of each signal alone} & \\multicolumn{6}{c}{Best fixed $\\alpha$} & "
            "\\multicolumn{2}{c}{\\% prefer ctx} \\\\\n"
            "\\cmidrule(lr){3-6}\\cmidrule(lr){7-12}\\cmidrule(lr){13-14}\n"
            " & & \\multicolumn{2}{c}{val} & \\multicolumn{2}{c}{test} & \\multicolumn{3}{c}{val} & "
            "\\multicolumn{3}{c}{test} & & \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-12}\n"
            "Hops & $n$ (test) & CE & GAT & CE & GAT & mm & z & rank & mm & z & rank & val & test \\\\\n\\midrule\n"
            + "\n".join(lines) +
            "\n\\bottomrule\n\\end{tabular}\n\\caption{MuSiQue by hop count. Left: the cross-encoder loses 10.8 (val) and "
            "12.5 (test) nDCG@10 points from 2 to 4 hops, the GAT only 5.1 and 4.7, and overtakes it on 4-hop "
            "questions. Middle: the best fixed $\\alpha$ falls with the hop count under every scale-free normaliser "
            "(mm = min--max, z = z-score); bold = below 0.5, i.e.\\ the contextual score dominates. Right: share of "
            "$\\alpha$-sensitive questions whose optimal $\\alpha$-interval is centred below 0.5. Tie-robust "
            f"per-question test of the trend: Spearman(preferred $\\alpha$, hops) = {sp['val'].spearman:+.3f} "
            f"[{sp['val'].ci_lo:+.3f}, {sp['val'].ci_hi:+.3f}] on val ($p={sp['val'].p:.4f}$) and "
            f"{sp['test'].spearman:+.3f} [{sp['test'].ci_lo:+.3f}, {sp['test'].ci_hi:+.3f}] on test ($p={sp['test'].p:.3f}$).}}\n"
            "\\label{tab:hops}\n\\end{table*}\n")
    save_tab("hops", body)


def tab_graph_hops():
    o = _hop_optima()
    lines = []
    for graph, name in (("none", "none (self-loops)"), ("random", "random"), ("knn_emb", "kNN emb. (default)"),
                        ("title_link", "title link"), ("hybrid", "kNN $\\cup$ title")):
        cells = [name]
        for split in ("val", "test"):
            r = o[(o.dataset == "musique") & (o.graph == graph) & (o.norm == "minmax") & (o.field == "hops") &
                  (o.split == split)].sort_values("bucket")
            cells += [f"\\textbf{{{a:.2f}}}" if a < 0.5 else f"{a:.2f}" for a in r.best_alpha]
        lines.append(" & ".join(cells) + " \\\\")
    body = ("\\begin{table}[t]\n\\centering\\small\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrrrrrr}\n\\toprule\n"
            " & \\multicolumn{3}{c}{val} & \\multicolumn{3}{c}{test} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
            "Graph & 2 & 3 & 4 & 2 & 3 & 4 \\\\\n\\midrule\n" + "\n".join(lines) +
            "\n\\bottomrule\n\\end{tabular}\n\\caption{MuSiQue: best fixed $\\alpha$ (min--max) by hop count for a GAT "
            "retrained on each graph type. The 4-hop optimum is the lowest and below 0.5 (bold) for every graph.}\n"
            "\\label{tab:graphhops}\n\\end{table}\n")
    save_tab("graph_hops", body)


def tab_2wiki():
    o = _hop_optima()
    g = pd.read_csv(RES / "report" / "hops_gains.csv")
    sg = pd.read_csv(RES / "report" / "hops_signals.csv")
    pf = pd.read_csv(RES / "report" / "hops_pref.csv")
    pf = pf[(pf.dataset == "twowiki") & (pf.split == "test")].iloc[0]
    lines = []
    for t, name in (("comparison", "comparison"), ("compositional", "compositional"), ("inference", "inference"),
                    ("bridge_comparison", "bridge-comparison")):
        s = sg[(sg.dataset == "twowiki") & (sg.split == "test") & (sg.bucket == t)].iloc[0]
        cells = [name, f"{100 * s.ce:.1f}", f"{100 * s.gat:.1f}"]
        for split in ("val", "test"):
            r = o[(o.dataset == "twowiki") & (o.norm == "minmax") & (o.split == split) & (o.bucket == t)].iloc[0]
            cells.append(f"\\textbf{{{r.best_alpha:.2f}}}" if r.best_alpha < 0.5 else f"{r.best_alpha:.2f}")
        cells.append(f"{100 * pf[f'frac_ctx_{t}']:.0f}")
        for norm in ("minmax", "zscore", "rank"):
            r = g[(g.dataset == "twowiki") & (g.norm == norm) & (g.dep == "Q") & (g.bucket == t)].iloc[0]
            m = "$^{\\blacktriangle}$" if (r.p < 0.05 and r.ci_lo > 0) else ("$^{\\triangledown}$" if (r.p < 0.05 and r.ci_hi < 0) else "")
            cells.append(f"{r.alpha_mean:.2f}")
            cells.append(f"{100 * r.gain:+.2f}{m}")
        lines.append(" & ".join(cells) + " \\\\")
    body = ("\\begin{table*}[t]\n\\centering\\small\\setlength{\\tabcolsep}{3.6pt}\n\\begin{tabular}{lrrrrrrrrrrr}\n\\toprule\n"
            " & \\multicolumn{2}{c}{nDCG@10 (test)} & \\multicolumn{2}{c}{Best fixed $\\alpha$} & \\% prefer & "
            "\\multicolumn{6}{c}{Learned \\gate{Q} gate: mean $\\alpha$ and $\\Delta$nDCG@10 vs.\\ fixed (test)} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){7-12}\n"
            "Type & CE & GAT & val & test & ctx & \\multicolumn{2}{c}{min--max} & \\multicolumn{2}{c}{z-score} & "
            "\\multicolumn{2}{c}{rank} \\\\\n\\midrule\n" + "\n".join(lines) +
            "\n\\bottomrule\n\\end{tabular}\n\\caption{2WikiMultiHopQA by question type, ordered from the simplest "
            "(comparison: both entities named in the question) to the most compositional (bridge-comparison). "
            "Bold $\\alpha<0.5$. The query-only gate, which never sees the type label, gives comparison questions the "
            "highest $\\alpha$ under all three normalisers, and gains significantly on bridge-comparison questions "
            "($^{\\blacktriangle}$/$^{\\triangledown}$: paired bootstrap $p<0.05$).}\n\\label{tab:twowiki}\n\\end{table*}\n")
    save_tab("twowiki_types", body)


def numbers(st, ga, n_pos):
    NUM["alpha_stats"] = st
    NUM["gate"] = {f"{k}/{d}": {kk: v for kk, v in x.items() if kk != "alpha"} for (k, d), x in ga.items()}
    NUM["n_positive_grid_cells"] = n_pos
    (OUT / "numbers.json").write_text(json.dumps(NUM, indent=1, default=float))


def main():
    load_config()
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    st = alpha_curve_stats()
    ga = gate_alpha()
    tab_datasets()
    tab_main()
    tab_dataset_alpha(st)
    n_pos = tab_adaptive_grid()
    tab_gate(ga)
    tab_buckets(ga)
    tab_bucketed()
    tab_hyp()
    tab_graph()
    tab_transfer()
    for k in KEYS:
        tab_full(k)
    tab_trec()
    tab_controls()
    fig_curves(st)
    fig_bucket_alpha(ga)
    fig_gap_vs_alpha()
    fig_gate(ga)
    if (RES / "report" / "hops_optima.csv").exists():
        fig_hops()
        fig_signals()
        tab_hops()
        tab_graph_hops()
        tab_2wiki()
    numbers(st, ga, n_pos)
    print("tables:", sorted(p.name for p in TAB.glob("*.tex")))
    print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
