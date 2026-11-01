"""Stage 6 analysis: plots + auto-generated results summary (also written into NOTES.md).

Usage:  python -m src.analysis --protocol A
Writes: results/{P}/figures/*.png|pdf, results/{P}/summary.md, NOTES.md results section
"""
import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.common import ROOT, get_logger, load_config, load_json, path, results_path, set_env  # noqa: E402
from src.evaluate import best_fixed_key, group_runs, load_runs  # noqa: E402

log = get_logger("analysis")

# reference palette (dataviz skill): categorical slots 1-4 in fixed order, blue sequential ramp
SERIES = {"fixed": "#2a78d6", "Q": "#eb6834", "QS": "#1baf7a", "S": "#eda100"}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DEPS = ("Q", "QS", "S")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "legend.frameon": False,
})


def _save(fig, cfg, P, name):
    out = results_path(cfg, P, "figures", f"{name}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _val(groups, k):
    return np.mean([r["metrics"]["val"]["ndcg@10"] for r in groups[k]])


def _test(groups, k, s="test", m="ndcg@10"):
    return np.mean([r["metrics"][s][m] for r in groups[k]])


def heatmap(cfg, P, groups, lam_tag):
    norms = [n for n in cfg["fusion"]["normalizers"] if f"{n}_fixedgrid" in groups]
    cols = ["fixed", *DEPS]
    M = np.full((len(norms), len(cols)), np.nan)
    for i, n in enumerate(norms):
        for j, d in enumerate(cols):
            k = best_fixed_key(groups, n, lam_tag) if d == "fixed" else f"{n}_{d}_listwise_{lam_tag}"
            if k in groups:
                M[i, j] = _test(groups, k)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    im = ax.imshow(M, cmap=LinearSegmentedColormap.from_list("seq", SEQ), aspect="auto")
    lo, hi = np.nanmin(M), np.nanmax(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                dark = (M[i, j] - lo) / max(hi - lo, 1e-12) > 0.55
                ax.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center", fontsize=8,
                        color=SURFACE if dark else INK)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(norms)), norms)
    ax.set_xlabel("alpha depends on")
    ax.set_ylabel("normalisation")
    ax.grid(False)
    ax.set_title(f"Test NDCG@10 (protocol {P}, mean over seeds)", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.85).outline.set_visible(False)
    _save(fig, cfg, P, "heatmap_ndcg10")
    return pd.DataFrame(M, index=norms, columns=cols)


def _alphas(groups, key, s="test", seed_run=0):
    return np.load(groups[key][seed_run]["arrays"])[f"{s}__alpha"]


def alpha_hist(cfg, P, groups, lam_tag):
    norms = cfg["fusion"]["normalizers"]
    fig, axes = plt.subplots(len(norms), 3, figsize=(8, 1.7 * len(norms)), sharex=True, squeeze=False)
    for i, n in enumerate(norms):
        for j, d in enumerate(DEPS):
            ax = axes[i, j]
            k = f"{n}_{d}_listwise_{lam_tag}"
            if k in groups:
                a = _alphas(groups, k)
                ax.hist(a, bins=40, range=(0, 1), color=SERIES[d], edgecolor=SURFACE, linewidth=0.5)
                a0 = groups[k][0]["alpha0"]
                ax.axvline(a0, color=INK2, linewidth=1, linestyle="--")
            if i == 0:
                ax.set_title(f"alpha = f({d})", loc="left")
            if j == 0:
                ax.set_ylabel(f"{n}\n# queries")
    for ax in axes[-1]:
        ax.set_xlabel("learned alpha (test, seed 0); dashed = alpha0")
    fig.tight_layout()
    _save(fig, cfg, P, "alpha_hist")


def _scatter_trend(ax, x, y, color):
    ax.scatter(x, y, s=8, color=color, alpha=0.35, linewidths=0)
    if len(np.unique(x)) > 1:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, a + b * xs, color=INK, linewidth=2, label=f"trend: slope {b:+.4f}")
        ax.legend(loc="upper right", fontsize=7)


def alpha_scatter(cfg, P, groups, lam_tag, norm, test):
    qlen = test.qlen.numpy()
    sem = test.scores["sem"].masked_fill(~test.mask, float("-inf"))
    top2 = sem.topk(min(2, sem.shape[1]), dim=1).values
    top2 = top2 if top2.shape[1] == 2 else top2.repeat(1, 2)
    gap = (top2[:, 0] - top2[:, 1]).numpy()
    rng = np.random.default_rng(0)
    for name, x, xlabel in (("alpha_vs_query_length", qlen + rng.uniform(-0.2, 0.2, len(qlen)), "query length (words, jittered)"),
                            ("alpha_vs_ce_confidence", gap, "cross-encoder top1 - top2 logit gap")):
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
        for ax, d in zip(axes, DEPS):
            k = f"{norm}_{d}_listwise_{lam_tag}"
            if k in groups:
                _scatter_trend(ax, x, _alphas(groups, k), SERIES[d])
            ax.set_title(f"alpha = f({d}), norm = {norm}", loc="left")
            ax.set_xlabel(xlabel)
        axes[0].set_ylabel("learned alpha (test, seed 0)")
        fig.tight_layout()
        _save(fig, cfg, P, name)


def length_buckets(cfg, P, groups, lam_tag, norm, test):
    short, med = cfg["fusion"]["eval"]["length_buckets"]
    qlen = test.qlen.numpy()
    buckets = {f"short (<= {short})": qlen <= short, f"medium ({short + 1}-{med})": (qlen > short) & (qlen <= med),
               f"long (>= {med + 1})": qlen > med}
    keys = {"fixed": best_fixed_key(groups, norm, lam_tag), **{d: f"{norm}_{d}_listwise_{lam_tag}" for d in DEPS}}
    from src.evaluate import per_query_mean
    rows = []
    for d, k in keys.items():
        if k not in groups:
            continue
        pq = per_query_mean(groups[k], "test", "ndcg@10")
        for b, m in buckets.items():
            rows.append({"method": d, "bucket": b, "ndcg@10": pq[m].mean() if m.any() else np.nan, "n": int(m.sum())})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    methods = list(dict.fromkeys(df.method))
    w = 0.8 / len(methods)
    for i, d in enumerate(methods):
        sub = df[df.method == d]
        ax.bar(np.arange(len(buckets)) + (i - (len(methods) - 1) / 2) * w, sub["ndcg@10"], width=w * 0.92,
               color=SERIES[d], label=d, edgecolor=SURFACE, linewidth=1)
    n = df.drop_duplicates("bucket").set_index("bucket")["n"]
    ax.set_xticks(range(len(buckets)), [f"{b}\nn={n[b]}" for b in buckets])
    ax.set_ylabel("test NDCG@10")
    ax.set_ylim(0, None)          # bars start at zero; exact values are in ndcg_by_query_length.csv
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=len(methods), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title(f"NDCG@10 by query length (norm = {norm})", loc="left", pad=22)
    _save(fig, cfg, P, "ndcg_by_query_length")
    df.to_csv(results_path(cfg, P, "ndcg_by_query_length.csv"), index=False, float_format="%.4f")


def summary(cfg, P, groups, lam_tag, hm):
    lines = [f"### Protocol {P} (auto-generated by `src/analysis.py`; selection on val, numbers on test)", ""]
    norms = list(hm.index)
    # which normalisation wins: best config per normaliser, chosen on val
    best_per_norm = {}
    for n in norms:
        ks = [best_fixed_key(groups, n, lam_tag)] + [f"{n}_{d}_listwise_{lam_tag}" for d in DEPS
                                                    if f"{n}_{d}_listwise_{lam_tag}" in groups]
        k = max(ks, key=lambda k: _val(groups, k))
        best_per_norm[n] = (k, _test(groups, k))
    win_n = max(best_per_norm, key=lambda n: _val(groups, best_per_norm[n][0]))
    lines.append(f"- **Normalisation:** best (by val) is `{win_n}` "
                 f"(its best config `{best_per_norm[win_n][0]}`, test NDCG@10 {best_per_norm[win_n][1]:.4f}). "
                 "Per-normaliser test NDCG@10 of the val-best config: "
                 + ", ".join(f"{n} {v:.4f}" for n, (_, v) in best_per_norm.items()) + ".")
    dep_mean = {d: np.nanmean(hm[d].values) for d in hm.columns}
    lines.append("- **Alpha dependence:** test NDCG@10 averaged over normalisers: "
                 + ", ".join(f"{d} {v:.4f}" for d, v in dep_mean.items())
                 + f" -> best `{max(dep_mean, key=dep_mean.get)}`.")
    sig_p = results_path(cfg, P, "significance.csv")
    if sig_p.exists():
        sig = pd.read_csv(sig_p)
        sig = sig[(sig.set == "test") & (sig.metric == "ndcg@10") & sig.config.str.endswith(lam_tag)]
        wins = sig[(sig.ci95_lo > 0) & (sig.p_value < 0.05)]
        losses = sig[(sig.ci95_hi < 0) & (sig.p_value < 0.05)]
        lines.append(f"- **Adaptive vs best fixed (same normaliser, paired bootstrap, test NDCG@10):** "
                     f"{len(wins)}/{len(sig)} adaptive configs significantly better, {len(losses)} significantly "
                     "worse. " + "; ".join(f"`{r.config}` {r['diff']:+.4f} [{r.ci95_lo:+.4f}, {r.ci95_hi:+.4f}] "
                                           f"p={r.p_value:.3g}" for _, r in sig.iterrows()))
    og_p = results_path(cfg, P, "oracle_gap.csv")
    if og_p.exists():
        og = pd.read_csv(og_p)
        og = og[og.set == "test"]
        lines.append("- **Oracle headroom (test, oracle minus best fixed, UPPER BOUND):** "
                     + ", ".join(f"{r.norm} {r.oracle_minus_fixed:+.4f}" for _, r in og.iterrows()))
        small = og[og.small_headroom_flag.astype(bool)]
        if len(small):
            lines.append(f"  - **WARNING: headroom below {cfg['fusion']['eval']['oracle_flag_points']:.3f} for "
                         f"{', '.join(small.norm)}; this caps what adaptive alpha can gain.**")
    gaps = path(cfg, "results_dir", "sem_gap_stats.json")
    if gaps.exists():
        g = load_json(gaps)
        tr = [g[s]["mean_top1_minus_top2"] for s in ("gat_train", "fusion_train", "val") if s in g]
        if tr and "test" in g:
            lines.append(f"- **CE over-confidence caveat:** mean top1-top2 logit gap train {np.mean(tr):.3f} vs "
                         f"dev {g['test']['mean_top1_minus_top2']:.3f}.")
    text = "\n".join(lines) + "\n"
    results_path(cfg, P, "summary.md").write_text(text)
    if cfg["profile"] == "full":   # keep NOTES.md free of smoke-test numbers
        notes = ROOT / "NOTES.md"
        s = notes.read_text()
        b, e = f"<!-- RESULTS:{P}:BEGIN -->", f"<!-- RESULTS:{P}:END -->"
        if b in s:
            s = re.sub(re.escape(b) + ".*?" + re.escape(e), b + "\n" + text + e, s, flags=re.S)
            notes.write_text(s)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="A")
    args, _ = ap.parse_known_args(argv)
    cfg = load_config()
    set_env(cfg)
    P = args.protocol
    from src.train_fusion import raw_split
    groups = group_runs(load_runs(cfg, P))
    lam_tag = f"lam{cfg['fusion']['train']['lambda']:g}"
    hm = heatmap(cfg, P, groups, lam_tag)
    alpha_hist(cfg, P, groups, lam_tag)
    # scatter/bucket plots use the normaliser where QS is best on val
    norm = max((n for n in cfg["fusion"]["normalizers"] if f"{n}_QS_listwise_{lam_tag}" in groups),
               key=lambda n: _val(groups, f"{n}_QS_listwise_{lam_tag}"))
    test = raw_split(cfg, P, "test")
    alpha_scatter(cfg, P, groups, lam_tag, norm, test)
    length_buckets(cfg, P, groups, lam_tag, norm, test)
    log.info("\n%s", summary(cfg, P, groups, lam_tag, hm))


if __name__ == "__main__":
    main()
