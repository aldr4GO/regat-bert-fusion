"""Fixed-alpha grid search on val (alpha0, the main baseline) and the per-query oracle alpha (upper bound).

Usage:  python -m src.oracle --protocol A
Writes: results/{P}/alpha0.json, results/{P}/oracle_summary.json,
        run files {P}_{norm}_fixedgrid and {P}_{norm}_oracle
"""
import argparse

import numpy as np
import torch

from src.common import get_logger, load_config, results_path, save_json, set_env
from src.fusion import fuse
from src.metrics import ndcg_at_k
from src.normalize import normalize
from src.train_fusion import raw_split

log = get_logger("oracle")


def alpha_grid(step):
    n = int(round(1 / step))
    return torch.tensor([round(i * step, 10) for i in range(n + 1)])


def per_alpha_ndcg(sem_n, ctx_n, gains, ideal, grid, cutoff=10):
    """-> [len(grid), N] per-query NDCG@cutoff for each alpha."""
    return torch.stack([ndcg_at_k(fuse(sem_n, ctx_n, float(a)), gains, ideal, cutoff) for a in grid])


def oracle_alpha(nd, grid, alpha0):
    """Per-query argmax over the grid; ties -> alpha closest to alpha0."""
    best = nd.max(0).values
    is_best = nd >= best.unsqueeze(0) - 1e-12
    dist = (grid - alpha0).abs().unsqueeze(1).expand_as(nd).clone()
    dist[~is_best] = float("inf")
    return grid[dist.argmin(0)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="A", choices=["A", "B"])
    args = ap.parse_args()
    cfg = load_config()
    set_env(cfg)
    from src.evaluate import save_run
    fc, P = cfg["fusion"], args.protocol
    grid = alpha_grid(fc["alpha_grid_step"])
    cutoff, eps = fc["eval"]["cutoff"], fc["eps"]
    sets = ["val"] + fc["eval"]["test_sets"]
    data = {s: raw_split(cfg, P, s) for s in sets}

    alpha0, summary = {}, {}
    for norm in fc["normalizers"]:
        nz = {s: (normalize(norm, d.scores["sem"], eps), normalize(norm, d.scores["ctx"], eps)) for s, d in data.items()}
        # --- (a) grid search on val only; ties -> first (smallest) alpha
        v = data["val"]
        curve = per_alpha_ndcg(*nz["val"], v.gains, v.ideal, grid, cutoff).mean(1)
        a0 = float(grid[int(curve.argmax())])
        alpha0[norm] = {"alpha0": a0, "val_ndcg@10": float(curve.max()),
                        "val_curve": {f"{float(a):.2f}": float(c) for a, c in zip(grid, curve)}}
        save_run(cfg, P, f"{P}_{norm}_fixedgrid", {"protocol": P, "norm": norm, "dep": "fixedgrid", "alpha0": a0},
                 {s: fuse(*nz[s], a0) for s in sets}, {s: torch.full((len(data[s]),), a0) for s in sets})
        # --- oracle: per query, alpha maximising that query's NDCG@10 (UPPER BOUND, not a method)
        o_scores, o_alpha = {}, {}
        for s in sets:
            d = data[s]
            nd = per_alpha_ndcg(*nz[s], d.gains, d.ideal, grid, cutoff)
            a = oracle_alpha(nd, grid, a0)
            o_scores[s], o_alpha[s] = fuse(*nz[s], a), a
        m = save_run(cfg, P, f"{P}_{norm}_oracle", {"protocol": P, "norm": norm, "dep": "oracle", "alpha0": a0,
                                                    "note": "upper bound, not a method"}, o_scores, o_alpha)
        fixed_val = alpha0[norm]["val_ndcg@10"]
        gap = m["val"]["ndcg@10"] - fixed_val
        summary[norm] = {"alpha0": a0, "val_fixed_ndcg@10": fixed_val, "val_oracle_ndcg@10": m["val"]["ndcg@10"],
                         "val_oracle_minus_fixed": gap,
                         "small_headroom_flag": bool(gap < fc["eval"]["oracle_flag_points"])}
        log.info("[%s] %-6s alpha0=%.2f val fixed %.4f oracle %.4f (headroom %.4f)%s", P, norm, a0, fixed_val,
                 m["val"]["ndcg@10"], gap, "  <-- SMALL HEADROOM" if gap < fc["eval"]["oracle_flag_points"] else "")
    save_json(alpha0, results_path(cfg, P, "alpha0.json"))
    save_json(summary, results_path(cfg, P, "oracle_summary.json"))


if __name__ == "__main__":
    main()
