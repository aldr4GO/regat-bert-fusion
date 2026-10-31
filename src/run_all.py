"""Stage 5 driver: oracle/fixed grid -> 16-config grid x seeds (+ paper rows, loss and lambda ablations)
-> evaluation tables -> plots. CPU only; runs are parallelised over worker processes.

Usage:  python -m src.run_all --protocol A [--workers 16] [--overwrite]
"""
import argparse
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from src.common import get_logger, load_config, load_json, results_path, set_env

log = get_logger("run_all")


def _worker_init(threads):
    import torch
    torch.set_num_threads(threads)


def _run(spec):
    from src.train_fusion import run_one
    cfg = load_config()
    set_env(cfg)
    return run_one(cfg, **spec)


def grid_specs(cfg, protocol):
    fc = cfg["fusion"]
    lam = fc["train"]["lambda"]
    specs = []
    for norm in fc["normalizers"]:
        for seed in fc["train"]["seeds"]:
            for dep in fc["dependencies"]:          # fixed = learned scalar; grid version is in oracle.py
                specs.append(dict(protocol=protocol, norm=norm, dep=dep, seed=seed, loss="listwise", lam=lam))
                if dep != "fixed":
                    for l in fc["train"]["lambda_extra"]:
                        specs.append(dict(protocol=protocol, norm=norm, dep=dep, seed=seed, loss="listwise", lam=l))
            specs.append(dict(protocol=protocol, norm=norm, dep="paper", seed=seed))
    return specs


def loss_ablation_specs(cfg, protocol):
    """fixed and QS with BCE, under the normaliser where QS (listwise) is best on val."""
    from src.evaluate import group_runs, load_runs
    fc = cfg["fusion"]
    groups = group_runs(load_runs(cfg, protocol))
    lam = fc["train"]["lambda"]
    val = {n: np.mean([r["metrics"]["val"]["ndcg@10"] for r in groups[f"{n}_QS_listwise_lam{lam:g}"]])
           for n in fc["normalizers"]}
    best = max(val, key=val.get)
    log.info("loss ablation normaliser (best QS on val): %s", best)
    return [dict(protocol=protocol, norm=best, dep=dep, seed=s, loss="bce", lam=lam)
            for dep in fc["loss_ablation"]["deps"] for s in fc["train"]["seeds"]]


def run_specs(specs, workers, threads, overwrite):
    t0 = time.time()
    for s in specs:
        s["overwrite"] = overwrite
    # spawn: forking after torch/OpenMP initialisation can deadlock
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_worker_init,
                             initargs=(threads,)) as ex:
        futs = [ex.submit(_run, s) for s in specs]
        for i, f in enumerate(as_completed(futs), 1):
            name = f.result()
            if i % 10 == 0 or i == len(futs):
                log.info("%d/%d runs done (%s) %.0fs", i, len(futs), name, time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="A", choices=["A", "B"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip_plots", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    set_env(cfg)
    rc = cfg["fusion"]["run_all"]
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", rc["workers"]))
    P = args.protocol

    from src import evaluate, oracle
    from src.train_fusion import feature_scaler, raw_split
    import sys
    # 1. references, fixed-alpha grid (alpha0) and oracle
    evaluate.reference_runs(cfg, P)
    sys.argv = [sys.argv[0], "--protocol", P]
    oracle.main()
    feature_scaler(cfg, P)               # fit on fusion_train once, before the workers start
    # 2. grid
    specs = grid_specs(cfg, P)
    log.info("[%s] %d grid runs on %d workers", P, len(specs), workers)
    run_specs(specs, workers, rc["threads_per_worker"], args.overwrite)
    # 3. loss ablation (needs the grid results to pick the normaliser on val)
    run_specs(loss_ablation_specs(cfg, P), workers, rc["threads_per_worker"], args.overwrite)
    # 4. tables + plots
    evaluate.main()
    if not args.skip_plots:
        from src import analysis
        analysis.main()
    log.info("[%s] all done -> %s", P, results_path(cfg, P, "main_table.csv"))


if __name__ == "__main__":
    main()
