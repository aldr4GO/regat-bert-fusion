"""Stage 5 driver: oracle/fixed grid -> 16-config grid x seeds (+ paper rows, loss and lambda ablations)
-> evaluation tables -> plots. CPU only; runs are parallelised over worker processes.

Usage:  python -m src.run_all --protocol A [--workers 16] [--overwrite]
        python -m src.run_all --protocol musique            # multi-hop dataset, full grid (all normalisers)
        python -m src.run_all --protocol musique@random --mode light   # H4 graph variant, primary normaliser
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


def run_inline(specs, overwrite=False):
    """Same as run_specs but in this process (tests, tiny subsets)."""
    return [_run({**s, "overwrite": overwrite}) for s in specs]


def grid_specs(cfg, protocol, norms, mode="full"):
    """full: base-plan grid (+ QSC under the primary normaliser); light: fixed/Q/QS/S only (H4 graph variants)."""
    fc = cfg["fusion"]
    lam = fc["train"]["lambda"]
    specs = []
    for norm in norms:
        for seed in fc["train"]["seeds"]:
            for dep in fc["dependencies"]:          # fixed = learned scalar; grid version is in oracle.py
                specs.append(dict(protocol=protocol, norm=norm, dep=dep, seed=seed, loss="listwise", lam=lam))
                if dep != "fixed" and mode == "full":
                    for l in fc["train"]["lambda_extra"]:
                        specs.append(dict(protocol=protocol, norm=norm, dep=dep, seed=seed, loss="listwise", lam=l))
            if mode == "full":
                specs.append(dict(protocol=protocol, norm=norm, dep="paper", seed=seed))
                if cfg["multihop"]["qsc"]["enabled"] and norm == primary_norm(cfg):
                    specs.append(dict(protocol=protocol, norm=norm, dep="QSC", seed=seed, loss="listwise", lam=lam))
    return specs


def primary_norm(cfg):
    from src.evaluate import primary_norm as pn
    return pn(cfg)


def loss_ablation_specs(cfg, protocol, norms):
    """fixed and QS with BCE, under the normaliser where QS (listwise) is best on val."""
    from src.evaluate import group_runs, load_runs
    fc = cfg["fusion"]
    groups = group_runs(load_runs(cfg, protocol))
    lam = fc["train"]["lambda"]
    val = {n: np.mean([r["metrics"]["val"]["ndcg@10"] for r in groups[f"{n}_QS_listwise_lam{lam:g}"]])
           for n in norms}
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="A", help="A | B | <dataset>[@variant]")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip_plots", action="store_true")
    ap.add_argument("--mode", default="full", choices=["full", "light"],
                    help="light = primary normaliser, fixed/Q/QS/S only (H4 graph variants)")
    ap.add_argument("--norms", nargs="*", default=None)
    ap.add_argument("--no_qsc", action="store_true", help="skip the E5 QSC runs (golden regression)")
    args = ap.parse_args(argv)
    cfg = load_config()
    set_env(cfg)
    if args.no_qsc:
        cfg["multihop"]["qsc"]["enabled"] = False
    rc = cfg["fusion"]["run_all"]
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", rc["workers"]))
    P = args.protocol
    norms = args.norms or ([primary_norm(cfg)] if args.mode == "light" else cfg["fusion"]["normalizers"])

    from src import evaluate, oracle
    from src.train_fusion import cfeat_scaler, feature_scaler
    # 1. references, fixed-alpha grid (alpha0) and oracle
    evaluate.reference_runs(cfg, P)
    oracle.main(["--protocol", P, "--norms", *norms])
    feature_scaler(cfg, P)               # fit on fusion_train once, before the workers start
    if args.mode == "full" and cfg["multihop"]["qsc"]["enabled"]:
        cfeat_scaler(cfg, P)
    # 2. grid
    specs = grid_specs(cfg, P, norms, args.mode)
    log.info("[%s] %d grid runs on %d workers (mode %s, norms %s)", P, len(specs), workers, args.mode, norms)
    run_specs(specs, workers, rc["threads_per_worker"], args.overwrite)
    # 3. loss ablation (needs the grid results to pick the normaliser on val)
    if args.mode == "full":
        run_specs(loss_ablation_specs(cfg, P, norms), workers, rc["threads_per_worker"], args.overwrite)
    # 4. tables + plots
    evaluate.main(["--protocol", P])
    if not args.skip_plots:
        from src import analysis
        analysis.main(["--protocol", P])
    log.info("[%s] all done -> %s", P, results_path(cfg, P, "main_table.csv"))


if __name__ == "__main__":
    main()
