"""Golden regression (multi-hop plan Sec. 3.7): rerun the MS MARCO fusion grid on the existing caches with the
refactored (masked, multi-dataset) code and compare with the results produced by the original code.

Usage:  python -m src.golden_regression --protocol A [--workers 32] [--subset] [--promote]
  --subset   only seed 0 of fixed/Q/QS/S/paper per normaliser (quick check)
  --promote  after a pass, archive results/{P}/runs as runs_v1 and use the new runs (which carry the extended
             metric set), then rebuild the tables and figures
Writes: results/{P}/golden_regression.json. Pass = every metric, per-query value and learned alpha within 1e-6.
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

from src.common import ROOT, get_logger, load_config, load_json, results_path, save_json

log = get_logger("golden")
TOL = 1e-6


def compare(ref_dir, new_dir, names=None):
    worst, n, problems = {}, 0, []
    for ref in sorted(ref_dir.glob("*.json")):
        if names is not None and ref.stem not in names:
            continue
        if "_QSC_" in ref.stem:          # E5 runs did not exist in the original grid
            continue
        new = new_dir / ref.name
        if not new.exists():
            problems.append(f"missing {ref.name}")
            continue
        a, b = load_json(ref), load_json(new)
        for s, ms in a["metrics"].items():
            for m, v in ms.items():
                d = abs(v - b["metrics"][s][m])
                worst[f"agg:{m}"] = max(worst.get(f"agg:{m}", 0.0), d)
        za, zb = np.load(ref.with_suffix(".npz")), np.load(new.with_suffix(".npz"))
        for k in za.files:
            if k in zb.files:
                d = float(np.abs(za[k].astype(np.float64) - zb[k].astype(np.float64)).max()) if za[k].size else 0.0
                tag = "alpha" if k.endswith("__alpha") else "per_query:" + k.split("__")[1]
                worst[tag] = max(worst.get(tag, 0.0), d)
        n += 1
    ok = n > 0 and not problems and all(v <= TOL for v in worst.values())
    return {"n_runs_compared": n, "max_abs_diff": worst, "problems": problems, "tolerance": TOL, "pass": ok}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="A", choices=["A", "B"])
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--subset", action="store_true")
    ap.add_argument("--promote", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    P = args.protocol
    res_root = ROOT / cfg["data"]["paths"]["results_dir"] / P       # plain paths: never create directories here
    ref_dir, archive = res_root / "runs", res_root / "runs_v1"
    if archive.exists() and not any(ref_dir.glob("*.json")):
        ref_dir = archive
    reg_root = ROOT / (cfg["data"]["paths"]["results_dir"] + "_regression")
    env = {**os.environ, "REGAT_RESULTS_DIR": str(reg_root)}
    new_dir = reg_root / P / "runs"
    names = None
    if args.subset:
        specs = [dict(protocol=P, norm=n, dep=d, seed=0, loss="listwise", lam=cfg["fusion"]["train"]["lambda"])
                 for n in cfg["fusion"]["normalizers"] for d in ("fixed", "Q", "QS", "S")]
        specs += [dict(protocol=P, norm=n, dep="paper", seed=0) for n in cfg["fusion"]["normalizers"]]
        code = (f"from src.common import load_config, set_env; from src import evaluate, oracle; "
                f"from src.run_all import run_specs; cfg=load_config(); set_env(cfg); "
                f"evaluate.reference_runs(cfg, '{P}'); oracle.main(['--protocol', '{P}']); "
                f"run_specs({specs!r}, {args.workers}, 1, True)")
        cmd = [sys.executable, "-c", code]
        from src.train_fusion import run_name
        names = {f"{P}_ref_{r}" for r in ("bm25", "bi", "ce", "gat")}
        names |= {f"{P}_{n}_{k}" for n in cfg["fusion"]["normalizers"] for k in ("fixedgrid", "oracle")}
        names |= {run_name(P, s["norm"], s["dep"], s.get("loss", "bce"), s.get("lam", 0.0), 0) for s in specs}
    else:
        cmd = [sys.executable, "-m", "src.run_all", "--protocol", P, "--workers", str(args.workers), "--no_qsc",
               "--skip_plots", "--overwrite"]
    log.info("rerunning %s grid into %s", P, reg_root)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    res = compare(ref_dir, new_dir, names)
    res["subset"] = args.subset
    save_json(res, results_path(cfg, P, "golden_regression.json"))
    log.info("golden regression %s: %s", P, res)
    if not res["pass"]:
        raise SystemExit(f"GOLDEN REGRESSION FAILED for {P}: {res}")
    if args.promote and not args.subset:
        if not archive.exists() or not any(archive.iterdir()):
            shutil.rmtree(archive, ignore_errors=True)
            shutil.move(str(ref_dir), str(archive))
        shutil.rmtree(ref_dir, ignore_errors=True)
        shutil.copytree(new_dir, ref_dir)
        for f in archive.iterdir():      # keep runs the regression did not regenerate (e.g. QSC)
            if not (ref_dir / f.name).exists():
                shutil.copy(f, ref_dir / f.name)
        for f in ("alpha0.json", "oracle_summary.json"):
            shutil.copy(reg_root / P / f, results_path(cfg, P, f))
        subprocess.run([sys.executable, "-m", "src.evaluate", "--protocol", P], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-m", "src.analysis", "--protocol", P], cwd=ROOT, check=True)
        log.info("promoted the new %s runs (old ones archived in %s)", P, archive)


if __name__ == "__main__":
    main()
