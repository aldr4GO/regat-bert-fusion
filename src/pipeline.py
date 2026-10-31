"""Runs the whole study in order, skipping every stage whose outputs already exist (resumable).

Usage:  python -m src.pipeline                  # everything not done yet
        python -m src.pipeline --only fusion    # one stage (see STAGES)
        python -m src.pipeline --fresh          # smoke profile only: wipe smoke outputs first
Each stage runs as a subprocess, so GPU memory is released between stages. Timings are appended to
results/pipeline_timing.json.
"""
import argparse
import shutil
import subprocess
import sys
import time

from src.common import ROOT, cache_path, get_logger, load_config, load_json, path, results_path, save_json

log = get_logger("pipeline")


def py(*args):
    return [sys.executable, "-m", *args]


def stages(cfg, workers):
    """(name, done-marker check, commands)."""
    gat_ctrl = [c["name"] for c in cfg["gat"]["controls"]]
    sanity = lambda P: results_path(cfg, P, "stage4_sanity.json")
    st = [
        ("download", lambda: False, [py("src.download")]),               # cheap no-op when files exist
        ("data", lambda: path(cfg, "results_dir", "bm25_first_stage.json").exists(), [py("src.data")]),
        ("cross_encoder", lambda: all(cache_path(cfg, f"sem_{s}.pt").exists() for s in cfg["data"]["cache_sets"]),
         [py("src.cache_semantic", "--skip_existing", "--num_workers", "8")]),
        ("bi_encoder", lambda: path(cfg, "results_dir", "stage12_check.json").exists()
         and load_json(path(cfg, "results_dir", "stage12_check.json"))["accept"],
         [py("src.cache_embeddings")]),
    ]
    for P in ("A", "B"):
        st.append((f"gat_{P}", lambda P=P: cache_path(cfg, "gat_best.pt", P).exists(),
                   [py("src.train_gat", "--protocol", P)]))
        st.append((f"ctx_{P}", lambda P=P: sanity(P).exists() and load_json(sanity(P))["gate_passed"],
                   [py("src.cache_contextual", "--protocol", P)]))
    st.append(("tests", lambda: False, [py("pytest", "-q", "tests")]))
    for P in ("A", "B"):
        st.append((f"fusion_{P}", lambda P=P: results_path(cfg, P, "summary.md").exists(),
                   [py("src.run_all", "--protocol", P, "--workers", str(workers))]))
    # graph controls last: they only feed the ablation table, so main results come first
    st.append(("gat_controls",
               lambda: all(results_path(cfg, "A", "gat_controls", f"{c}.json").exists() for c in gat_ctrl),
               [py("src.train_gat", "--protocol", "A", "--control", c) for c in gat_ctrl]
               + [py("src.train_gat", "--protocol", "A", "--summarize")]))
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None, help="run only these stages (ignores done markers)")
    ap.add_argument("--workers", type=int, default=16, help="CPU workers for the fusion grid")
    ap.add_argument("--fresh", action="store_true", help="smoke profile only: delete smoke outputs first")
    args = ap.parse_args()
    cfg = load_config()
    if args.fresh:
        assert cfg["profile"] == "smoke", "--fresh is only allowed with REGAT_PROFILE=smoke"
        for key in ("data_dir", "cache_dir", "results_dir"):
            shutil.rmtree(ROOT / cfg["data"]["paths"][key], ignore_errors=True)
    timing_p = path(cfg, "results_dir", "pipeline_timing.json")
    timing = load_json(timing_p) if timing_p.exists() else {}
    for name, done, cmds in stages(cfg, args.workers):
        if args.only is not None and name not in args.only:
            continue
        if args.only is None and done():
            log.info("== %s: already done, skipping", name)
            continue
        if name == "gat_controls":   # resume the controls one by one
            cmds = [c for c in cmds if "--control" not in c
                    or not results_path(cfg, "A", "gat_controls", f"{c[-1]}.json").exists()]
        log.info("== %s: starting", name)
        t0 = time.time()
        for cmd in cmds:
            log.info("$ %s", " ".join(cmd[1:]))
            r = subprocess.run(cmd, cwd=ROOT)
            if r.returncode != 0:
                log.error("== %s FAILED (exit %d): %s", name, r.returncode, " ".join(cmd[1:]))
                sys.exit(r.returncode)
        timing[name] = round(time.time() - t0, 1)
        save_json(timing, timing_p)
        log.info("== %s: done in %.0fs", name, timing[name])
    log.info("pipeline finished; timings: %s", timing)


if __name__ == "__main__":
    main()
