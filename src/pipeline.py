"""Runs the whole study in order, skipping every stage whose outputs already exist (resumable).

Usage:  python -m src.pipeline                        # everything not done yet (MS MARCO + multi-hop extension)
        python -m src.pipeline --part msmarco         # only the original MS MARCO study
        python -m src.pipeline --part multihop        # only the extension (needs the MS MARCO results)
        python -m src.pipeline --only fusion_A musique:fusion   # named stages (ignores done-markers)
        python -m src.pipeline --list                 # print the stage names
        python -m src.pipeline --fresh                # smoke profile only: wipe smoke outputs first
Each stage runs as a subprocess, so GPU memory is released between stages. Timings are appended to
results/pipeline_timing.json. A failing MS MARCO / golden-regression stage stops the pipeline; a failing stage
of a multi-hop dataset skips that dataset's remaining stages, the others continue, and the job exits non-zero.
"""
import argparse
import shutil
import subprocess
import sys
import time

from src.common import (ROOT, cache_path, data_cache_path, get_logger, load_config, load_json, path, results_path,
                        save_json)

log = get_logger("pipeline")


def py(*args):
    return [sys.executable, "-m", *args]


def _json_ok(p, key):
    return p.exists() and bool(load_json(p).get(key))


def msmarco_stages(cfg, workers):
    """(name, group, done-marker, commands); group None = global stage."""
    gat_ctrl = [c["name"] for c in cfg["gat"]["controls"]]
    sanity = lambda P: results_path(cfg, P, "stage4_sanity.json")
    st = [
        ("download", None, lambda: False, [py("src.download")]),               # cheap no-op when files exist
        ("data", None, lambda: path(cfg, "results_dir", "bm25_first_stage.json").exists(), [py("src.data")]),
        ("cross_encoder", None,
         lambda: all(cache_path(cfg, f"sem_{s}.pt").exists() for s in cfg["data"]["cache_sets"]),
         [py("src.cache_semantic", "--skip_existing", "--num_workers", "8")]),
        ("bi_encoder", None, lambda: _json_ok(path(cfg, "results_dir", "stage12_check.json"), "accept"),
         [py("src.cache_embeddings")]),
    ]
    for P in ("A", "B"):
        st.append((f"gat_{P}", None, lambda P=P: cache_path(cfg, "gat_best.pt", P).exists(),
                   [py("src.train_gat", "--protocol", P)]))
        st.append((f"ctx_{P}", None, lambda P=P: _json_ok(sanity(P), "gate_passed"),
                   [py("src.cache_contextual", "--protocol", P)]))
    st.append(("tests", None, lambda: False, [py("pytest", "-q", "tests")]))
    for P in ("A", "B"):
        st.append((f"fusion_{P}", None, lambda P=P: results_path(cfg, P, "summary.md").exists(),
                   [py("src.run_all", "--protocol", P, "--workers", str(workers))]))
    # graph controls last: they only feed the ablation table, so main results come first
    st.append(("gat_controls", None,
               lambda: all(results_path(cfg, "A", "gat_controls", f"{c}.json").exists() for c in gat_ctrl),
               [py("src.train_gat", "--protocol", "A", "--control", c) for c in gat_ctrl]
               + [py("src.train_gat", "--protocol", "A", "--summarize")]))
    return st


def dataset_core(cfg, ds, workers):
    sets = ["gat_train", "fusion_train", "val", "test"]
    ctx_ok = lambda k: (lambda: results_path(cfg, k, "stage4_sanity.json").exists() and (
        load_json(results_path(cfg, k, "stage4_sanity.json"))["gate_passed"]
        or not load_json(results_path(cfg, k, "stage4_sanity.json"))["gate_enforced"]))
    st = [
        (f"{ds}:download", ds, lambda: False, [py("src.datasets.sources", "--dataset", ds)]),
        (f"{ds}:prep", ds, lambda: data_cache_path(cfg, "prep_stats.json", ds).exists(),
         [py("src.prep_dataset", "--dataset", ds)]),
        (f"{ds}:cross_encoder", ds, lambda: all(data_cache_path(cfg, f"sem_{s}.pt", ds).exists() for s in sets),
         [py("src.cache_semantic", "--protocol", ds, "--skip_existing", "--num_workers", "8")]),
        (f"{ds}:bi_encoder", ds, lambda: _json_ok(results_path(cfg, ds, "stage12_check.json"), "accept"),
         [py("src.cache_embeddings", "--protocol", ds)]),
        (f"{ds}:graph_prep", ds, lambda: data_cache_path(cfg, "edge_stats.json", ds).exists(),
         [py("src.graph_prep", "--protocol", ds)]),
        (f"{ds}:gat", ds, lambda: cache_path(cfg, "gat_best.pt", ds).exists(), [py("src.train_gat", "--protocol", ds)]),
        (f"{ds}:ctx", ds, ctx_ok(ds), [py("src.cache_contextual", "--protocol", ds)]),
        (f"{ds}:fusion", ds, lambda: results_path(cfg, ds, "summary.md").exists(),
         [py("src.run_all", "--protocol", ds, "--workers", str(workers))]),
    ]
    if ds in cfg["multihop"]["transfer"]["zero_shot_datasets"]:
        k = f"{ds}@msmarco_gat"
        st.append((f"{k}:ctx", ds, lambda k=k: results_path(cfg, k, "stage4_sanity.json").exists(),
                   [py("src.cache_contextual", "--protocol", k)]))
    return st


def dataset_h4(cfg, ds, workers):
    st = []
    for gt in cfg["multihop"]["graph_types"]:
        if gt == "knn_emb":
            continue    # the main GAT of the dataset
        k = f"{ds}@{gt}"
        st += [(f"{k}:gat", ds, lambda k=k: cache_path(cfg, "gat_best.pt", k).exists(),
                [py("src.train_gat", "--protocol", k)]),
               (f"{k}:ctx", ds, lambda k=k: results_path(cfg, k, "stage4_sanity.json").exists(),
                [py("src.cache_contextual", "--protocol", k)]),
               (f"{k}:fusion", ds, lambda k=k: results_path(cfg, k, "summary.md").exists(),
                [py("src.run_all", "--protocol", k, "--mode", "light", "--workers", str(workers))])]
    names = [f"knn_k{x}" for x in cfg["multihop"]["knn_sweep"]]
    st.append((f"{ds}:knn_sweep", ds,
               lambda: all(results_path(cfg, ds, "gat_controls", f"{n}.json").exists() for n in names),
               [py("src.train_gat", "--protocol", ds, "--control", n) for n in names]
               + [py("src.train_gat", "--protocol", ds, "--summarize")]))
    return st


def multihop_stages(cfg, workers):
    mh = cfg["multihop"]
    on = [ds for ds, d in mh["datasets"].items() if d["enabled"]]
    st = []
    for P in ("A", "B"):     # acceptance gate of the refactor (plan Sec. 3.7), then use the new runs
        st.append((f"golden_{P}", None,
                   lambda P=P: _json_ok(results_path(cfg, P, "golden_regression.json"), "pass")
                   and not load_json(results_path(cfg, P, "golden_regression.json")).get("subset")
                   and any((ROOT / cfg["data"]["paths"]["results_dir"] / P / "runs_v1").glob("*.json")),
                   [py("src.golden_regression", "--protocol", P, "--promote", "--workers", str(workers))]))
    st.append(("diagnose_msmarco", None, lambda: path(cfg, "results_dir", "diagnostics_msmarco.json").exists(),
               [py("src.diagnose_msmarco")]))
    st.append(("msmarco_edge_stats", None, lambda: data_cache_path(cfg, "edge_stats.json", "A").exists(),
               [py("src.graph_prep", "--protocol", "A")]))
    closed = [ds for ds in on if mh["datasets"][ds]["kind"] == "closed" and ds != "twowiki"]
    for ds in closed:                                   # (1) MuSiQue, HotpotQA: E1 + oracle
        st += dataset_core(cfg, ds, workers)
    for ds in mh["h4_datasets"]:                        # E3 graph controls (H4) + k_nn sweep
        if ds in on:
            st += dataset_h4(cfg, ds, workers)
    if "beir_hotpotqa" in on:                           # (2) open retrieval
        st += dataset_core(cfg, "beir_hotpotqa", workers)
    st.append(("tests_multihop", None, lambda: False, [py("pytest", "-q", "tests")]))
    st.append(("transfer", None, lambda: path(cfg, "results_dir", "transfer.csv").exists(), [py("src.transfer")]))
    if "twowiki" in on:                                 # (4) optional
        st += dataset_core(cfg, "twowiki", workers)
    st.append(("hypotheses", None, lambda: False, [py("src.hypotheses")]))
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None, help="run only these stages (ignores done markers)")
    ap.add_argument("--part", default="all", choices=["all", "msmarco", "multihop"])
    ap.add_argument("--workers", type=int, default=16, help="CPU workers for the fusion grids")
    ap.add_argument("--fresh", action="store_true", help="smoke profile only: delete smoke outputs first")
    ap.add_argument("--list", action="store_true", help="print the stages and exit")
    args = ap.parse_args()
    cfg = load_config()
    if args.fresh:
        assert cfg["profile"] == "smoke", "--fresh is only allowed with REGAT_PROFILE=smoke"
        for key in ("data_dir", "cache_dir", "results_dir"):
            shutil.rmtree(ROOT / cfg["data"]["paths"][key], ignore_errors=True)
    st = []
    if args.part in ("all", "msmarco"):
        st += msmarco_stages(cfg, args.workers)
    if args.part in ("all", "multihop"):
        st += multihop_stages(cfg, args.workers)
    if args.list:
        print("\n".join(f"{n}  [{g or 'global'}]" for n, g, _, _ in st))
        return
    timing_p = path(cfg, "results_dir", "pipeline_timing.json")
    timing = load_json(timing_p) if timing_p.exists() else {}
    failed = {}
    for name, group, done, cmds in st:
        if args.only is not None and name not in args.only:
            continue
        if group in failed:
            log.warning("== %s: skipped because %s failed earlier (%s)", name, group, failed[group])
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
                if group is None:
                    sys.exit(r.returncode)
                failed[group] = name
                break
        if group in failed:
            continue
        timing[name] = round(time.time() - t0, 1)
        save_json(timing, timing_p)
        log.info("== %s: done in %.0fs", name, timing[name])
    log.info("pipeline finished; timings: %s", timing)
    if failed:
        log.error("FAILED datasets: %s (their later stages were skipped; see the log above)", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
