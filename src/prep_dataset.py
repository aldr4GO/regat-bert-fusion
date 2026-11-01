"""Stage 0 for the multi-hop datasets: adapter -> queries, candidate pools, labels, meta, splits, BM25 scores.

Usage:  python -m src.prep_dataset --dataset musique|hotpotqa|beir_hotpotqa|twowiki
Writes: cache/<dataset>/{queries,qrels,meta,splits,passages,prep_stats}.json, cands_{split}.json, bm25_{split}.pt
        results/<dataset>/first_stage.json (BEIR HotpotQA: BM25 R@100 / AllR@100 on test)
"""
import argparse

import numpy as np

from src.common import dataset_cfg, get_logger, load_config, results_path, save_json, set_env
from src.datasets import adapter
from src.datasets.common import write_dataset

log = get_logger("prep_dataset")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    cfg = load_config()
    set_env(cfg)
    dcfg = dataset_cfg(cfg, args.dataset)
    obj = adapter(args.dataset).build(cfg, dcfg)
    if "smoke_test_size" in dcfg and dcfg["kind"] == "closed":   # smoke profile: subsample the test split
        rng = np.random.RandomState(cfg["data"]["seed"])
        obj["splits"]["test"] = sorted(rng.choice(obj["splits"]["test"], dcfg["smoke_test_size"],
                                                  replace=False).tolist())
    stats = write_dataset(cfg, args.dataset, obj)
    if "first_stage" in obj:
        save_json(obj["first_stage"], results_path(cfg, args.dataset, "first_stage.json"))
        stats["first_stage"] = obj["first_stage"]
    save_json(stats, results_path(cfg, args.dataset, "prep_stats.json"))


if __name__ == "__main__":
    main()
