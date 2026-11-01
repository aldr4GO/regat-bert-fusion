"""E4: transfer and mixed training of the gate (multi-hop plan Sec. 4.4). Primary normaliser, listwise loss.

 zero_shot  gate trained on MS MARCO (protocol A) fusion_train, applied to each multi-hop test set whose s_ctx
            comes from the frozen MS MARCO GAT (key <dataset>@msmarco_gat); scaler and alpha0 from MS MARCO.
 mixed      one gate trained on the union of fusion_train sets (MS MARCO + MuSiQue + HotpotQA, in-domain GATs),
            no dataset-id feature; early stopping on the union of val sets; alpha0 = best fixed alpha on that union.
 lodo       leave one dataset out: train on MS MARCO + HotpotQA, test on MuSiQue (3-4 hop questions never seen).
Every setting is compared with the in-domain gate of the same variant (results/<dataset>/runs).

Usage:  python -m src.transfer [--seeds 0 1 2 3 4]
Writes: results/transfer/runs/*.json|npz, results/transfer.csv
"""
import argparse

import numpy as np
import pandas as pd
import torch

from src.common import get_logger, load_config, load_json, path, results_path, save_json, set_env
from src.evaluate import group_runs, load_runs, per_query, primary_norm
from src.features import FeatureScaler
from src.fusion import fuse
from src.metrics import ndcg_at_k
from src.oracle import alpha_grid
from src.train_fusion import FusionData, predict, raw_features, raw_split, train_fusion
from src.normalize import normalize

log = get_logger("transfer")
VARIANTS = ("fixed", "Q", "QS", "S")


def make_data(cfg, key, split, norm, scaler):
    d = raw_split(cfg, key, split)
    eps = cfg["fusion"]["eps"]
    return FusionData(d.q_emb, scaler(raw_features(cfg, key, split)),
                      normalize(norm, d.scores["sem"], eps, d.mask), normalize(norm, d.scores["ctx"], eps, d.mask),
                      d.labels, d.gains, d.ideal, d.mask)


def concat(datas):
    """Concatenate FusionData of different pool sizes (padded to the largest K, masked)."""
    K = max(x.sem_n.shape[1] for x in datas)
    pad = lambda t, v: torch.nn.functional.pad(t, (0, K - t.shape[1]), value=v)
    cat = lambda f, v: torch.cat([pad(getattr(x, f), v) for x in datas])
    return FusionData(torch.cat([x.q_emb for x in datas]), torch.cat([x.feats for x in datas]), cat("sem_n", 0.0),
                      cat("ctx_n", 0.0), cat("labels", 0.0), cat("gains", 0.0), torch.cat([x.ideal for x in datas]),
                      torch.cat([pad(x.mask, False) for x in datas]))


def union_scaler(cfg, keys):
    F = torch.cat([raw_features(cfg, k, "fusion_train") for k in keys])
    return FeatureScaler(eps=cfg["fusion"]["eps"]).fit(F)


def key_scaler(cfg, key):
    from src.train_fusion import feature_scaler
    return feature_scaler(cfg, key)


def grid_alpha0(cfg, val):
    grid = alpha_grid(cfg["fusion"]["alpha_grid_step"])
    curve = torch.stack([ndcg_at_k(fuse(val.sem_n, val.ctx_n, float(a)), val.gains, val.ideal, mask=val.mask).mean()
                         for a in grid])
    a0 = float(grid[int(curve.argmax())])
    lo, hi = cfg["fusion"]["alpha0_clip"]
    return min(max(a0, lo), hi)


def evaluate_on(cfg, model, test_key, norm, scaler, name, meta):
    d = raw_split(cfg, test_key, "test")
    f, a = predict(model, make_data(cfg, test_key, "test", norm, scaler))
    pq = per_query(d, f, cfg)
    p = results_path(cfg, "transfer", "runs", f"{name}.json")
    np.savez_compressed(p.with_suffix(".npz"), test__alpha=a.numpy(), test__ndcg=pq["ndcg@10"].astype(np.float32))
    m = {k: float(v.mean()) for k, v in pq.items()}
    save_json({"name": name, **meta, "metrics": {"test": m}, "alpha_mean": float(a.mean()),
               "alpha_std": float(a.std())}, p)
    return m, a


def in_domain(cfg, key, norm, dep, lt):
    if not results_path(cfg, key, "runs").exists():
        return None
    g = group_runs(load_runs(cfg, key))
    k = f"{norm}_{dep}_listwise_{lt}"
    if k not in g:
        return None
    return {"ndcg@10": np.mean([r["metrics"]["test"]["ndcg@10"] for r in g[k]]),
            "alpha_mean": np.mean([np.load(r["arrays"])["test__alpha"].mean() for r in g[k]])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    set_env(cfg)
    tc = cfg["multihop"]["transfer"]
    fc = cfg["fusion"]
    pn = primary_norm(cfg)
    lt = f"lam{fc['train']['lambda']:g}"
    seeds = args.seeds if args.seeds is not None else fc["train"]["seeds"]
    have = lambda key: results_path(cfg, key, "stage4_sanity.json").exists()
    rows = []

    settings = []
    # 1. zero-shot: MS MARCO gate -> <ds>@msmarco_gat
    a_scaler = key_scaler(cfg, "A")
    a_alpha0 = min(max(load_json(results_path(cfg, "A", "alpha0.json"))[pn]["alpha0"], fc["alpha0_clip"][0]),
                   fc["alpha0_clip"][1])
    zs = [ds for ds in tc["zero_shot_datasets"] if have(f"{ds}@msmarco_gat")]
    if zs:
        settings.append(("zero_shot", ["A"], [(ds, f"{ds}@msmarco_gat") for ds in zs], a_scaler, a_alpha0))
    # 2. mixed and 3. leave-one-dataset-out
    for setting, train_keys, test_list in (("mixed", tc["mixed"], [(k, k) for k in tc["mixed"]]),
                                           ("lodo", tc["lodo"]["train"], [(tc["lodo"]["test"], tc["lodo"]["test"])])):
        if all(k == "A" or have(k) for k in train_keys + [t for _, t in test_list]):
            sc = union_scaler(cfg, train_keys)
            val = concat([make_data(cfg, k, "val", pn, sc) for k in train_keys])
            settings.append((setting, train_keys, test_list, sc, grid_alpha0(cfg, val)))
        else:
            log.warning("skipping %s: missing caches for %s", setting, train_keys)

    for setting, train_keys, test_list, scaler, alpha0 in settings:
        tr = concat([make_data(cfg, k, "fusion_train", pn, scaler) for k in train_keys])
        va = concat([make_data(cfg, k, "val", pn, scaler) for k in train_keys])
        for dep in VARIANTS:
            for seed in seeds:
                model, info = train_fusion(fc, dep, alpha0, seed, tr, va, "listwise", fc["train"]["lambda"])
                for ds, test_key in test_list:
                    name = f"{setting}_{'+'.join(train_keys)}_to_{ds}_{pn}_{dep}_s{seed}"
                    m, a = evaluate_on(cfg, model, test_key, pn, scaler, name,
                                       {"setting": setting, "train": train_keys, "test": ds, "test_key": test_key,
                                        "norm": pn, "dep": dep, "seed": seed, "alpha0": alpha0, **info})
                    rows.append({"setting": setting, "train": "+".join(train_keys), "test": ds, "variant": dep,
                                 "seed": seed, "alpha0": alpha0, "alpha_mean": float(a.mean()),
                                 "alpha_std": float(a.std()), **{k: m[k] for k in ("ndcg@10", "mrr@10", "map")}})
            log.info("%s %s done", setting, dep)
    df = pd.DataFrame(rows)
    if len(df):
        agg = df.groupby(["setting", "train", "test", "variant"]).agg(
            ndcg10_mean=("ndcg@10", "mean"), ndcg10_std=("ndcg@10", "std"), mrr10_mean=("mrr@10", "mean"),
            map_mean=("map", "mean"), alpha_mean=("alpha_mean", "mean"), alpha_std_over_queries=("alpha_std", "mean"),
            alpha0=("alpha0", "first"), n_seeds=("seed", "count")).reset_index()
        indom = [in_domain(cfg, t, pn, v, lt) for t, v in zip(agg.test, agg.variant)]
        agg["in_domain_ndcg10"] = [x["ndcg@10"] if x else np.nan for x in indom]
        agg["in_domain_alpha_mean"] = [x["alpha_mean"] if x else np.nan for x in indom]
        agg["ndcg10_minus_in_domain"] = agg.ndcg10_mean - agg.in_domain_ndcg10
        agg.to_csv(path(cfg, "results_dir", "transfer.csv"), index=False, float_format="%.5f")
        log.info("transfer:\n%s", agg.to_string(index=False))


if __name__ == "__main__":
    main()
