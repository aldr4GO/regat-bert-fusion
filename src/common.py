"""Shared helpers: config loading, paths, cache I/O, seeding, logging."""
import copy
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def _deep_update(base, upd):
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def profile():
    return os.environ.get("REGAT_PROFILE", "full")


def load_config():
    """Returns {'data', 'gat', 'fusion', 'multihop'} configs; REGAT_PROFILE=smoke applies configs/smoke.yaml."""
    cfg = {}
    for name in ("data", "gat", "fusion", "multihop"):
        with open(CONFIG_DIR / f"{name}.yaml") as f:
            cfg[name] = yaml.safe_load(f)
    if profile() == "smoke":
        with open(CONFIG_DIR / "smoke.yaml") as f:
            _deep_update(cfg, yaml.safe_load(f))
    cfg["profile"] = profile()
    # REGAT_RESULTS_DIR redirects all result files (golden regression reruns); results_dir_main keeps the
    # canonical location for files that must be shared (e.g. the fixed primary normaliser)
    cfg["data"]["paths"]["results_dir_main"] = cfg["data"]["paths"]["results_dir"]
    if os.environ.get("REGAT_RESULTS_DIR"):
        cfg["data"]["paths"]["results_dir"] = os.environ["REGAT_RESULTS_DIR"]
    return cfg


def path(cfg, key, *parts):
    p = ROOT / cfg["data"]["paths"][key]
    p = p.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------- experiment keys ----------
# "A" / "B": MS MARCO protocols (B only owns GAT-dependent files, everything else is shared with A).
# "<dataset>": a multi-hop dataset (cache/<dataset>/), "<dataset>@<variant>": same data, another GAT/graph
# (H4 graph types, zero-shot MS MARCO GAT); only GAT-dependent files live in cache/<dataset>/graphs/<variant>/.
MSMARCO_KEYS = ("A", "B")
GAT_DEPENDENT = ("ctx_", "gat_best", "feat_stats", "cfeat_stats", "stage4", "gat_train_summary")


def split_key(key):
    """'musique@random' -> ('musique', 'random'); 'A' -> ('A', None); 'B' -> ('A', 'B')."""
    if key == "B":
        return "A", "B"
    ds, _, variant = key.partition("@")
    return ds, (variant or None)


def base_key(key):
    return split_key(key)[0]


def is_msmarco(key):
    return base_key(key) == "A"


def cache_path(cfg, name, protocol="A"):
    """Path of a cache file for an experiment key (see above)."""
    ds, variant = split_key(protocol)
    if ds == "A":
        return path(cfg, "cache_dir", "B", name) if variant == "B" else path(cfg, "cache_dir", name)
    if variant is not None:
        return path(cfg, "cache_dir", ds, "graphs", variant, name)
    return path(cfg, "cache_dir", ds, name)


def data_cache_path(cfg, name, key):
    """Files shared by all variants of a dataset (cands, sem, bi, bm25, embeddings, splits, ...)."""
    return cache_path(cfg, name, base_key(key))


def test_sets(cfg, key):
    return cfg["fusion"]["eval"]["test_sets"] if is_msmarco(key) else ["test"]


def dataset_cfg(cfg, key):
    ds = base_key(key)
    return None if ds == "A" else {**cfg["multihop"]["datasets"][ds], "name": ds}


def results_path(cfg, protocol, *parts):
    return path(cfg, "results_dir", protocol, *parts)


def set_env(cfg):
    os.environ.setdefault("IR_DATASETS_HOME", str(ROOT / cfg["data"]["paths"]["ir_datasets_home"]))
    os.environ.setdefault("PYSERINI_CACHE", str(ROOT / cfg["data"]["paths"]["pyserini_cache"]))
    # pyserini>=1.x builds an OpenAI client at import time and the openai SDK rejects an empty key;
    # BM25 never calls OpenAI, so a placeholder only lets the import succeed.
    os.environ.setdefault("OPENAI_API_KEY", "unused-placeholder")


def get_logger(name):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return logging.getLogger(name)


def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, p):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=1)


def load_json(p):
    with open(p) as f:
        return json.load(f)


# ---------- packed score / embedding caches ----------
# Score caches: {'qids': [qid,...], 'scores': FloatTensor[N, K]}, rows aligned with cands_{set}.json.
# Embedding caches: {'ids': [id,...], 'emb': Tensor[N, d]} (fp16 on disk, cast to fp32 on load).

def save_scores(qids, scores, p):
    torch.save({"qids": list(qids), "scores": scores.float().cpu()}, p)


def load_scores(p):
    return torch.load(p, weights_only=False)


def save_emb(ids, emb, p, fp16=True):
    torch.save({"ids": list(ids), "emb": emb.half().cpu() if fp16 else emb.float().cpu()}, p)


def load_emb(p, cast=True):
    d = torch.load(p, weights_only=False)
    if cast:
        d["emb"] = d["emb"].float()
    d["index"] = {k: i for i, k in enumerate(d["ids"])}
    return d


def deepcopy(x):
    return copy.deepcopy(x)
