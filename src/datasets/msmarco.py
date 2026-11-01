"""MS MARCO adapter view of the existing protocol A caches (built by src/data.py, never rebuilt here)."""
from src.common import load_config


def build(cfg=None, dcfg=None):
    from src.data import load_cands, load_qrels, load_queries, load_splits
    cfg = cfg or load_config()
    splits = {k: v for k, v in load_splits(cfg, "A").items() if k in ("gat_train", "fusion_train", "val", "test")}
    queries, qrels = load_queries(cfg), load_qrels(cfg)
    cands = {q: c for s in splits for q, c in load_cands(cfg, s).items()}
    meta = {q: {"hops": None, "type": None, "level": None, "n_words": len(queries[q].split())} for q in cands}
    return {"queries": {q: queries[q] for q in cands}, "cands": cands, "passages": None,
            "labels": {q: qrels[q] for q in cands}, "meta": meta, "splits": splits, "scores": None}
