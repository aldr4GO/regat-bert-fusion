"""Shared pieces of the dataset adapters: the common output object, in-pool BM25, splits, writing caches.

Adapter output (plan Sec. 3.1):
  queries  {qid: text}
  cands    {qid: [pid, ...]}                    variable length; ordered by BM25 (in-pool BM25 for closed pools)
  passages {pid: {"title": str, "text": str}}
  labels   {qid: {pid: grade}}                  binary here
  meta     {qid: {"hops", "type", "level", "n_words"}}
  splits   {"gat_train", "fusion_train", "val", "test"}
  scores   {qid: [bm25 score per candidate]}
Closed-pool pids are unique per question: f"{qid}::{i}" (the same paragraph appears under many questions).
"""
import math
import re
from collections import Counter

import numpy as np
import torch

from src.common import data_cache_path, get_logger, save_json

log = get_logger("datasets")
_TOKEN = re.compile(r"\w+", re.UNICODE)


class Tokenizer:
    """Lower-case word tokens, English stop words removed, Snowball-stemmed (PyStemmer)."""

    def __init__(self):
        import Stemmer
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        self.stem = Stemmer.Stemmer("english").stemWords
        self.stop = ENGLISH_STOP_WORDS

    def __call__(self, text):
        return self.stem([t for t in _TOKEN.findall(text.lower()) if t not in self.stop])


def pool_bm25(queries, pools, passages, k1=0.9, b=0.4):
    """BM25 of each query against the paragraphs of its own pool, with corpus statistics (IDF, avgdl) over
    every paragraph of the dataset. Returns {qid: [score per pool entry]} (pool order)."""
    tok = Tokenizer()
    ptoks = {pid: tok(f"{p['title']} {p['text']}") for pid, p in passages.items()}
    df = Counter()
    for t in ptoks.values():
        df.update(set(t))
    N = len(ptoks)
    avgdl = sum(len(t) for t in ptoks.values()) / max(N, 1)
    idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
    out = {}
    for q, pids in pools.items():
        qt = tok(queries[q])
        sc = []
        for pid in pids:
            tf = Counter(ptoks[pid])
            dl = len(ptoks[pid])
            sc.append(sum(idf.get(w, 0.0) * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * dl / avgdl))
                          for w in qt if w in tf))
        out[q] = sc
    return out


def order_by_score(pools, scores):
    """Stable descending sort of every pool by its score (ties keep the original pool order)."""
    cands, cscores = {}, {}
    for q, pids in pools.items():
        idx = sorted(range(len(pids)), key=lambda i: -scores[q][i])
        cands[q] = [pids[i] for i in idx]
        cscores[q] = [scores[q][i] for i in idx]
    return cands, cscores


def make_splits(train_ids, test_ids, sizes, has_pos, seed):
    """Shuffle official-train ids (seed); gat_train / fusion_train only keep queries with a gold paragraph in
    the pool; val (sizes['val'] == -1 -> the rest of train) and test keep every query."""
    rng = np.random.RandomState(seed)
    ids = rng.permutation(sorted(train_ids)).tolist()
    pos = [q for q in ids if has_pos[q]]
    n_g, n_f = sizes["gat_train"], sizes["fusion_train"]
    assert len(pos) >= n_g + n_f, f"not enough train queries with a positive: {len(pos)} < {n_g + n_f}"
    gat, fus = pos[:n_g], pos[n_g:n_g + n_f]
    used = set(gat) | set(fus)
    rest = [q for q in ids if q not in used]
    val = rest if sizes["val"] == -1 else rest[:sizes["val"]]
    return {"gat_train": gat, "fusion_train": fus, "val": val, "test": sorted(test_ids)}


def norm_title(t):
    """Lower-case, strip parenthetical suffixes such as '(film)' (title_link edges, plan Sec. 3.3)."""
    t = re.sub(r"\s*\([^)]*\)\s*$", "", t.strip().lower())
    return re.sub(r"\s+", " ", t)


def pad_matrix(rows, K=None, fill=0.0):
    K = K or max(len(r) for r in rows)
    x = torch.full((len(rows), K), fill)
    m = torch.zeros(len(rows), K, dtype=torch.bool)
    for i, r in enumerate(rows):
        x[i, :len(r)] = torch.tensor(r, dtype=torch.float)
        m[i, :len(r)] = True
    return x, m


def save_padded(cfg, key, name, qids, rows):
    x, m = pad_matrix(rows)
    torch.save({"qids": list(qids), "scores": x, "mask": m}, data_cache_path(cfg, name, key))


def write_dataset(cfg, ds, obj):
    """Writes queries/qrels/meta/splits/passages and per-split cands_ + bm25_ caches into cache/<ds>/."""
    f = lambda n: data_cache_path(cfg, n, ds)
    splits = obj["splits"]
    keep = {q for s in splits.values() for q in s}
    save_json({q: obj["queries"][q] for q in keep}, f("queries.json"))
    save_json({q: obj["labels"].get(q, {}) for q in keep}, f("qrels.json"))
    save_json({q: obj["meta"][q] for q in keep}, f("meta.json"))
    save_json(splits, f("splits.json"))
    if obj.get("passages") is not None:
        need = {p for q in keep for p in obj["cands"][q]}
        save_json({p: obj["passages"][p] for p in need}, f("passages.json"))
    stats = {"split_sizes": {k: len(v) for k, v in splits.items()}}
    for s, qs in splits.items():
        save_json({q: obj["cands"][q] for q in qs}, f(f"cands_{s}.json"))
        save_padded(cfg, ds, f"bm25_{s}.pt", qs, [obj["scores"][q] for q in qs])
        lens = [len(obj["cands"][q]) for q in qs]
        gold = [sum(obj["labels"].get(q, {}).get(p, 0) > 0 for p in obj["cands"][q]) for q in qs]
        stats[s] = {"n": len(qs), "mean_K": float(np.mean(lens)), "min_K": int(min(lens)), "max_K": int(max(lens)),
                    "mean_gold_in_pool": float(np.mean(gold)),
                    "frac_with_gold": float(np.mean([g > 0 for g in gold]))}
    save_json(stats, f("prep_stats.json"))
    log.info("%s written: %s", ds, stats)
    return stats
