"""BEIR HotpotQA (open retrieval): Pyserini BM25 top-100 over the 5.2M-abstract corpus for train and test queries
(same BM25 for both). type / level joined from HotpotQA distractor by question id."""
import numpy as np

from src.datasets.common import log
from src.datasets.hotpotqa_distractor import labels_by_id


def _ird(name):
    import ir_datasets
    return ir_datasets.load(name)


def _queries_qrels(name):
    d = _ird(name)
    q = {x.query_id: x.text for x in d.queries_iter()}
    qr = {}
    for r in d.qrels_iter():
        if r.relevance > 0:
            qr.setdefault(r.query_id, {})[r.doc_id] = int(r.relevance)
    return q, qr


def build(cfg, dcfg):
    from src.data import BM25
    seed, K = cfg["data"]["seed"], dcfg["K"]
    tq, tqr = _queries_qrels(dcfg["ird"]["train"])
    eq, eqr = _queries_qrels(dcfg["ird"]["test"])
    rng = np.random.RandomState(seed)
    sz = dcfg["sizes"]
    pool = rng.permutation(sorted(q for q in tq if q in tqr))[:dcfg["train_pool"]].tolist()
    test_ids = sorted(q for q in eq if q in eqr)
    if "smoke_test_size" in dcfg:
        test_ids = sorted(rng.choice(test_ids, dcfg["smoke_test_size"], replace=False).tolist())
    bcfg = {**cfg, "data": {**cfg["data"], "bm25": {**cfg["data"]["bm25"], **dcfg["bm25"],
                                                    "pyserini_index": dcfg["pyserini_index"]}}}
    bm25 = BM25(bcfg)
    res = bm25.search({q: tq[q] for q in pool}, K)
    res.update(bm25.search({q: eq[q] for q in test_ids}, K))
    short = sum(len(v[0]) < K for v in res.values())   # kept variable length (masked), never padded
    has_pos = {q: any(p in tqr[q] for p in res[q][0]) for q in pool}
    val = pool[:sz["val"]]
    rest = [q for q in pool[sz["val"]:] if has_pos[q]]
    need = sz["gat_train"] + sz["fusion_train"]
    assert len(rest) >= need, f"BEIR train pool too small after filtering: {len(rest)} < {need}"
    splits = {"gat_train": rest[:sz["gat_train"]], "fusion_train": rest[sz["gat_train"]:need], "val": val,
              "test": test_ids}
    lab = labels_by_id(cfg)
    used = [q for s in splits.values() for q in s]
    queries = {q: (tq.get(q) or eq.get(q)) for q in used}
    labels = {q: (tqr.get(q) or eqr.get(q)) for q in used}
    meta = {q: {"hops": None, "type": lab.get(q, {}).get("type"), "level": lab.get(q, {}).get("level"),
                "n_words": len(queries[q].split())} for q in used}
    n_joined = sum(meta[q]["type"] is not None for q in used)
    log.info("beir_hotpotqa: %d/%d queries joined with HotpotQA type/level; %d runs shorter than K", n_joined,
             len(used), short)
    # first-stage ceiling on test (plan Sec. 3.5): Recall@100 and AllRecall@100
    rec, allr = [], []
    for q in test_ids:
        got = set(res[q][0])
        rel = set(labels[q])
        rec.append(len(got & rel) / len(rel))
        allr.append(float(rel <= got))
    first_stage = {"R@100": float(np.mean(rec)), "AllR@100": float(np.mean(allr)), "n_short_runs": short,
                   "n_joined_labels": n_joined}
    return {"queries": queries, "cands": {q: res[q][0] for q in used}, "passages": None, "labels": labels,
            "meta": meta, "splits": splits, "scores": {q: res[q][1] for q in used}, "first_stage": first_stage}
