"""Builder shared by the closed-pool adapters (MuSiQue, HotpotQA distractor, 2WikiMultiHopQA)."""
from src.datasets.common import log, make_splits, order_by_score, pool_bm25


def build_closed(cfg, dcfg, train_records, test_records):
    """records: dicts {qid, question, paragraphs: [(title, text, is_gold)], meta}. Returns the adapter object."""
    recs = {r["qid"]: r for r in list(train_records) + list(test_records)}
    train_ids = [r["qid"] for r in train_records]
    test_ids = [r["qid"] for r in test_records]
    has_pos = {q: any(g for _, _, g in recs[q]["paragraphs"]) for q in recs}
    splits = make_splits(train_ids, test_ids, dcfg["sizes"], has_pos, cfg["data"]["seed"])
    used = [q for s in splits.values() for q in s]
    queries, pools, passages, labels, meta = {}, {}, {}, {}, {}
    for q in used:
        r = recs[q]
        queries[q] = r["question"]
        pids = [f"{q}::{i}" for i in range(len(r["paragraphs"]))]
        pools[q] = pids
        labels[q] = {}
        for pid, (title, text, gold) in zip(pids, r["paragraphs"]):
            passages[pid] = {"title": title, "text": text}
            if gold:
                labels[q][pid] = 1
        meta[q] = {"hops": None, "type": None, "level": None, **r["meta"], "n_words": len(r["question"].split())}
    log.info("%s: %d train / %d test records, %d used queries, %d paragraphs; computing in-pool BM25",
             dcfg["name"], len(train_ids), len(test_ids), len(used), len(passages))
    scores = pool_bm25(queries, pools, passages)
    cands, cscores = order_by_score(pools, scores)
    return {"queries": queries, "cands": cands, "passages": passages, "labels": labels, "meta": meta,
            "splits": splits, "scores": cscores}
