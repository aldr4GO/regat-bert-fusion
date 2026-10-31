"""Stage 0: queries, qrels, BM25 candidates, splits. Also the loaders used by every later stage.

Usage:  python -m src.data
Writes: data/queries.json, data/qrels.json, data/splits_A.json, data/splits_B.json,
        cache/cands_{set}.json, cache/bm25_{set}.pt, results/bm25_first_stage.json
"""
import argparse
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from src.common import (cache_path, get_logger, load_config, load_json, path, results_path, save_json,
                        save_scores, set_env)

log = get_logger("data")


# ---------------------------------------------------------------- ir_datasets
def _ird(name):
    import ir_datasets
    return ir_datasets.load(name)


def read_queries(name):
    return {q.query_id: q.text for q in _ird(name).queries_iter()}


def read_qrels(name):
    qrels = defaultdict(dict)
    for r in _ird(name).qrels_iter():
        qrels[r.query_id][r.doc_id] = int(r.relevance)
    return dict(qrels)


def read_passages(cfg, pids):
    """Returns {pid: text} for the requested pids (one pass over the collection)."""
    need = set(pids)
    out = {}
    for d in tqdm(_ird(cfg["data"]["datasets"]["collection"]).docs_iter(), desc="collection",
                  mininterval=30):
        if d.doc_id in need:
            out[d.doc_id] = d.text
    missing = need - out.keys()
    assert not missing, f"{len(missing)} pids not found in collection"
    return out


# ---------------------------------------------------------------- BM25
class BM25:
    def __init__(self, cfg):
        bc = cfg["data"]["bm25"]
        self.cfg, self.bc = cfg, bc
        self.backend = bc["backend"]
        if self.backend == "pyserini":
            try:
                from pyserini.search.lucene import LuceneSearcher
                self.searcher = LuceneSearcher.from_prebuilt_index(bc["pyserini_index"])
                self.searcher.set_bm25(bc["k1"], bc["b"])
                log.info("BM25 backend: pyserini prebuilt %s (k1=%s b=%s)", bc["pyserini_index"], bc["k1"], bc["b"])
            except Exception as e:  # pragma: no cover - fallback path
                log.warning("pyserini unavailable (%r); falling back to bm25s", e)
                self.backend = "bm25s"
        if self.backend == "bm25s":
            self._build_bm25s()

    def _build_bm25s(self):  # pragma: no cover - only used if pyserini fails
        import bm25s
        import Stemmer
        idx_dir = path(self.cfg, "data_dir", "bm25s_index")
        stemmer = Stemmer.Stemmer("english")
        if (idx_dir / "params.index.json").exists():
            self.retriever = bm25s.BM25.load(str(idx_dir))
            self.doc_ids = load_json(idx_dir / "doc_ids.json")
        else:
            ids, texts = [], []
            for d in _ird(self.cfg["data"]["datasets"]["collection"]).docs_iter():
                ids.append(d.doc_id)
                texts.append(d.text)
            tok = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer)
            self.retriever = bm25s.BM25(k1=self.bc["k1"], b=self.bc["b"])
            self.retriever.index(tok)
            self.retriever.save(str(idx_dir))
            save_json(ids, idx_dir / "doc_ids.json")
            self.doc_ids = ids
        self.stemmer = stemmer
        log.info("BM25 backend: bm25s (k1=%s b=%s)", self.bc["k1"], self.bc["b"])

    def search(self, queries, k):
        """queries: {qid: text} -> {qid: ([pid,...], [score,...])} sorted by score desc."""
        qids = list(queries)
        out = {}
        bs = self.bc["batch_size"]
        for i in tqdm(range(0, len(qids), bs), desc=f"bm25 k={k}", mininterval=30):
            chunk = qids[i:i + bs]
            if self.backend == "pyserini":
                res = self.searcher.batch_search([queries[q] for q in chunk], chunk, k=k,
                                                 threads=self.bc["threads"])
                for q in chunk:
                    out[q] = ([h.docid for h in res[q]], [float(h.score) for h in res[q]])
            else:  # pragma: no cover
                import bm25s
                tok = bm25s.tokenize([queries[q] for q in chunk], stopwords="en", stemmer=self.stemmer)
                docs, scores = self.retriever.retrieve(tok, k=k, n_threads=self.bc["threads"])
                for j, q in enumerate(chunk):
                    out[q] = ([self.doc_ids[d] for d in docs[j]], [float(s) for s in scores[j]])
        return out


def pad_candidates(res, K, rng, n_docs=8841823):
    """Pads runs shorter than K with random passages scored below the last hit (logged)."""
    n_pad = 0
    for q, (pids, scores) in res.items():
        if len(pids) < K:
            n_pad += 1
            low = (min(scores) if scores else 0.0) - 1.0
            have = set(pids)
            while len(pids) < K:
                p = str(int(rng.randint(n_docs)))
                if p not in have:
                    have.add(p)
                    pids.append(p)
                    scores.append(low)
                    low -= 1e-3
    return n_pad


def has_positive(pids, qrel):
    return any(qrel.get(p, 0) > 0 for p in pids)


# ---------------------------------------------------------------- Stage 0 main
def build(cfg):
    dc = cfg["data"]
    K, seed = dc["K"], dc["seed"]
    rng = np.random.RandomState(seed)
    ds = dc["datasets"]

    log.info("loading queries/qrels")
    train_q, train_qrels = read_queries(ds["train"]), read_qrels(ds["train"])
    dev_q, dev_qrels = read_queries(ds["dev"]), read_qrels(ds["dev"])
    trec = {s: (read_queries(ds[s]), read_qrels(ds[s])) for s in ("trec19", "trec20")}
    # only judged TREC queries are evaluated
    trec = {s: ({q: t for q, t in qs.items() if q in qr}, qr) for s, (qs, qr) in trec.items()}

    dev_ids = sorted(q for q in dev_q if q in dev_qrels)
    if "smoke_dev_size" in dc:
        dev_ids = sorted(rng.choice(dev_ids, dc["smoke_dev_size"], replace=False).tolist())
        trec = {s: ({q: qs[q] for q in sorted(qs)[:dc["smoke_trec_size"]]}, qr) for s, (qs, qr) in trec.items()}
    log.info("train(judged)=%d dev/small=%d trec19=%d trec20=%d", len(train_q), len(dev_ids),
             len(trec["trec19"][0]), len(trec["trec20"][0]))

    train_ids = sorted(q for q in train_q if q in train_qrels)
    pool = rng.permutation(train_ids)[:dc["protocol_A"]["train_pool"]].tolist()

    bm25 = BM25(cfg)
    pool_res = bm25.search({q: train_q[q] for q in pool}, K)
    deep = {}
    deep["test"] = bm25.search({q: dev_q[q] for q in dev_ids}, dc["recall_depth"])
    for s in ("trec19", "trec20"):
        deep[s] = bm25.search(trec[s][0], dc["recall_depth"])

    # first-stage recall (before truncation)
    import ir_measures
    from ir_measures import R
    first_stage = {}
    for s, res in deep.items():
        qrels = dev_qrels if s == "test" else trec[s][1]
        rel_level = 1 if s == "test" else dc["trec_binary_threshold"]
        run = {q: {p: sc for p, sc in zip(*res[q])} for q in res}
        qr = {q: qrels[q] for q in res}
        first_stage[s] = {str(m): v for m, v in ir_measures.calc_aggregate(
            [R(rel=rel_level) @ 100, R(rel=rel_level) @ 1000], qr, run).items()}
    log.info("BM25 first-stage recall: %s", first_stage)

    n_pad = pad_candidates(pool_res, K, rng)
    for s in deep:
        deep[s] = {q: (p[:K], sc[:K]) for q, (p, sc) in deep[s].items()}
        n_pad += pad_candidates(deep[s], K, rng)

    # ---- protocol A splits
    sz = dc["protocol_A"]["sizes"]
    val = pool[:sz["val"]]
    rest = [q for q in pool[sz["val"]:] if has_positive(pool_res[q][0], train_qrels[q])]
    n_dropped_pool = len(pool) - sz["val"] - len(rest)
    need = sz["gat_train"] + sz["fusion_train"]
    assert len(rest) >= need, f"train pool too small after filtering: {len(rest)} < {need}; raise train_pool"
    gat_train, fusion_train = rest[:sz["gat_train"]], rest[sz["gat_train"]:need]
    splits_A = {"gat_train": gat_train, "fusion_train": fusion_train, "val": val, "test": dev_ids,
                "trec19": sorted(trec["trec19"][0]), "trec20": sorted(trec["trec20"][0])}

    # ---- protocol B splits (subsets of dev/small)
    fr = dc["protocol_B"]["fractions"]
    perm = np.random.RandomState(seed).permutation(dev_ids).tolist()
    n = len(perm)
    cuts = np.cumsum([int(round(fr[k] * n)) for k in ("gat_train", "fusion_train", "val")])
    raw_B = {"gat_train": perm[:cuts[0]], "fusion_train": perm[cuts[0]:cuts[1]],
             "val": perm[cuts[1]:cuts[2]], "test": perm[cuts[2]:]}
    dropped_B = {}
    splits_B = {}
    for s, qs in raw_B.items():
        if s in ("gat_train", "fusion_train"):
            keep = [q for q in qs if has_positive(deep["test"][q][0], dev_qrels[q])]
            dropped_B[s] = [q for q in qs if q not in set(keep)]
            splits_B[s] = keep
        else:
            splits_B[s] = qs
    splits_B["trec19"], splits_B["trec20"] = splits_A["trec19"], splits_A["trec20"]
    splits_B["_dropped"] = dropped_B

    # ---- write
    queries, qrels = {}, {}
    sets = {"gat_train": gat_train, "fusion_train": fusion_train, "val": val}
    for s, qs in sets.items():
        for q in qs:
            queries[q], qrels[q] = train_q[q], train_qrels[q]
    for q in dev_ids:
        queries[q], qrels[q] = dev_q[q], dev_qrels[q]
    for s in ("trec19", "trec20"):
        for q in trec[s][0]:
            queries[q], qrels[q] = trec[s][0][q], trec[s][1][q]
    save_json(queries, path(cfg, "data_dir", "queries.json"))
    save_json(qrels, path(cfg, "data_dir", "qrels.json"))
    save_json(splits_A, path(cfg, "data_dir", "splits_A.json"))
    save_json(splits_B, path(cfg, "data_dir", "splits_B.json"))

    all_res = {**{s: {q: pool_res[q] for q in qs} for s, qs in sets.items()}, **deep}
    stats = {"first_stage_bm25": first_stage, "n_padded_runs": n_pad,
             "n_dropped_train_no_positive": n_dropped_pool,
             "n_dropped_B": {k: len(v) for k, v in dropped_B.items()},
             "split_sizes_A": {k: len(v) for k, v in splits_A.items()},
             "split_sizes_B": {k: len(v) for k, v in splits_B.items() if not k.startswith("_")}}
    for s, res in all_res.items():
        qids = splits_A[s]
        save_json({q: res[q][0] for q in qids}, cache_path(cfg, f"cands_{s}.json"))
        save_scores(qids, torch.tensor([res[q][1] for q in qids]), cache_path(cfg, f"bm25_{s}.pt"))
        qr = {q: qrels[q] for q in qids}
        lvl = dc["trec_binary_threshold"] if s.startswith("trec") else 1
        stats[f"cand_recall@{K}_{s}"] = float(np.mean([
            sum(qr[q].get(p, 0) >= lvl for p in res[q][0]) / max(1, sum(v >= lvl for v in qr[q].values()))
            for q in qids]))
        stats[f"frac_with_positive_{s}"] = float(np.mean([any(qr[q].get(p, 0) >= lvl for p in res[q][0])
                                                          for q in qids]))
    save_json(stats, path(cfg, "results_dir", "bm25_first_stage.json"))
    log.info("Stage 0 done: %s", stats)
    return stats


# ---------------------------------------------------------------- loaders
def load_queries(cfg):
    return load_json(path(cfg, "data_dir", "queries.json"))


def load_qrels(cfg):
    return load_json(path(cfg, "data_dir", "qrels.json"))


def load_splits(cfg, protocol):
    return load_json(path(cfg, "data_dir", f"splits_{protocol}.json"))


def load_cands(cfg, set_name):
    return load_json(cache_path(cfg, f"cands_{set_name}.json"))


def physical_set(protocol, split):
    """Which cached query set a protocol split lives in."""
    if protocol == "B" and split in ("gat_train", "fusion_train", "val", "test"):
        return "test"
    return split


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    cfg = load_config()
    set_env(cfg)
    build(cfg)


if __name__ == "__main__":
    main()
