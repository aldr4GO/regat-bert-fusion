"""Rule-based query properties (plan Step 0, item 5): n_words, wh-word, numeric-answer indicator, number of
named entities (spaCy en_core_web_sm). Used for analysis buckets and the optional QSC gate (E5) only.

Usage:  python -m src.query_meta --protocol A|<dataset>
Writes: query_meta.json in the dataset's cache dir ({qid: {n_words, wh, numeric, n_entities}})
"""
import argparse
import re

from src.common import base_key, data_cache_path, get_logger, is_msmarco, load_config, load_json, save_json

log = get_logger("query_meta")
WH = ("what", "who", "when", "where", "how", "why")
NUMERIC = re.compile(r"\bhow (many|much|long|old)\b|\byears?\b", re.I)


def wh_type(text):
    for tok in re.findall(r"[a-z]+", text.lower()):
        if tok in WH:
            return tok
    return "other"


def _queries(cfg, key):
    if is_msmarco(key):
        from src.data import load_queries
        return load_queries(cfg)
    return load_json(data_cache_path(cfg, "queries.json", key))


def compute(cfg, key):
    import spacy
    nlp = spacy.load(cfg["multihop"]["spacy_model"], disable=["parser", "lemmatizer"])
    queries = _queries(cfg, key)
    qids = list(queries)
    out = {}
    for q, doc in zip(qids, nlp.pipe((queries[q] for q in qids), batch_size=512)):
        t = queries[q]
        out[q] = {"n_words": len(t.split()), "wh": wh_type(t), "numeric": int(bool(NUMERIC.search(t))),
                  "n_entities": len(doc.ents)}
    return out


def query_meta(cfg, key, _memo={}):
    k = (cfg["profile"], base_key(key))
    if k not in _memo:
        p = data_cache_path(cfg, "query_meta.json", key)
        if not p.exists():
            log.info("computing query properties for %s", base_key(key))
            save_json(compute(cfg, key), p)
        _memo[k] = load_json(p)
    return _memo[k]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="A")
    args = ap.parse_args()
    cfg = load_config()
    m = query_meta(cfg, args.protocol)
    log.info("%d queries; example %s", len(m), next(iter(m.items())))


if __name__ == "__main__":
    main()
