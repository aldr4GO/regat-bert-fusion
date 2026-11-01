"""HotpotQA distractor (closed pool of 10 paragraphs, 2 gold). Gold = title listed in supporting_facts.
Labels: type (bridge / comparison), level (easy / medium / hard)."""
import pyarrow.parquet as pq

from src.datasets.closed import build_closed
from src.datasets.sources import hf_files


def rows(dcfg, split, columns=None):
    for p in hf_files(dcfg, split):
        yield from pq.read_table(p, columns=columns).to_pylist()


def _records(dcfg, split, joiner=""):
    out = []
    for r in rows(dcfg, split):
        gold = set(r["supporting_facts"]["title"])
        ctx = r["context"]
        paras = [(t, joiner.join(sents).strip(), t in gold) for t, sents in zip(ctx["title"], ctx["sentences"])]
        meta = {"type": r.get("type"), "level": r.get("level")}
        out.append({"qid": r["id"], "question": r["question"], "paragraphs": paras, "meta": meta})
    return out


def build(cfg, dcfg):
    return build_closed(cfg, dcfg, _records(dcfg, "train"), _records(dcfg, "test"))


def labels_by_id(cfg):
    """{qid: {type, level}} for every HotpotQA train + dev question (used to label BEIR HotpotQA queries)."""
    dcfg = {**cfg["multihop"]["datasets"]["hotpotqa"], "name": "hotpotqa"}
    out = {}
    for split in ("train", "test"):
        for r in rows(dcfg, split, columns=["id", "type", "level"]):
            out[r["id"]] = {"type": r["type"], "level": r["level"]}
    return out
