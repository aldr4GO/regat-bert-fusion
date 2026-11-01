"""MuSiQue-Ans (closed pool of 20 paragraphs, 2-4 gold). Hop count from the id prefix ('2hop__', '3hop1__', ...)
and from len(question_decomposition); both are stored (meta hops / hops_decomp)."""
import json
import re

from src.datasets.closed import build_closed
from src.datasets.common import log
from src.datasets.sources import hf_files


def _records(dcfg, split, limit=None):
    out, mismatch = [], 0
    for p in hf_files(dcfg, split):
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                hops = int(re.match(r"(\d)hop", r["id"]).group(1))
                dec = len(r["question_decomposition"])
                mismatch += hops != dec
                paras = [(x["title"], x["paragraph_text"], bool(x["is_supporting"])) for x in r["paragraphs"]]
                out.append({"qid": r["id"], "question": r["question"], "paragraphs": paras,
                            "meta": {"hops": hops, "hops_decomp": dec, "type": re.match(r"\dhop\d?", r["id"]).group(0)}})
    log.info("musique %s: %d questions, hop prefix != decomposition length for %d", split, len(out), mismatch)
    return out


def build(cfg, dcfg):
    return build_closed(cfg, dcfg, _records(dcfg, "train"), _records(dcfg, "test"))
