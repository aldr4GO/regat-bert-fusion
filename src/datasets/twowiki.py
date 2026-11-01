"""2WikiMultiHopQA (optional, closed pool of ~10 paragraphs). Labels: type (comparison / bridge_comparison /
compositional / inference). Official dev is the test split (the HF test split has no supporting facts)."""
from src.datasets.closed import build_closed
from src.datasets.hotpotqa_distractor import _records


def build(cfg, dcfg):
    return build_closed(cfg, dcfg, _records(dcfg, "train", " "), _records(dcfg, "test", " "))
