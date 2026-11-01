"""Dataset adapters (multi-hop plan Sec. 3.1). Each `build(cfg, dcfg)` returns the common object described in
src/datasets/common.py; src/prep_dataset.py writes it to cache/<dataset>/."""
from importlib import import_module

ADAPTERS = {"musique": "musique", "hotpotqa": "hotpotqa_distractor", "beir_hotpotqa": "beir_hotpotqa",
            "twowiki": "twowiki", "msmarco": "msmarco"}


def adapter(name):
    return import_module(f"src.datasets.{ADAPTERS[name]}")
