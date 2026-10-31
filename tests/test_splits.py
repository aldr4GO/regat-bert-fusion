"""Checks the split files written by Stage 0 (skipped until Stage 0 has run for the active profile)."""
import pytest

from src.common import load_json, path


def _splits(cfg, P):
    p = path(cfg, "data_dir", f"splits_{P}.json")
    if not p.exists():
        pytest.skip(f"{p} not built yet")
    return load_json(p)


@pytest.mark.parametrize("P", ["A", "B"])
def test_disjoint(cfg, P):
    s = _splits(cfg, P)
    names = ["gat_train", "fusion_train", "val", "test"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not set(s[a]) & set(s[b]), f"{a} and {b} overlap"


def test_B_covers_dev(cfg):
    a, b = _splits(cfg, "A"), _splits(cfg, "B")
    covered = set().union(*(b[k] for k in ("gat_train", "fusion_train", "val", "test")),
                          *b["_dropped"].values())
    assert covered == set(a["test"])


def test_A_sizes(cfg):
    s = _splits(cfg, "A")
    for k, n in cfg["data"]["protocol_A"]["sizes"].items():
        assert len(s[k]) == n


def test_cache_alignment(cfg):
    """sem_/ctx_/cands_ files share the same query order and length."""
    import torch
    from src.common import cache_path, load_scores
    from src.data import load_cands
    for s in cfg["data"]["cache_sets"]:
        files = [cache_path(cfg, f"{k}_{s}.pt") for k in ("sem", "ctx", "bi", "bm25")]
        if not all(f.exists() for f in files):
            pytest.skip("caches not built yet")
        c = load_cands(cfg, s)
        for f in files:
            d = load_scores(f)
            assert d["qids"] == list(c)
            assert d["scores"].shape == (len(c), len(next(iter(c.values()))))
            assert torch.isfinite(d["scores"]).all()
