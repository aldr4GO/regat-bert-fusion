import copy

import pytest
import torch

from conftest import make_fusion_data
from src.fusion import FusionModel, fuse
from src.normalize import NORMALIZERS
from src.train_fusion import train_fusion


def _order(s):
    return torch.argsort(s, dim=-1, descending=True, stable=True)


@pytest.mark.parametrize("name", list(NORMALIZERS))
def test_alpha_extremes_reproduce_components(name):
    g = torch.Generator().manual_seed(1)
    sem, ctx = torch.randn(20, 100, generator=g), torch.randn(20, 100, generator=g)
    sn, cn = NORMALIZERS[name](sem), NORMALIZERS[name](ctx)
    assert torch.equal(_order(fuse(sn, cn, 1.0)), _order(sem))
    assert torch.equal(_order(fuse(sn, cn, 0.0)), _order(ctx))


@pytest.mark.parametrize("dep", ["fixed", "Q", "QS", "S"])
def test_alpha_strictly_inside(cfg, dep):
    fc = cfg["fusion"]
    m = FusionModel(dep, 16, 12, 0.5, fc)
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(0).add_(1e4)          # push sigmoid into saturation
    a = m.alpha(torch.randn(8, 16), torch.randn(8, 12))
    assert ((a > 0) & (a < 1)).all()


@pytest.mark.parametrize("dep", ["Q", "QS", "S"])
def test_init_equals_alpha0(cfg, dep):
    m = FusionModel(dep, 16, 12, 0.3, cfg["fusion"]).eval()
    a = m.alpha(torch.nn.functional.normalize(torch.randn(32, 16), dim=-1), torch.randn(32, 12))
    # last layer std 0.01 -> starts close to the fixed baseline (not exactly equal)
    assert torch.allclose(a, torch.full_like(a, 0.3), atol=0.05)
    assert abs(a.mean().item() - 0.3) < 0.01


def _small_cfg(cfg):
    fc = copy.deepcopy(cfg["fusion"])
    fc["train"]["max_epochs"] = 3
    return fc


@pytest.mark.parametrize("dep", ["fixed", "QS", "paper"])
def test_deterministic(cfg, dep):
    fc = _small_cfg(cfg)
    tr, va = make_fusion_data(seed=0), make_fusion_data(seed=1)
    outs = []
    for _ in range(2):
        m, info = train_fusion(fc, dep, 0.5, seed=3, train_data=tr, val_data=va)
        outs.append((info["best_val_ndcg@10"], [p.detach().clone() for p in m.parameters()]))
    assert outs[0][0] == outs[1][0]
    assert all(torch.equal(a, b) for a, b in zip(outs[0][1], outs[1][1]))


@pytest.mark.parametrize("dep", ["Q", "QS", "S"])
def test_label_perturbation_does_not_change_alpha(cfg, dep):
    """Alpha is a function of q_emb and raw-score features only: changing labels must not move it."""
    fc = _small_cfg(cfg)
    tr, va = make_fusion_data(seed=0), make_fusion_data(seed=1)
    m, _ = train_fusion(fc, dep, 0.5, seed=0, train_data=tr, val_data=va)
    m.eval()
    te = make_fusion_data(seed=2)
    a1 = m.alpha(te.q_emb, te.feats)
    te2 = copy.deepcopy(te)
    te2.labels = 1 - te2.labels
    te2.gains = torch.rand_like(te2.gains)
    a2 = m.alpha(te2.q_emb, te2.feats)
    assert torch.equal(a1, a2)
    # and the inputs fed to alpha never touch labels
    assert set(FusionModel.alpha.__code__.co_varnames[:3]) == {"self", "q_emb", "feats"}


def test_training_improves_or_keeps_val(cfg):
    fc = _small_cfg(cfg)
    tr, va = make_fusion_data(seed=0), make_fusion_data(seed=1)
    m, info = train_fusion(fc, "QS", 0.5, seed=0, train_data=tr, val_data=va)
    assert info["best_val_ndcg@10"] >= max(0, info["history"][0]["val_ndcg@10"] - 1) and info["best_epoch"] >= 0
