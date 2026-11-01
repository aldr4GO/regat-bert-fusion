"""Variable candidate counts (multi-hop plan Sec. 3.2 / 3.7): padding invariance and small-K edge cases."""
import pytest
import torch

from src.features import score_features
from src.metrics import all_recall_at_k, ideal_gains, ndcg_at_k, rr_at_k
from src.normalize import NORMALIZERS
from src.train_fusion import listwise_loss

N, K, PAD = 6, 10, 7


@pytest.fixture
def data():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N, K, generator=g) * 2
    y = torch.randn(N, K, generator=g)
    lab = torch.zeros(N, K)
    lab[torch.arange(N), torch.randint(0, K, (N,), generator=g)] = 1
    return x, y, lab


def pad(t, value=0.0):
    return torch.cat([t, torch.full((t.shape[0], PAD), value)], -1)


def pmask():
    return torch.cat([torch.ones(N, K, dtype=torch.bool), torch.zeros(N, PAD, dtype=torch.bool)], -1)


@pytest.mark.parametrize("name", list(NORMALIZERS))
def test_normalizer_padding_invariance(name, data):
    x, _, _ = data
    ref = NORMALIZERS[name](x)
    out = NORMALIZERS[name](pad(x, 123.0), mask=pmask())
    assert torch.allclose(out[:, :K], ref, atol=1e-5)
    assert (out[:, K:] == 0).all()


@pytest.mark.parametrize("m", [3, 10])
def test_features_padding_invariance(data, m):
    x, y, _ = data
    ref = score_features(x, y, top_k=m)
    out = score_features(pad(x, 50.0), pad(y, -50.0), top_k=m, mask=pmask())
    assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max(0)


def test_loss_and_metrics_padding_invariance(data):
    x, _, lab = data
    m = pmask()
    xp, lp = pad(x, 99.0), pad(lab)
    assert torch.allclose(listwise_loss(xp, lp, m), listwise_loss(x, lab), atol=1e-6)
    ideal = ideal_gains([{"r": 1}] * N, 10)
    assert torch.allclose(ndcg_at_k(xp, lp, ideal, mask=m), ndcg_at_k(x, lab, ideal))
    assert torch.allclose(rr_at_k(xp, lp, mask=m), rr_at_k(x, lab))
    n_rel = lab.sum(-1)
    assert torch.equal(all_recall_at_k(xp, lp, n_rel, 5, m), all_recall_at_k(x, lab, n_rel, 5))


@pytest.mark.parametrize("k_valid", [1, 2, 3])
def test_small_k_finite(k_valid):
    x = torch.randn(4, 5)
    m = torch.zeros(4, 5, dtype=torch.bool)
    m[:, :k_valid] = True
    for name in NORMALIZERS:
        assert torch.isfinite(NORMALIZERS[name](x, mask=m)).all()
    F = score_features(x, x.flip(-1), top_k=3, mask=m)
    assert F.shape == (4, 12) and torch.isfinite(F).all()
    if k_valid == 1:   # a single candidate: rank-normalised score 1, overlap 1, gap 0
        assert torch.allclose(NORMALIZERS["rank"](x, mask=m)[:, 0], torch.ones(4))
        assert torch.allclose(F[:, 3], torch.zeros(4))


def test_all_equal_and_no_relevant():
    x = torch.full((3, 4), 2.0)
    m = torch.ones(3, 4, dtype=torch.bool)
    m[0, 2:] = False
    assert torch.isfinite(score_features(x, x, top_k=3, mask=m)).all()
    for name in NORMALIZERS:
        assert torch.isfinite(NORMALIZERS[name](x, mask=m)).all()
    lab = torch.zeros(3, 4)
    ideal = ideal_gains([{}] * 3, 10)
    assert (ndcg_at_k(x, lab, ideal, mask=m) == 0).all()
    assert (rr_at_k(x, lab, mask=m) == 0).all()
    assert (all_recall_at_k(x, lab, lab.sum(-1), 2, m) == 0).all()


def test_fewer_than_k_candidates_uses_all():
    s = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    m = torch.tensor([[True, True, True, False]])
    g = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    assert abs(rr_at_k(s, g, 10, m).item() - 1 / 3) < 1e-7
    assert all_recall_at_k(s, g, torch.tensor([1.0]), 10, m).item() == 1.0
