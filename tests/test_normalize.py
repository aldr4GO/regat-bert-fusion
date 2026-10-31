import pytest
import torch

from src.normalize import NORMALIZERS, average_rank_desc, minmax, rank, zscore


@pytest.fixture
def x():
    return torch.randn(50, 100, generator=torch.Generator().manual_seed(0)) * 3 + 1


def test_minmax_range(x):
    y = minmax(x)
    assert y.min() >= 0 and y.max() <= 1


def test_zscore_moments(x):
    y = zscore(x)
    assert torch.allclose(y.mean(-1), torch.zeros(50), atol=1e-5)
    assert torch.allclose(y.std(-1, unbiased=False), torch.ones(50), atol=1e-4)


def test_rank_monotone_invariant(x):
    assert torch.allclose(rank(x), rank(torch.exp(x / 3) * 7 - 2))


def test_rank_range_and_ties():
    v = torch.tensor([[3.0, 1.0, 3.0, 2.0]])
    assert torch.equal(average_rank_desc(v), torch.tensor([[1.5, 4.0, 1.5, 3.0]]))
    r = rank(torch.tensor([[4.0, 3.0, 2.0, 1.0]]))
    assert torch.allclose(r, torch.tensor([[1.0, 2 / 3, 1 / 3, 0.0]]))


@pytest.mark.parametrize("name", list(NORMALIZERS))
def test_preserves_order(name, x):
    y = NORMALIZERS[name](x)
    assert torch.equal(torch.argsort(x, dim=-1, descending=True, stable=True),
                       torch.argsort(y, dim=-1, descending=True, stable=True))


@pytest.mark.parametrize("name", list(NORMALIZERS))
def test_constant_input_finite(name):
    assert torch.isfinite(NORMALIZERS[name](torch.ones(3, 100))).all()
