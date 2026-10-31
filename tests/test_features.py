import torch

from src.features import FEATURE_NAMES, score_features, spearman


def test_shape_and_finite():
    F = score_features(torch.randn(10, 100), torch.randn(10, 100))
    assert F.shape == (10, 12) and len(FEATURE_NAMES) == 12
    assert torch.isfinite(F).all()


def test_identical_scores_no_nan():
    F = score_features(torch.full((2, 100), 3.0), torch.zeros(2, 100))
    assert torch.isfinite(F).all()


def test_spearman():
    a = torch.randn(4, 100)
    assert torch.allclose(spearman(a, a), torch.ones(4), atol=1e-5)
    assert torch.allclose(spearman(a, -a), -torch.ones(4), atol=1e-5)
    assert torch.allclose(spearman(a, a ** 3), torch.ones(4), atol=1e-5)
