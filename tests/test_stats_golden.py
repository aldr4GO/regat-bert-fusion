import numpy as np
import pytest

from src.common import load_json, results_path
from src.stats import bootstrap, diff_of_means_ci, holm, spearman


def test_holm():
    adj = holm([0.01, 0.04, 0.03, np.nan])
    assert np.allclose(adj[:3], [0.03, 0.06, 0.06]) and np.isnan(adj[3])


def test_spearman_and_bootstrap():
    rng = np.random.default_rng(0)
    x = rng.random(300)
    assert spearman(x, x ** 3) == pytest.approx(1.0)
    v, lo, hi, p = bootstrap(spearman, [x, -x + rng.normal(0, 0.1, 300)], B=500)
    assert hi < 0 and p <= 0.01
    v, lo, hi, p = diff_of_means_ci(x + 1, x, B=500)
    assert lo > 0.9


@pytest.mark.parametrize("P", ["A", "B"])
def test_golden_regression_report(cfg, P):
    """Acceptance gate of the refactor: written by src/golden_regression.py (pipeline stage golden_A/B)."""
    p = results_path(cfg, P, "golden_regression.json")
    if not p.exists():
        pytest.skip("golden regression not run yet")
    r = load_json(p)
    assert r["pass"], r
