import numpy as np

from src.evaluate import config_key, paired_bootstrap


def test_config_key():
    assert config_key("A_zscore_QS_listwise_lam0.01_s3") == "zscore_QS_listwise_lam0.01"
    assert config_key("B_rank_fixedgrid") == "rank_fixedgrid"
    assert config_key("A_ref_bm25") == "ref_bm25"


def test_bootstrap_identical_and_shifted():
    rng = np.random.default_rng(0)
    a = rng.random(500)
    d, lo, hi, p = paired_bootstrap(a, a, B=1000)
    assert d == 0 and lo == 0 and hi == 0 and p == 1.0
    d, lo, hi, p = paired_bootstrap(a + 0.05, a, B=1000)
    assert abs(d - 0.05) < 1e-12 and lo > 0 and p <= 1e-3
    d, lo, hi, p = paired_bootstrap(a + rng.normal(0, 0.2, 500), a, B=2000)
    assert lo < d < hi
