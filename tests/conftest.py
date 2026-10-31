import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import load_config  # noqa: E402
from src.metrics import ideal_gains  # noqa: E402
from src.train_fusion import FusionData  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def make_fusion_data(N=64, K=20, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    labels = torch.zeros(N, K)
    labels[torch.arange(N), torch.randint(0, K, (N,), generator=g)] = 1.0
    sem = torch.randn(N, K, generator=g) + 2 * labels
    ctx = torch.randn(N, K, generator=g) + labels
    q = torch.nn.functional.normalize(torch.randn(N, d, generator=g), dim=-1)
    from src.features import FeatureScaler, score_features
    F = score_features(sem, ctx)
    F = FeatureScaler().fit(F)(F)
    from src.normalize import zscore
    qrels = [{str(j): 1 for j in labels[i].nonzero().flatten().tolist()} for i in range(N)]
    return FusionData(q, F, zscore(sem), zscore(ctx), labels, labels.clone(), ideal_gains(qrels, 10))
