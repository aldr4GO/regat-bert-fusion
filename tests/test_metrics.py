import math

import torch

from src.metrics import ideal_gains, ndcg_at_k, rr_at_k


def test_perfect_ranking():
    gains = torch.tensor([[0, 1, 0, 1, 0.0]])
    ideal = ideal_gains([{"a": 1, "b": 1}], 10)
    assert ndcg_at_k(gains, gains, ideal).item() == 1.0
    assert rr_at_k(gains, gains).item() == 1.0


def test_toy_manual():
    # ranking puts relevant docs at ranks 2 and 4; a third relevant doc is not retrieved
    scores = torch.tensor([[5.0, 4, 3, 2, 1]])
    gains = torch.tensor([[0, 1, 0, 1, 0.0]])
    ideal = ideal_gains([{"x": 1, "y": 1, "z": 1}], 10)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert abs(ndcg_at_k(scores, gains, ideal).item() - dcg / idcg) < 1e-6
    assert abs(rr_at_k(scores, gains).item() - 0.5) < 1e-7


def test_matches_ir_measures():
    import ir_measures
    from ir_measures import RR, nDCG
    g = torch.Generator().manual_seed(0)
    N, K = 30, 100
    scores = torch.randn(N, K, generator=g)
    gains = (torch.rand(N, K, generator=g) < 0.03).float() * torch.randint(1, 4, (N, K), generator=g)
    qrels = {}
    for i in range(N):
        qrels[str(i)] = {f"d{j}": int(gains[i, j]) for j in range(K) if gains[i, j] > 0}
        qrels[str(i)]["unretrieved"] = 2
    run = {str(i): {f"d{j}": float(scores[i, j]) for j in range(K)} for i in range(N)}
    ref = ir_measures.calc_aggregate([nDCG @ 10, RR @ 10], qrels, run)
    ideal = ideal_gains([qrels[str(i)] for i in range(N)], 10)
    assert abs(ndcg_at_k(scores, gains, ideal).mean().item() - ref[nDCG @ 10]) < 1e-5
    assert abs(rr_at_k(scores, (gains > 0).float()).mean().item() - ref[RR @ 10]) < 1e-5
