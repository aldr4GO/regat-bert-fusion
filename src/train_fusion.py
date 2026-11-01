"""Stage 5: train one fusion configuration on cached tensors (CPU). Only the fusion module is trainable.

Usage:  python -m src.train_fusion --protocol A --norm zscore --dep QS --seed 0 [--loss listwise|bce] [--lam 0.01]
        python -m src.train_fusion --protocol A --norm none --dep paper --seed 0      (paper row, BCE)
        python -m src.train_fusion --protocol musique --norm minmax --dep QSC --seed 0  (E5 complexity features)
Writes: results/{key}/runs/{name}.json (+ .npz with per-query metrics and alphas)
"""
import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common import (cache_path, dataset_cfg, get_logger, load_config, load_json, results_path,
                        seed_everything, set_env, test_sets)
from src.features import FeatureScaler, score_features
from src.fusion import FusionModel, PaperFusion
from src.metrics import ndcg_at_k
from src.normalize import normalize
from src.store import load_split

log = get_logger("train_fusion")


@dataclass
class FusionData:
    q_emb: torch.Tensor     # [N, d]
    feats: torch.Tensor     # [N, 12] standardised
    sem_n: torch.Tensor     # [N, K]
    ctx_n: torch.Tensor     # [N, K]
    labels: torch.Tensor    # [N, K] binary (training only)
    gains: torch.Tensor     # [N, K]
    ideal: torch.Tensor     # [N, cutoff]
    mask: torch.Tensor = None    # [N, K] bool valid candidates (variable pool sizes)
    cfeats: torch.Tensor = None  # [N, 2] standardised query length / entity count (QSC only)


# ------------------------------------------------------------------ data preparation
_SPLIT_MEMO, _FEAT_MEMO = {}, {}


def raw_split(cfg, protocol, split):
    key = (cfg["profile"], protocol, split)
    if key not in _SPLIT_MEMO:
        _SPLIT_MEMO[key] = load_split(cfg, protocol, split, scores=("sem", "ctx", "bi", "bm25"),
                                      cutoff=cfg["fusion"]["eval"]["cutoff"])
    return _SPLIT_MEMO[key]


def overlap_m(cfg, protocol):
    """Top-m overlap feature: m = 10 for K >= 50 (MS MARCO, BEIR), m = 3 for closed pools (multihop.yaml)."""
    dc = dataset_cfg(cfg, protocol)
    return cfg["fusion"]["features"]["top_overlap_k"] if dc is None else dc["overlap_m"]


def raw_features(cfg, protocol, split):
    key = (cfg["profile"], protocol, split)
    if key not in _FEAT_MEMO:
        d = raw_split(cfg, protocol, split)
        _FEAT_MEMO[key] = score_features(d.scores["sem"], d.scores["ctx"], cfg["fusion"]["eps"],
                                         overlap_m(cfg, protocol), mask=d.mask)
    return _FEAT_MEMO[key]


def feature_scaler(cfg, protocol):
    p = cache_path(cfg, "feat_stats.pt", protocol)
    if p.exists():
        st = torch.load(p, weights_only=False)
        return FeatureScaler(st["mean"], st["std"], cfg["fusion"]["eps"])
    sc = FeatureScaler(eps=cfg["fusion"]["eps"]).fit(raw_features(cfg, protocol, "fusion_train"))
    torch.save(sc.state(), p)
    return sc


def raw_cfeats(cfg, protocol, split):
    """E5 complexity features from the query text: [n_words, n_entities] (never from gold labels)."""
    from src.query_meta import query_meta
    qm = query_meta(cfg, protocol)
    d = raw_split(cfg, protocol, split)
    return torch.tensor([[qm[q]["n_words"], qm[q]["n_entities"]] for q in d.qids], dtype=torch.float)


def cfeat_scaler(cfg, protocol):
    p = cache_path(cfg, "cfeat_stats.pt", protocol)
    if p.exists():
        st = torch.load(p, weights_only=False)
        return FeatureScaler(st["mean"], st["std"], cfg["fusion"]["eps"])
    sc = FeatureScaler(eps=cfg["fusion"]["eps"]).fit(raw_cfeats(cfg, protocol, "fusion_train"))
    torch.save({"mean": sc.mean, "std": sc.std, "names": ["n_words", "n_entities"]}, p)
    return sc


def fusion_data(cfg, protocol, split, norm, with_cfeats=False):
    d = raw_split(cfg, protocol, split)
    eps = cfg["fusion"]["eps"]
    cf = cfeat_scaler(cfg, protocol)(raw_cfeats(cfg, protocol, split)) if with_cfeats else None
    return FusionData(d.q_emb, feature_scaler(cfg, protocol)(raw_features(cfg, protocol, split)),
                      normalize(norm, d.scores["sem"], eps, d.mask), normalize(norm, d.scores["ctx"], eps, d.mask),
                      d.labels, d.gains, d.ideal, d.mask, cf)


def load_alpha0(cfg, protocol, norm):
    a = load_json(results_path(cfg, protocol, "alpha0.json"))[norm]["alpha0"]
    lo, hi = cfg["fusion"]["alpha0_clip"]
    return min(max(a, lo), hi)


# ------------------------------------------------------------------ training
def listwise_loss(scores, labels, mask=None):
    """-log sum_{pos} softmax(scores)_p, averaged over queries with >= 1 positive (masked softmax)."""
    has = labels.sum(-1) > 0
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    pos = scores.masked_fill(labels <= 0, float("-inf"))
    l = torch.logsumexp(scores, -1) - torch.logsumexp(pos, -1)
    return l[has].mean()


def bce_loss(logits, y, mask=None):
    if mask is None or bool(mask.all()):
        return F.binary_cross_entropy_with_logits(logits, y)
    l = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (l * mask).sum() / mask.sum()


def build_model(dep, d, n_feat, alpha0, fcfg, loss, n_cfeat=0):
    if dep == "paper":
        return PaperFusion(fcfg)
    m = FusionModel(dep, d, n_feat, alpha0, fcfg, n_cfeat)
    if loss == "bce":
        m.bias = nn.Parameter(torch.tensor(0.0))
    return m


def model_scores(model, data, idx=None):
    """Returns (scores used by the loss, fused ranking scores, alpha)."""
    sl = (lambda t: t) if idx is None else (lambda t: t[idx])
    cf = None if data.cfeats is None else sl(data.cfeats)
    fused, alpha = model(sl(data.q_emb), sl(data.feats), sl(data.sem_n), sl(data.ctx_n), cf)
    return fused, alpha


@torch.no_grad()
def predict(model, data):
    model.eval()
    fused, alpha = model_scores(model, data)
    return fused, alpha


def train_fusion(fcfg, dep, alpha0, seed, train_data, val_data, loss="listwise", lam=None):
    """Pure function over tensors. Returns (best model, history)."""
    tc = fcfg["train"]
    lam = tc["lambda"] if lam is None else lam
    seed_everything(seed)
    n_cf = 0 if train_data.cfeats is None else train_data.cfeats.shape[1]
    model = build_model(dep, train_data.q_emb.shape[1], train_data.feats.shape[1], alpha0, fcfg, loss, n_cf)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    N = train_data.q_emb.shape[0]

    def val_ndcg():
        f, _ = predict(model, val_data)
        return float(ndcg_at_k(f, val_data.gains, val_data.ideal, mask=val_data.mask).mean())

    best = val_ndcg()
    best_state, best_ep, bad, hist = {k: v.clone() for k, v in model.state_dict().items()}, 0, 0, []
    for ep in range(1, tc["max_epochs"] + 1):
        model.train()
        t0, tot = time.time(), 0.0
        perm = torch.randperm(N, generator=gen)
        for i in range(0, N, tc["batch_queries"]):
            b = perm[i:i + tc["batch_queries"]]
            fused, alpha = model_scores(model, train_data, b)
            y = train_data.labels[b]
            m = None if train_data.mask is None else train_data.mask[b]
            if dep == "paper":
                L = bce_loss(fused + model.bias if model.bias is not None else fused, y, m)
            else:
                z = fused / model.log_T.exp()
                if loss == "listwise":
                    L = listwise_loss(z, y, m)
                else:
                    L = bce_loss(z + model.bias, y, m)
                if lam > 0:
                    L = L + lam * (alpha - alpha0).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            L.backward()
            opt.step()
            tot += L.item() * len(b)
        v = val_ndcg()
        hist.append({"epoch": ep, "loss": tot / N, "val_ndcg@10": v, "sec": round(time.time() - t0, 3)})
        if v > best:
            best, best_ep, bad = v, ep, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": hist, "best_epoch": best_ep, "best_val_ndcg@10": best}


# ------------------------------------------------------------------ one experiment
def run_name(protocol, norm, dep, loss, lam, seed):
    return f"{protocol}_{norm}_{dep}_{loss}_lam{lam:g}_s{seed}"


def run_one(cfg, protocol, norm, dep, seed, loss="listwise", lam=None, overwrite=False):
    from src.evaluate import save_run
    fc = cfg["fusion"]
    lam = fc["train"]["lambda"] if lam is None else lam
    if dep == "paper":
        loss, lam = "bce", 0.0
    name = run_name(protocol, norm, dep, loss, lam, seed)
    out = results_path(cfg, protocol, "runs", f"{name}.json")
    if out.exists() and not overwrite:
        return name
    alpha0 = load_alpha0(cfg, protocol, norm)
    cf = dep == "QSC"
    tr, va = fusion_data(cfg, protocol, "fusion_train", norm, cf), fusion_data(cfg, protocol, "val", norm, cf)
    t0 = time.time()
    model, info = train_fusion(fc, dep, alpha0, seed, tr, va, loss, lam)
    scores, alphas = {}, {}
    for s in ["val"] + test_sets(cfg, protocol):
        f, a = predict(model, fusion_data(cfg, protocol, s, norm, cf))
        scores[s], alphas[s] = f, a
    extra = {}
    if dep == "paper":
        extra = {"a": model.a.item(), "b": model.b.item()}
    else:
        extra = {"T": model.log_T.exp().item()}
    meta = {"protocol": protocol, "norm": norm, "dep": dep, "seed": seed, "loss": loss, "lam": lam,
            "alpha0": alpha0, "train_sec": round(time.time() - t0, 2), **info, **extra}
    save_run(cfg, protocol, name, meta, scores, alphas)
    log.info("%s: best epoch %d val NDCG@10 %.4f (%.1fs)", name, info["best_epoch"], info["best_val_ndcg@10"],
             time.time() - t0)
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="A", help="A | B | <dataset>[@variant]")
    ap.add_argument("--norm", required=True, choices=["none", "minmax", "zscore", "rank"])
    ap.add_argument("--dep", required=True, choices=["fixed", "Q", "QS", "S", "QSC", "paper"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", default="listwise", choices=["listwise", "bce"])
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    set_env(cfg)
    run_one(cfg, args.protocol, args.norm, args.dep, args.seed, args.loss, args.lam, args.overwrite)


if __name__ == "__main__":
    main()
