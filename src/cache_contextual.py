"""Stage 4: run the frozen GAT over every query set and cache s_ctx, then check the sanity gates.

Usage:  python -m src.cache_contextual --protocol A|B
Writes: cache/ctx_{set}.pt (A) or cache/B/ctx_{set}.pt (B), results/{P}/stage4_sanity.json
"""
import argparse

import torch

from src.common import (cache_path, device, get_logger, load_config, load_emb, load_scores, results_path,
                        save_json, save_scores, set_env)
from src.data import load_cands
from src.gat import PassageGAT
from src.graph import build_adjacency
from src.metrics import ndcg_at_k
from src.store import gather_passage_emb, load_split
from src.train_gat import score_all

log = get_logger("cache_contextual")


def load_frozen_gat(cfg, protocol, dev):
    ck = torch.load(cache_path(cfg, "gat_best.pt", protocol), weights_only=False)
    m = ck["model"]
    gat = PassageGAT(ck["d"], m["hidden"], m["heads"], m["dropout"]).to(dev)
    gat.load_state_dict(ck["state_dict"])
    gat.eval()
    gat.requires_grad_(False)
    return gat, ck["graph"]


def physical_sets(cfg, protocol):
    # protocol B splits are all subsets of dev/small (`test`); TREC DL kept as OOD sets
    return cfg["data"]["cache_sets"] if protocol == "A" else ["test", "trec19", "trec20"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="A", choices=["A", "B"])
    args = ap.parse_args()
    cfg = load_config()
    set_env(cfg)
    dev = device()
    gat, gcfg = load_frozen_gat(cfg, args.protocol, dev)
    p_emb = load_emb(cache_path(cfg, "p_emb.pt"), cast=False)
    q_emb = load_emb(cache_path(cfg, "q_emb.pt"))
    ebs = cfg["gat"]["train"]["eval_batch_queries"]
    for s in physical_sets(cfg, args.protocol):
        cands = load_cands(cfg, s)
        qids = list(cands)
        P = gather_passage_emb(cfg, [cands[q] for q in qids], p_emb).to(dev)
        Q = q_emb["emb"][torch.tensor([q_emb["index"][q] for q in qids])].to(dev)
        gen = torch.Generator().manual_seed(cfg["gat"]["graph"]["random_seed"])
        A = build_adjacency(P, gcfg["k_nn"], gcfg["edge_min_sim"], gcfg["mode"], generator=gen)
        ctx = score_all(gat, P, Q, A, ebs)
        save_scores(qids, ctx, cache_path(cfg, f"ctx_{s}.pt", args.protocol))
        log.info("ctx_%s (%s): %d queries", s, args.protocol, len(qids))
        del P, A

    # ---- sanity gates (val only)
    for s in physical_sets(cfg, args.protocol):
        c = load_cands(cfg, s)
        sem = load_scores(cache_path(cfg, f"sem_{s}.pt"))
        ctx = load_scores(cache_path(cfg, f"ctx_{s}.pt", args.protocol))
        assert sem["qids"] == ctx["qids"] == list(c), f"query order differs across files for {s}"
        assert sem["scores"].shape == ctx["scores"].shape == (len(c), len(next(iter(c.values()))))
    v = load_split(cfg, args.protocol, "val", scores=("ctx", "bi"), q_emb=False)
    gat_ndcg = float(ndcg_at_k(v.scores["ctx"], v.gains, v.ideal).mean())
    bi_ndcg = float(ndcg_at_k(v.scores["bi"], v.gains, v.ideal).mean())
    ok = gat_ndcg >= bi_ndcg - cfg["gat"]["sanity"]["max_drop_vs_bi"]
    res = {"val_ndcg@10_gat_only": gat_ndcg, "val_ndcg@10_s_bi": bi_ndcg, "alignment_ok": True, "gate_passed": ok}
    save_json(res, results_path(cfg, args.protocol, "stage4_sanity.json"))
    log.info("Stage 4 sanity (%s): %s", args.protocol, res)
    if not ok:
        raise SystemExit("SANITY GATE FAILED: GAT-only is clearly worse than s_bi on val. Debug residual/tau/"
                         "edge threshold before Stage 5.")


if __name__ == "__main__":
    main()
