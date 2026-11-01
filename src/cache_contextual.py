"""Stage 4: run the frozen GAT over every query set and cache s_ctx, then check the sanity gates.

Usage:  python -m src.cache_contextual --protocol A|B|<dataset>|<dataset>@<variant>
        <dataset>@msmarco_gat uses the frozen MS MARCO (protocol A) GAT on the dataset's own graph (E4 zero-shot).
Writes: ctx_{set}.pt under the key's cache dir, results/{key}/stage4_sanity.json
"""
import argparse

import torch

from src.common import (cache_path, data_cache_path, device, get_logger, is_msmarco, load_config, load_emb,
                        load_json, load_scores, results_path, save_json, set_env, split_key)
from src.gat import PassageGAT
from src.graph import build_adjacency
from src.metrics import ndcg_at_k
from src.store import gather_passage_emb, load_split, pids_mask
from src.train_gat import graph_config, score_all, title_adjacency

log = get_logger("cache_contextual")


def load_frozen_gat(cfg, protocol, dev):
    owner = "A" if split_key(protocol)[1] == "msmarco_gat" else protocol
    ck = torch.load(cache_path(cfg, "gat_best.pt", owner), weights_only=False)
    m = ck["model"]
    gat = PassageGAT(ck["d"], m["hidden"], m["heads"], m["dropout"]).to(dev)
    gat.load_state_dict(ck["state_dict"])
    gat.eval()
    gat.requires_grad_(False)
    gcfg = ck["graph"] if owner == protocol else graph_config(cfg, protocol)
    return gat, gcfg


def physical_sets(cfg, protocol):
    if protocol == "A":
        return cfg["data"]["cache_sets"]
    if protocol == "B":   # protocol B splits are all subsets of dev/small (`test`); TREC DL kept as OOD sets
        return ["test", "trec19", "trec20"]
    return ["gat_train", "fusion_train", "val", "test"]


def load_set_cands(cfg, protocol, s):
    if is_msmarco(protocol):
        from src.data import load_cands
        return load_cands(cfg, s)
    return load_json(data_cache_path(cfg, f"cands_{s}.json", protocol))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="A")
    args = ap.parse_args()
    cfg = load_config()
    set_env(cfg)
    dev = device()
    P_ = args.protocol
    gat, gcfg = load_frozen_gat(cfg, P_, dev)
    p_emb = load_emb(data_cache_path(cfg, "p_emb.pt", P_), cast=False)
    q_emb = load_emb(data_cache_path(cfg, "q_emb.pt", P_))
    ebs = cfg["gat"]["train"]["eval_batch_queries"]
    for s in physical_sets(cfg, P_):
        cands = load_set_cands(cfg, P_, s)
        qids = list(cands)
        pids = [cands[q] for q in qids]
        mask = pids_mask(pids)
        P = gather_passage_emb(cfg, pids, p_emb, key=P_).to(dev)
        Q = q_emb["emb"][torch.tensor([q_emb["index"][q] for q in qids])].to(dev)
        gen = torch.Generator().manual_seed(cfg["gat"]["graph"]["random_seed"])
        m = None if bool(mask.all()) else mask.to(dev)
        A = build_adjacency(P, gcfg["k_nn"], gcfg["edge_min_sim"], gcfg["mode"], generator=gen, mask=m,
                            title_adj=title_adjacency(cfg, P_, s, gcfg))
        ctx = score_all(gat, P, Q, A, ebs, mask)
        out = cache_path(cfg, f"ctx_{s}.pt", P_)
        if is_msmarco(P_):
            torch.save({"qids": qids, "scores": ctx.float().cpu()}, out)
        else:
            torch.save({"qids": qids, "scores": ctx.float().cpu(), "mask": mask}, out)
        log.info("ctx_%s (%s): %d queries", s, P_, len(qids))
        del P, A

    # ---- sanity gates (val only)
    for s in physical_sets(cfg, P_):
        c = load_set_cands(cfg, P_, s)
        sem = load_scores(data_cache_path(cfg, f"sem_{s}.pt", P_))
        ctx = load_scores(cache_path(cfg, f"ctx_{s}.pt", P_))
        K = max(len(v) for v in c.values())
        assert sem["qids"] == ctx["qids"] == list(c), f"query order differs across files for {s}"
        assert sem["scores"].shape == ctx["scores"].shape == (len(c), K)
    v = load_split(cfg, P_, "val", scores=("ctx", "bi"), q_emb=False)
    gat_ndcg = float(ndcg_at_k(v.scores["ctx"], v.gains, v.ideal, mask=v.mask).mean())
    bi_ndcg = float(ndcg_at_k(v.scores["bi"], v.gains, v.ideal, mask=v.mask).mean())
    ok = gat_ndcg >= bi_ndcg - cfg["gat"]["sanity"]["max_drop_vs_bi"]
    # graph-type / transfer variants (key "<dataset>@<variant>") are controls: the gate is reported, not enforced
    enforced = (split_key(P_)[1] is None or P_ == "B") and cfg["gat"]["sanity"]["enforce"]
    res = {"val_ndcg@10_gat_only": gat_ndcg, "val_ndcg@10_s_bi": bi_ndcg, "alignment_ok": True, "gate_passed": ok,
           "gate_enforced": enforced, "graph": gcfg}
    save_json(res, results_path(cfg, P_, "stage4_sanity.json"))
    log.info("Stage 4 sanity (%s): %s", P_, res)
    if enforced and not ok:
        raise SystemExit("SANITY GATE FAILED: GAT-only is clearly worse than s_bi on val. Debug residual/tau/"
                         "edge threshold before Stage 5.")


if __name__ == "__main__":
    main()
