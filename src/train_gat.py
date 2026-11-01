"""Stage 3: train the GAT once per experiment key (encoders frozen, listwise InfoNCE), then freeze it.

Usage:
  python -m src.train_gat --protocol A                 # main GAT -> cache/gat_best.pt
  python -m src.train_gat --protocol B                 # leak-free GAT -> cache/B/gat_best.pt
  python -m src.train_gat --protocol musique           # multi-hop dataset (knn_emb, k_nn from multihop.yaml)
  python -m src.train_gat --protocol musique@random    # H4 graph type -> cache/musique/graphs/random/gat_best.pt
  python -m src.train_gat --protocol A --control random_graph   # Sec. 6.6 controls (val only)
  python -m src.train_gat --protocol musique --control all      # closed pools: k_nn sweep {2,3,5} on val
  python -m src.train_gat --protocol A --summarize    # -> results/A/gat_controls.csv
"""
import argparse
import time

import torch

from src.common import (cache_path, data_cache_path, dataset_cfg, device, get_logger, is_msmarco, load_config,
                        load_emb, results_path, save_json, seed_everything, set_env, split_key)
from src.gat import PassageGAT, contextual_scores
from src.graph import build_adjacency
from src.metrics import ndcg_at_k, rr_at_k
from src.store import gather_passage_emb, load_split

log = get_logger("train_gat")


def graph_config(cfg, key):
    """Graph used by the key's GAT: MS MARCO from gat.yaml, multi-hop from multihop.yaml (+ graph variant)."""
    if is_msmarco(key):
        return dict(cfg["gat"]["graph"])
    dc, variant = dataset_cfg(cfg, key), split_key(key)[1]
    mode = variant if variant in cfg["multihop"]["graph_types"] else "knn_emb"
    return {"mode": mode, "k_nn": dc["k_nn"], "edge_min_sim": cfg["multihop"]["edge_min_sim"],
            "random_seed": cfg["gat"]["graph"]["random_seed"]}


def controls(cfg, key):
    if is_msmarco(key):
        return cfg["gat"]["controls"]
    ms = cfg["multihop"]["edge_min_sim"]
    return [{"name": f"knn_k{k}", "mode": "knn_emb", "k_nn": k, "edge_min_sim": ms}
            for k in cfg["multihop"]["knn_sweep"]]


def title_adjacency(cfg, key, split, gcfg):
    if gcfg["mode"] not in ("title_link", "hybrid"):
        return None
    return torch.load(data_cache_path(cfg, f"titlelink_{split}.pt", key), weights_only=False)["adj"]


def prepare(cfg, protocol, split, p_emb, gcfg, dev):
    d = load_split(cfg, protocol, split, scores=("bi",), q_emb=True)
    P = gather_passage_emb(cfg, d.pids, p_emb, key=protocol).to(dev)
    gen = torch.Generator().manual_seed(cfg["gat"]["graph"]["random_seed"])
    mask = None if bool(d.mask.all()) else d.mask.to(dev)
    A = build_adjacency(P, gcfg["k_nn"], gcfg["edge_min_sim"], gcfg["mode"], generator=gen, mask=mask,
                        title_adj=title_adjacency(cfg, protocol, split, gcfg))
    return d, P, d.q_emb.to(dev), A


@torch.no_grad()
def score_all(gat, P, Q, A, bs, mask=None):
    gat.eval()
    out = [contextual_scores(gat, P[i:i + bs].float(), Q[i:i + bs], A[i:i + bs]).float().cpu()
           for i in range(0, P.shape[0], bs)]
    s = torch.cat(out)
    return s if mask is None or bool(mask.all()) else s.masked_fill(~mask.cpu(), 0.0)


def infonce(s, labels, tau, mask=None):
    """L = -log( sum_pos exp(s/tau) / sum_all exp(s/tau) ), averaged over queries (padded entries ignored)."""
    z = s / tau
    if mask is not None:
        z = z.masked_fill(~mask, float("-inf"))
    pos = z.masked_fill(labels <= 0, float("-inf"))
    return (torch.logsumexp(z, -1) - torch.logsumexp(pos, -1)).mean()


def mean_degree(A, mask):
    """Mean number of neighbours (self-loop excluded) over valid nodes."""
    deg = A.float().sum(-1) - 1
    return float(deg[mask.to(A.device)].mean()) if mask is not None else float(deg.mean())


def train(cfg, protocol, gcfg, tag, dev):
    tc, mc = cfg["gat"]["train"], cfg["gat"]["model"]
    seed_everything(tc["seed"])
    p_emb = load_emb(data_cache_path(cfg, "p_emb.pt", protocol), cast=False)
    tr, Ptr, Qtr, Atr = prepare(cfg, protocol, "gat_train", p_emb, gcfg, dev)
    va, Pva, Qva, Ava = prepare(cfg, protocol, "val", p_emb, gcfg, dev)
    del p_emb
    ytr, mtr = tr.labels.to(dev), tr.mask.to(dev)
    log.info("[%s] graph=%s gat_train=%d val=%d mean degree(train)=%.2f", tag, gcfg, len(tr), len(va),
             mean_degree(Atr, tr.mask))

    gat = PassageGAT(Ptr.shape[-1], mc["hidden"], mc["heads"], mc["dropout"]).to(dev)
    opt = torch.optim.AdamW(gat.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    ebs = tc["eval_batch_queries"]

    def val_metrics():
        s = score_all(gat, Pva, Qva, Ava, ebs, va.mask)
        return (float(ndcg_at_k(s, va.gains, va.ideal, mask=va.mask).mean()),
                float(rr_at_k(s, va.labels, mask=va.mask).mean()))

    bi_ndcg = float(ndcg_at_k(va.scores["bi"], va.gains, va.ideal, mask=va.mask).mean())
    n0, r0 = val_metrics()
    hist = [{"epoch": 0, "val_ndcg@10": n0, "val_mrr@10": r0}]
    log.info("[%s] s_bi val NDCG@10=%.4f | untrained GAT val NDCG@10=%.4f", tag, bi_ndcg, n0)
    best, best_state, bad = n0, {k: v.clone() for k, v in gat.state_dict().items()}, 0
    gen = torch.Generator().manual_seed(tc["seed"])
    for ep in range(1, tc["max_epochs"] + 1):
        gat.train()
        t0, tot = time.time(), 0.0
        perm = torch.randperm(len(tr), generator=gen)
        for i in range(0, len(tr), tc["batch_queries"]):
            b = perm[i:i + tc["batch_queries"]].to(dev)
            s = contextual_scores(gat, Ptr[b].float(), Qtr[b], Atr[b])
            loss = infonce(s, ytr[b], tc["tau"], mtr[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        n, r = val_metrics()
        hist.append({"epoch": ep, "train_loss": tot / len(tr), "val_ndcg@10": n, "val_mrr@10": r,
                     "sec": time.time() - t0})
        log.info("[%s] epoch %d loss %.4f val NDCG@10 %.4f MRR@10 %.4f (%.1fs)", tag, ep, tot / len(tr), n, r,
                 time.time() - t0)
        if n > best:
            best, best_state, bad = n, {k: v.clone() for k, v in gat.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= tc["patience"]:
                log.info("[%s] early stop at epoch %d", tag, ep)
                break
    gat.load_state_dict(best_state)
    gat.eval()
    gat.requires_grad_(False)
    s = score_all(gat, Pva, Qva, Ava, ebs, va.mask)
    summary = {"tag": tag, "protocol": protocol, "graph": gcfg,
               "val_ndcg@10": float(ndcg_at_k(s, va.gains, va.ideal, mask=va.mask).mean()),
               "val_mrr@10": float(rr_at_k(s, va.labels, mask=va.mask).mean()), "val_ndcg@10_s_bi": bi_ndcg,
               "best_epoch": max(hist, key=lambda h: h["val_ndcg@10"])["epoch"], "history": hist,
               "mean_degree_train": mean_degree(Atr, tr.mask)}
    return gat, summary


def summarize_controls(cfg, protocol):
    import pandas as pd
    from src.common import load_json
    rows = []
    for c in controls(cfg, protocol):
        p = results_path(cfg, protocol, "gat_controls", f"{c['name']}.json")
        if p.exists():
            r = load_json(p)
            rows.append({"control": c["name"], **{k: v for k, v in r["graph"].items() if k != "random_seed"},
                         "val_ndcg@10": r["val_ndcg@10"], "val_mrr@10": r["val_mrr@10"],
                         "val_ndcg@10_s_bi": r["val_ndcg@10_s_bi"], "best_epoch": r["best_epoch"],
                         "mean_degree": r["mean_degree_train"]})
    df = pd.DataFrame(rows)
    df.to_csv(results_path(cfg, protocol, "gat_controls.csv"), index=False, float_format="%.4f")
    log.info("GAT controls (val):\n%s", df.to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="A", help="A | B | <dataset> | <dataset>@<graph type>")
    ap.add_argument("--control", default=None, help="control name (gat.yaml / knn sweep), or 'all'")
    ap.add_argument("--summarize", action="store_true", help="collect control results into a csv")
    args = ap.parse_args()
    cfg = load_config()
    set_env(cfg)
    if args.summarize:
        summarize_controls(cfg, args.protocol)
        return
    dev = device()
    gc = cfg["gat"]
    if args.control is None:
        assert split_key(args.protocol)[1] != "msmarco_gat", "msmarco_gat reuses the MS MARCO GAT; nothing to train"
        gcfg = graph_config(cfg, args.protocol)
        out = cache_path(cfg, "gat_best.pt", args.protocol)
        if out.exists():
            raise SystemExit(f"{out} already exists; the GAT is trained once and frozen. Delete it explicitly "
                             "if you really want to retrain.")
        gat, summary = train(cfg, args.protocol, gcfg, f"main_{args.protocol}", dev)
        torch.save({"state_dict": gat.state_dict(), "graph": gcfg, "model": gc["model"],
                    "d": gat.norm.normalized_shape[0]}, out)
        save_json(summary, results_path(cfg, args.protocol, "gat_train_summary.json"))
        log.info("saved frozen GAT to %s (val NDCG@10 %.4f, s_bi %.4f)", out, summary["val_ndcg@10"],
                 summary["val_ndcg@10_s_bi"])
        return
    ctrl = controls(cfg, args.protocol)
    sel = ctrl if args.control == "all" else [c for c in ctrl if c["name"] == args.control]
    assert sel, f"unknown control {args.control}"
    for c in sel:
        gcfg = {"mode": c["mode"], "k_nn": c["k_nn"], "edge_min_sim": c["edge_min_sim"]}
        _, summary = train(cfg, args.protocol, gcfg, c["name"], dev)
        save_json(summary, results_path(cfg, args.protocol, "gat_controls", f"{c['name']}.json"))


if __name__ == "__main__":
    main()
