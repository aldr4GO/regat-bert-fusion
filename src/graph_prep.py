"""Title-link adjacency and edge statistics per dataset (multi-hop plan Sec. 3.3).

title_link: paragraphs i and j are linked if the normalised title of j occurs as a whole phrase in the text of i
(or vice versa). Titles are lower-cased, parenthetical suffixes such as "(film)" are stripped, and titles shorter
than `min_title_chars` are ignored. Each text is scanned once with an alternation of the pool's titles
(leftmost-longest, non-overlapping matches).
edge_stats: per graph type, mean degree (self-loop excluded, valid nodes) and the fraction of gold-gold pairs
that are connected, next to the overall edge density (the chance level for a random pair).

Usage:  python -m src.graph_prep --protocol musique|hotpotqa|beir_hotpotqa|twowiki|A
Writes: cache/<ds>/titlelink_{split}.pt, cache/<ds>/edge_stats.json, results/edge_stats.csv (all datasets)
"""
import argparse
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from src.common import (data_cache_path, dataset_cfg, get_logger, is_msmarco, load_config, load_emb, load_json,
                        path, save_json, set_env)
from src.datasets.common import norm_title
from src.graph import build_adjacency
from src.store import gather_passage_emb, load_split
from src.train_gat import graph_config

log = get_logger("graph_prep")
GRAPH_TYPES = ["none", "random", "knn_emb", "title_link", "hybrid"]


def title_links(titles, texts, min_chars):
    K = len(titles)
    A = np.zeros((K, K), dtype=bool)
    groups = defaultdict(list)
    for j, t in enumerate(titles):
        t = norm_title(t)
        if len(t) >= min_chars:
            groups[t].append(j)
    if not groups:
        return A
    alts = "|".join(re.escape(t) for t in sorted(groups, key=len, reverse=True))
    pat = re.compile(r"(?<!\w)(" + alts + r")(?!\w)")
    for i, text in enumerate(texts):
        for m in pat.finditer(text.lower()):
            for j in groups[m.group(1)]:
                if j != i:
                    A[i, j] = True
    return A | A.T


def build_title_adjacency(cfg, key, split):
    from src.datasets.texts import cands as load_c, raw_passages
    c = load_c(cfg, key, split)
    qids = list(c)
    raw = raw_passages(cfg, key, {p for pl in c.values() for p in pl})
    K = max(len(v) for v in c.values())
    adj = torch.zeros(len(qids), K, K, dtype=torch.bool)
    mc = cfg["multihop"]["title_link"]["min_title_chars"]
    for n, q in enumerate(qids):
        pl = c[q]
        A = title_links([raw[p]["title"] for p in pl], [raw[p]["text"] for p in pl], mc)
        adj[n, :len(pl), :len(pl)] = torch.from_numpy(A)
    torch.save({"qids": qids, "adj": adj}, data_cache_path(cfg, f"titlelink_{split}.pt", key))
    return adj


def edge_stats(cfg, key, split, title_adj, p_emb):
    d = load_split(cfg, key, split, scores=(), q_emb=False)
    P = gather_passage_emb(cfg, d.pids, p_emb, key=key)
    mask = None if bool(d.mask.all()) else d.mask
    base = graph_config(cfg, key)
    gold = d.labels.bool() & d.mask
    two_gold = gold.sum(-1) >= 2
    rows = []
    for gt in GRAPH_TYPES:
        if gt in ("title_link", "hybrid") and title_adj is None:
            continue
        gen = torch.Generator().manual_seed(cfg["gat"]["graph"]["random_seed"])
        A = build_adjacency(P, base["k_nn"], base["edge_min_sim"], gt, generator=gen, mask=mask, title_adj=title_adj)
        A = A & ~torch.eye(A.shape[-1], dtype=torch.bool)
        valid_pair = d.mask.unsqueeze(-1) & d.mask.unsqueeze(1) & ~torch.eye(A.shape[-1], dtype=torch.bool)
        gg = gold.unsqueeze(-1) & gold.unsqueeze(1) & ~torch.eye(A.shape[-1], dtype=torch.bool)
        gg = gg & two_gold.view(-1, 1, 1)
        rows.append({"dataset": key, "split": split, "graph_type": gt, "k_nn": base["k_nn"],
                     "mean_degree": float(A.float().sum(-1)[d.mask].mean()),
                     "edge_density": float(A[valid_pair].float().mean()),
                     "n_gold_pairs": int(gg.sum() // 2),
                     "frac_gold_pairs_connected": float(A[gg].float().mean()) if gg.any() else float("nan")})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    args = ap.parse_args()
    cfg = load_config()
    set_env(cfg)
    key = args.protocol
    dc = dataset_cfg(cfg, key)
    rows = []
    p_emb = load_emb(data_cache_path(cfg, "p_emb.pt", key), cast=False)
    if is_msmarco(key):     # no titles: knn / random / none only, on val and test
        for s in ("val", "test"):
            rows += edge_stats(cfg, key, s, None, p_emb)
    else:
        splits = ["gat_train", "fusion_train", "val", "test"] if dc["kind"] == "closed" else ["val", "test"]
        for s in splits:
            adj = build_title_adjacency(cfg, key, s)
            log.info("%s %s: title_link mean degree %.2f", key, s,
                     float(adj.float().sum(-1)[load_split(cfg, key, s, scores=(), q_emb=False).mask].mean()))
            if s in ("val", "test"):
                rows += edge_stats(cfg, key, s, adj, p_emb)
    out = data_cache_path(cfg, "edge_stats.json", key)
    save_json(rows, out)
    # collect every dataset's stats into one csv
    allrows = []
    for ds in ["A"] + list(cfg["multihop"]["datasets"]):
        p = data_cache_path(cfg, "edge_stats.json", ds)
        if p.exists():
            allrows += load_json(p)
    df = pd.DataFrame(allrows)
    df.to_csv(path(cfg, "results_dir", "edge_stats.csv"), index=False, float_format="%.4f")
    log.info("edge stats (%s):\n%s", key, pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
