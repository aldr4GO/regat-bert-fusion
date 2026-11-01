"""Loads aligned cached tensors for an (experiment key, split). All rows follow cands_{set}.json order.

Keys: "A"/"B" (MS MARCO protocols) or a multi-hop dataset ("musique", "musique@random", ...; see common.py).
Candidate lists may have different lengths (closed pools): tensors are padded to the longest list with
`mask` marking valid entries (padded scores / gains are 0).
"""
from dataclasses import dataclass, field

import torch

from src.common import base_key, cache_path, data_cache_path, is_msmarco, load_emb, load_json, load_scores
from src.metrics import ideal_gains


@dataclass
class SplitData:
    name: str
    qids: list
    pids: list                      # [N][K_q] candidate pids (variable length)
    qrels: dict                     # {qid: {pid: rel}}
    gains: torch.Tensor             # [N, K] graded relevance of each candidate (0 when padded)
    labels: torch.Tensor            # [N, K] binary relevance (training / RR / AP)
    ideal: torch.Tensor             # [N, cutoff] ideal gains from full qrels
    qlen: torch.Tensor              # [N] query length in words
    scores: dict = field(default_factory=dict)   # name -> [N, K]
    q_emb: torch.Tensor = None      # [N, d]
    mask: torch.Tensor = None       # [N, K] bool, valid candidates
    n_rel: torch.Tensor = None      # [N] number of relevant passages in the full qrels (binary level)
    meta: list = None               # [N] {hops, type, level, n_words} (multi-hop datasets)
    key: str = "A"

    def __len__(self):
        return len(self.qids)

    def subset(self, idx):
        idx = torch.as_tensor(idx, dtype=torch.long)
        il = idx.tolist()
        return SplitData(self.name, [self.qids[i] for i in il], [self.pids[i] for i in il],
                         {self.qids[i]: self.qrels[self.qids[i]] for i in il}, self.gains[idx],
                         self.labels[idx], self.ideal[idx], self.qlen[idx],
                         {k: v[idx] for k, v in self.scores.items()},
                         None if self.q_emb is None else self.q_emb[idx], self.mask[idx], self.n_rel[idx],
                         None if self.meta is None else [self.meta[i] for i in il], self.key)


def binary_level(split):
    return 2 if split.startswith("trec") else 1


def _rows(cache, qids):
    index = {q: i for i, q in enumerate(cache["qids"])}
    return cache["scores"][torch.tensor([index[q] for q in qids], dtype=torch.long)]


def _dataset_files(cfg, key, split, _memo):
    """(qids, cands, qrels_all, queries, meta, physical set) for any key."""
    if is_msmarco(key):
        from src.data import load_cands, load_qrels, load_queries, load_splits, physical_set
        pset = physical_set(key, split)
        mk = ("meta", cfg["profile"], pset)
        if mk not in _memo:
            _memo[mk] = (load_cands(cfg, pset), load_qrels(cfg), load_queries(cfg))
        cands, qrels_all, queries = _memo[mk]
        return load_splits(cfg, key)[split], cands, qrels_all, queries, None, pset
    mk = ("ds", cfg["profile"], base_key(key), split)
    if mk not in _memo:
        f = lambda n: load_json(data_cache_path(cfg, n, key))
        _memo[mk] = (f("splits.json")[split], f(f"cands_{split}.json"), f("qrels.json"), f("queries.json"),
                     f("meta.json"))
    return (*_memo[mk], split)


def load_split(cfg, protocol, split, scores=("sem", "ctx", "bi", "bm25"), q_emb=True, cutoff=10,
               _memo={}):
    """scores: which cached score matrices to load (ctx is specific to the key's GAT)."""
    qids, cands, qrels_all, queries, meta_all, pset = _dataset_files(cfg, protocol, split, _memo)
    pids = [cands[q] for q in qids]
    K = max(len(p) for p in pids)
    if is_msmarco(protocol):
        assert all(len(p) == K for p in pids), "candidate lists must have identical length"
    mask = torch.zeros(len(qids), K, dtype=torch.bool)
    for i, p in enumerate(pids):
        mask[i, :len(p)] = True
    qrels = {q: qrels_all.get(q, {}) for q in qids}
    gains = torch.zeros(len(qids), K)
    for i, (q, pl) in enumerate(zip(qids, pids)):
        gains[i, :len(pl)] = torch.tensor([qrels[q].get(p, 0) for p in pl], dtype=torch.float)
    gains = gains.clamp_min(0)
    lvl = binary_level(split)
    labels = (gains >= lvl).float()
    n_rel = torch.tensor([sum(v >= lvl for v in qrels[q].values()) for q in qids], dtype=torch.float)
    d = SplitData(split, qids, pids, qrels, gains, labels, ideal_gains([qrels[q] for q in qids], cutoff),
                  torch.tensor([len(queries[q].split()) for q in qids], dtype=torch.float),
                  mask=mask, n_rel=n_rel, meta=None if meta_all is None else [meta_all[q] for q in qids],
                  key=protocol)
    for s in scores:
        owner = protocol if s == "ctx" else base_key(protocol)   # only ctx depends on the key's GAT
        c = load_scores(cache_path(cfg, f"{s}_{pset}.pt", owner))
        rows = _rows(c, qids)
        assert rows.shape == (len(qids), K), f"{s}_{pset} misaligned: {tuple(rows.shape)} vs {(len(qids), K)}"
        d.scores[s] = rows.masked_fill(~mask, 0.0) if not bool(mask.all()) else rows
    if q_emb:
        ek = ("q_emb", cfg["profile"], base_key(protocol))
        e = _memo.get(ek)
        if e is None:
            e = _memo[ek] = load_emb(data_cache_path(cfg, "q_emb.pt", protocol))
        d.q_emb = e["emb"][torch.tensor([e["index"][q] for q in qids], dtype=torch.long)]
    return d


def gather_passage_emb(cfg, pids, p_emb=None, dtype=torch.float16, key="A"):
    """pids: [N][K_q] -> [N, K, d] tensor (padded positions hold passage 0; use the mask)."""
    if p_emb is None:
        p_emb = load_emb(data_cache_path(cfg, "p_emb.pt", key), cast=False)
    K = max(len(p) for p in pids)
    idx = torch.zeros(len(pids), K, dtype=torch.long)
    for i, pl in enumerate(pids):
        idx[i, :len(pl)] = torch.tensor([p_emb["index"][p] for p in pl], dtype=torch.long)
    return p_emb["emb"][idx].to(dtype)


def pids_mask(pids):
    K = max(len(p) for p in pids)
    m = torch.zeros(len(pids), K, dtype=torch.bool)
    for i, p in enumerate(pids):
        m[i, :len(p)] = True
    return m
