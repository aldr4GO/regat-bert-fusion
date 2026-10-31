"""Loads aligned cached tensors for a (protocol, split). All rows follow cands_{set}.json order."""
from dataclasses import dataclass, field

import torch

from src.common import cache_path, load_emb, load_scores
from src.data import load_cands, load_qrels, load_queries, load_splits, physical_set
from src.metrics import ideal_gains


@dataclass
class SplitData:
    name: str
    qids: list
    pids: list                      # [N][K] candidate pids
    qrels: dict                     # {qid: {pid: rel}}
    gains: torch.Tensor             # [N, K] graded relevance of each candidate
    labels: torch.Tensor            # [N, K] binary relevance (training / RR / AP)
    ideal: torch.Tensor             # [N, cutoff] ideal gains from full qrels
    qlen: torch.Tensor              # [N] query length in words
    scores: dict = field(default_factory=dict)   # name -> [N, K]
    q_emb: torch.Tensor = None      # [N, d]

    def __len__(self):
        return len(self.qids)

    def subset(self, idx):
        idx = torch.as_tensor(idx, dtype=torch.long)
        il = idx.tolist()
        return SplitData(self.name, [self.qids[i] for i in il], [self.pids[i] for i in il],
                         {self.qids[i]: self.qrels[self.qids[i]] for i in il}, self.gains[idx],
                         self.labels[idx], self.ideal[idx], self.qlen[idx],
                         {k: v[idx] for k, v in self.scores.items()},
                         None if self.q_emb is None else self.q_emb[idx])


def binary_level(split):
    return 2 if split.startswith("trec") else 1


def _rows(cache, qids):
    index = {q: i for i, q in enumerate(cache["qids"])}
    return cache["scores"][torch.tensor([index[q] for q in qids], dtype=torch.long)]


def load_split(cfg, protocol, split, scores=("sem", "ctx", "bi", "bm25"), q_emb=True, cutoff=10,
               _memo={}):
    """scores: which cached score matrices to load (ctx is protocol specific)."""
    splits = load_splits(cfg, protocol)
    qids = splits[split]
    pset = physical_set(protocol, split)
    key = ("meta", cfg["profile"], pset)
    if key not in _memo:
        _memo[key] = (load_cands(cfg, pset), load_qrels(cfg), load_queries(cfg))
    cands, qrels_all, queries = _memo[key]
    pids = [cands[q] for q in qids]
    K = len(pids[0])
    assert all(len(p) == K for p in pids), "candidate lists must have identical length"
    qrels = {q: qrels_all[q] for q in qids}
    gains = torch.tensor([[qrels[q].get(p, 0) for p in pl] for q, pl in zip(qids, pids)], dtype=torch.float)
    gains = gains.clamp_min(0)
    labels = (gains >= binary_level(split)).float()
    d = SplitData(split, qids, pids, qrels, gains, labels,
                  ideal_gains([qrels[q] for q in qids], cutoff),
                  torch.tensor([len(queries[q].split()) for q in qids], dtype=torch.float))
    for s in scores:
        proto = protocol if s == "ctx" else "A"   # only ctx depends on the (protocol-specific) GAT
        c = load_scores(cache_path(cfg, f"{s}_{pset}.pt", proto))
        rows = _rows(c, qids)
        assert rows.shape == (len(qids), K), f"{s}_{pset} misaligned"
        d.scores[s] = rows
    if q_emb:
        e = _memo.get(("q_emb", cfg["profile"]))
        if e is None:
            e = _memo[("q_emb", cfg["profile"])] = load_emb(cache_path(cfg, "q_emb.pt"))
        d.q_emb = e["emb"][torch.tensor([e["index"][q] for q in qids], dtype=torch.long)]
    return d


def gather_passage_emb(cfg, pids, p_emb=None, dtype=torch.float16):
    """pids: [N][K] -> [N, K, d] tensor."""
    if p_emb is None:
        p_emb = load_emb(cache_path(cfg, "p_emb.pt"), cast=False)
    idx = torch.tensor([[p_emb["index"][p] for p in pl] for pl in pids], dtype=torch.long)
    return p_emb["emb"][idx].to(dtype)
