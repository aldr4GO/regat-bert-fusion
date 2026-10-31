"""Stage 1: frozen cross-encoder raw logits s_sem for every (query, candidate) pair.

Usage:  python -m src.cache_semantic [--sets gat_train val ...]
Writes: cache/sem_{set}.pt  ({'qids', 'scores' [N, K]}, aligned with cache/cands_{set}.json)
        results/sem_gap_stats.json  (train vs dev over-confidence check, Sec. 3.1)
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import cache_path, device, get_logger, load_config, path, save_json, save_scores, set_env
from src.data import load_cands, load_queries, read_passages

log = get_logger("cache_semantic")


class PairCollate:
    def __init__(self, tok, max_length):
        self.tok, self.max_length = tok, max_length

    def __call__(self, batch):
        idx = torch.tensor([b[0] for b in batch])
        enc = self.tok([b[1] for b in batch], [b[2] for b in batch], truncation="longest_first",
                       max_length=self.max_length, padding=True, return_tensors="pt")
        return idx, enc


def load_cross_encoder(cfg, dev):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = cfg["data"]["cross_encoder"]["model"]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(dev).eval()
    if cfg["data"]["cross_encoder"]["fp16"] and dev.type == "cuda":
        model = model.half()
    model.requires_grad_(False)
    return tok, model


@torch.no_grad()
def score_pairs(tok, model, pairs, cfg, dev, num_workers=4):
    """pairs: list of (q_text, p_text). Returns raw logits (no sigmoid) in input order."""
    cc = cfg["data"]["cross_encoder"]
    # length-sorted batching removes most padding; results are scattered back to input order
    order = np.argsort([-(len(q) + len(p)) for q, p in pairs], kind="stable")
    items = [(int(i), pairs[i][0], pairs[i][1]) for i in order]
    dl = DataLoader(items, batch_size=cc["batch_size"], shuffle=False, num_workers=num_workers,
                    collate_fn=PairCollate(tok, cc["max_length"]))
    out = torch.empty(len(pairs))
    for idx, enc in tqdm(dl, desc="cross-encoder", mininterval=60):
        enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
        out[idx] = model(**enc).logits[:, 0].float().cpu()
    return out


def verify_against_sentence_transformers(cfg, pairs, ours, dev):
    """VERIFY: our logits == sentence_transformers.CrossEncoder.predict(activation=identity)."""
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(cfg["data"]["cross_encoder"]["model"], max_length=cfg["data"]["cross_encoder"]["max_length"],
                      device=str(dev))
    try:
        ref = ce.predict(pairs, activation_fn=torch.nn.Identity(), convert_to_tensor=True).float().cpu()
    except TypeError:  # older sentence-transformers
        ref = torch.tensor(ce.predict(pairs, activation_fct=torch.nn.Identity()))
    diff = (ref - ours).abs().max().item()
    log.info("VERIFY cross-encoder logits vs sentence-transformers: max |diff| = %.4g", diff)
    assert diff < 0.05, f"cross-encoder logits disagree with sentence-transformers ({diff})"
    return diff


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config()
    set_env(cfg)
    dev = device()
    sets = args.sets or cfg["data"]["cache_sets"]
    queries = load_queries(cfg)
    cands = {s: load_cands(cfg, s) for s in sets}
    all_pids = {p for c in cands.values() for pl in c.values() for p in pl}
    log.info("loading %d passage texts", len(all_pids))
    passages = read_passages(cfg, all_pids)
    tok, model = load_cross_encoder(cfg, dev)
    log.info("cross-encoder %s on %s", cfg["data"]["cross_encoder"]["model"], dev)

    gaps = {}
    verified = False
    for s in sets:
        qids = list(cands[s])
        pairs = [(queries[q], passages[p]) for q in qids for p in cands[s][q]]
        K = len(cands[s][qids[0]])
        scores = score_pairs(tok, model, pairs, cfg, dev, args.num_workers).view(len(qids), K)
        if not verified:
            verify_against_sentence_transformers(cfg, pairs[:64], scores.view(-1)[:64], dev)
            verified = True
        save_scores(qids, scores, cache_path(cfg, f"sem_{s}.pt"))
        top2 = scores.topk(2, dim=1).values
        gaps[s] = {"mean_top1_minus_top2": float((top2[:, 0] - top2[:, 1]).mean()),
                   "mean_logit": float(scores.mean()), "mean_max_logit": float(top2[:, 0].mean())}
        log.info("sem_%s: %s pairs, gap stats %s", s, len(pairs), gaps[s])
    p = path(cfg, "results_dir", "sem_gap_stats.json")
    if p.exists():
        from src.common import load_json
        gaps = {**load_json(p), **gaps}
    save_json(gaps, p)


if __name__ == "__main__":
    main()
