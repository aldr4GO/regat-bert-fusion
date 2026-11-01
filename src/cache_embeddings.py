"""Stage 2: frozen bi-encoder embeddings (L2-normalised) and the s_bi baseline score.

Usage:  python -m src.cache_embeddings
        python -m src.cache_embeddings --protocol musique    # multi-hop dataset
Writes: cache/q_emb.pt, cache/p_emb.pt  ({'ids', 'emb' [N, d]})
        cache/bi_{set}.pt               (cosine q·p per candidate, aligned with cands)
        results/stage12_check.json      (acceptance: CE beats BM25 and s_bi on val)
"""
import argparse

import torch

from src.common import (cache_path, device, get_logger, load_config, load_emb, path, save_emb, save_json,
                        save_scores, set_env)
from src.data import load_cands, load_queries, read_passages

log = get_logger("cache_embeddings")


def load_bi_encoder(cfg, dev):
    from sentence_transformers import SentenceTransformer
    bc = cfg["data"]["bi_encoder"]
    try:
        model = SentenceTransformer(bc["model"], device=str(dev))
        name = bc["model"]
    except Exception as e:
        log.warning("VERIFY: %s failed to load (%r); using fallback %s", bc["model"], e, bc["fallback"])
        model = SentenceTransformer(bc["fallback"], device=str(dev))
        name = bc["fallback"]
    model.max_seq_length = bc["max_length"]
    if bc["fp16"] and dev.type == "cuda":
        model = model.half()
    model.eval()
    model.requires_grad_(False)
    return model, name


@torch.no_grad()
def encode(model, texts, cfg, chunk=500_000):
    out = []
    for i in range(0, len(texts), chunk):
        e = model.encode(texts[i:i + chunk], batch_size=cfg["data"]["bi_encoder"]["batch_size"],
                         convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        out.append(e.float().cpu())
        log.info("encoded %d / %d", min(i + chunk, len(texts)), len(texts))
    return torch.cat(out)


def stage12_check(cfg):
    from src.store import load_split
    from src.metrics import ndcg_at_k, rr_at_k
    d = load_split(cfg, "A", "val", scores=("sem", "bi", "bm25"), q_emb=False)
    res = {}
    for s in ("bm25", "bi", "sem"):
        res[s] = {"ndcg@10": float(ndcg_at_k(d.scores[s], d.gains, d.ideal).mean()),
                  "mrr@10": float(rr_at_k(d.scores[s], d.labels).mean())}
    res["accept"] = bool(res["sem"]["ndcg@10"] > max(res["bm25"]["ndcg@10"], res["bi"]["ndcg@10"]))
    save_json(res, path(cfg, "results_dir", "stage12_check.json"))
    log.info("Stage 1-2 acceptance on val: %s", res)
    if not res["accept"]:
        raise SystemExit("ACCEPTANCE FAILED: cross-encoder does not beat BM25 and s_bi on val. "
                         "Check model id and max_length.")
    return res


def stage12_check_dataset(cfg, key):
    """Multi-hop gate (plan Sec. 3.4): CE-only beats BM25 (in-pool BM25 for closed pools) on val."""
    from src.common import results_path
    from src.metrics import ndcg_at_k, rr_at_k
    from src.store import load_split
    d = load_split(cfg, key, "val", scores=("sem", "bi", "bm25"), q_emb=False)
    res = {s: {"ndcg@10": float(ndcg_at_k(d.scores[s], d.gains, d.ideal, mask=d.mask).mean()),
               "mrr@10": float(rr_at_k(d.scores[s], d.labels, mask=d.mask).mean())} for s in ("bm25", "bi", "sem")}
    res["accept"] = bool(res["sem"]["ndcg@10"] > res["bm25"]["ndcg@10"])
    save_json(res, results_path(cfg, key, "stage12_check.json"))
    log.info("Stage 1-2 acceptance on %s val: %s", key, res)
    if not res["accept"]:
        raise SystemExit(f"ACCEPTANCE FAILED ({key}): cross-encoder does not beat BM25 on val.")
    return res


def run_dataset(cfg, key, dev):
    from src.common import data_cache_path
    from src.datasets.common import save_padded
    from src.datasets.texts import cands as load_c, passage_texts, queries as load_q
    sets = ["gat_train", "fusion_train", "val", "test"]
    queries = load_q(cfg, key)
    cands = {s: load_c(cfg, key, s) for s in sets}
    qids = sorted({q for c in cands.values() for q in c})
    pids = sorted({p for c in cands.values() for pl in c.values() for p in pl})
    model, name = load_bi_encoder(cfg, dev)
    log.info("bi-encoder %s on %s (%s): %d queries, %d passages", name, dev, key, len(qids), len(pids))
    passages = passage_texts(cfg, key, pids)
    q_emb = encode(model, [queries[q] for q in qids], cfg)
    save_emb(qids, q_emb, data_cache_path(cfg, "q_emb.pt", key), fp16=False)
    p_emb = encode(model, [passages[p] for p in pids], cfg)
    del passages
    save_emb(pids, p_emb, data_cache_path(cfg, "p_emb.pt", key), fp16=cfg["data"]["bi_encoder"]["store_fp16"])
    pe = load_emb(data_cache_path(cfg, "p_emb.pt", key))
    qi = {q: i for i, q in enumerate(qids)}
    for s in sets:
        rows = []
        for q in cands[s]:
            P = pe["emb"][torch.tensor([pe["index"][p] for p in cands[s][q]])]
            rows.append((P @ q_emb[qi[q]]).tolist())
        save_padded(cfg, key, f"bi_{s}.pt", list(cands[s]), rows)
    stage12_check_dataset(cfg, key)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check_only", action="store_true")
    ap.add_argument("--protocol", default="A", help="A (MS MARCO) or a multi-hop dataset")
    args = ap.parse_args()
    cfg = load_config()
    set_env(cfg)
    if args.protocol != "A":
        if args.check_only:
            stage12_check_dataset(cfg, args.protocol)
        else:
            run_dataset(cfg, args.protocol, device())
        return
    if args.check_only:
        stage12_check(cfg)
        return
    dev = device()
    sets = cfg["data"]["cache_sets"]
    queries = load_queries(cfg)
    cands = {s: load_cands(cfg, s) for s in sets}
    qids = sorted({q for c in cands.values() for q in c})
    pids = sorted({p for c in cands.values() for pl in c.values() for p in pl})
    model, name = load_bi_encoder(cfg, dev)
    log.info("bi-encoder %s on %s: %d queries, %d passages", name, dev, len(qids), len(pids))
    passages = read_passages(cfg, pids)

    fp16 = cfg["data"]["bi_encoder"]["store_fp16"]
    q_emb = encode(model, [queries[q] for q in qids], cfg)
    save_emb(qids, q_emb, cache_path(cfg, "q_emb.pt"), fp16=False)
    p_emb = encode(model, [passages[p] for p in pids], cfg)
    del passages
    save_emb(pids, p_emb, cache_path(cfg, "p_emb.pt"), fp16=fp16)
    save_json({"bi_encoder": name, "dim": q_emb.shape[1]}, path(cfg, "cache_dir", "bi_encoder.json"))

    # s_bi computed from the stored (possibly fp16-rounded) embeddings so it matches what the GAT sees
    pe = load_emb(cache_path(cfg, "p_emb.pt"))
    qi = {q: i for i, q in enumerate(qids)}
    for s in sets:
        sq = list(cands[s])
        P = pe["emb"][torch.tensor([[pe["index"][p] for p in cands[s][q]] for q in sq])]
        Q = q_emb[torch.tensor([qi[q] for q in sq])]
        save_scores(sq, torch.einsum("nkd,nd->nk", P, Q), cache_path(cfg, f"bi_{s}.pt"))
    stage12_check(cfg)


if __name__ == "__main__":
    main()
