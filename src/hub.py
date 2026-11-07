"""Push / pull the frozen GAT checkpoints to / from the Hugging Face Hub (configs/hub.yaml).

Only the GATs listed in configs/hub.yaml are stored; datasets come from their public sources (src.download,
src.datasets.sources) and all caches are rebuilt by the pipeline. A pulled GAT lands at its usual cache path,
so the pipeline's GAT stages see it as done and never retrain it.

Usage:  python -m src.hub status                       # which listed GATs exist locally
        python -m src.hub push --repo <user>/<name>    # upload them + a generated model card (needs `hf auth login`)
        python -m src.hub pull --repo <user>/<name>    # download them into cache/ (existing files are kept)
"""
import argparse
import os
import tempfile
from pathlib import Path

import yaml

from src.common import CONFIG_DIR, cache_path, get_logger, load_config, load_json, results_path

log = get_logger("hub")
NAME = "gat_best.pt"


def hub_cfg(repo=None):
    with open(CONFIG_DIR / "hub.yaml") as f:
        hc = yaml.safe_load(f)
    hc["repo_id"] = repo or os.environ.get("REGAT_HF_REPO") or hc["repo_id"]
    return hc


def entries(cfg, hc):
    """(experiment key, path in the hub repo, local path) of every listed GAT."""
    return [(k, f"{k}/{NAME}", cache_path(cfg, NAME, k)) for k in hc["models"]]


def model_card(cfg, hc, items):
    rows = []
    for k, remote, _ in items:
        p = results_path(cfg, k, "gat_train_summary.json")
        s = load_json(p) if p.exists() else {}
        f = lambda x: f"{s[x]:.4f}" if x in s else "-"
        rows.append(f"| `{k}` | `{remote}` | {f('val_ndcg@10')} | {f('val_mrr@10')} | {f('val_ndcg@10_s_bi')} "
                    f"| {s.get('best_epoch', '-')} |")
    return "\n".join([
        "---", "library_name: pytorch", "tags: [graph-attention-network, reranking, information-retrieval]", "---",
        "", "# ReGAT-BERT frozen GATs", "",
        "Frozen graph attention networks of the ReGAT-BERT score-fusion study (code and results: see the GitHub "
        "repository). Each GAT re-scores the BM25 top-K candidates of a query over a kNN graph of bi-encoder "
        "embeddings and produces `s_ctx`, which is fused with the cross-encoder score `s_sem`.", "",
        "| experiment key | file | val nDCG@10 | val MRR@10 | val nDCG@10 (s_bi) | best epoch |",
        "|---|---|---|---|---|---|", *rows, "",
        "`A`/`B`: MS MARCO protocols A and B; the others are the multi-hop datasets (MULTIHOP_EXTENSION_PLAN.md). "
        "Val metrics are on the validation split only (test is never used for decisions).", "",
        "## Usage", "", "```bash", f"python -m src.hub pull --repo {hc['repo_id']}   # -> cache/.../gat_best.pt",
        "```", "", "Each file is a `torch.save` dict `{state_dict, graph, model, d}` loaded by "
        "`src/cache_contextual.py` (`torch.load(..., weights_only=False)`).", ""])


def push(cfg, hc, private):
    from huggingface_hub import CommitOperationAdd, HfApi
    items = entries(cfg, hc)
    missing = [str(p) for _, _, p in items if not p.exists()]
    assert not missing, f"missing checkpoints: {missing}"
    api = HfApi()
    api.create_repo(hc["repo_id"], repo_type=hc["repo_type"], private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        card = Path(tmp) / "README.md"
        card.write_text(model_card(cfg, hc, items))
        ops = [CommitOperationAdd(remote, str(local)) for _, remote, local in items]
        ops.append(CommitOperationAdd("README.md", str(card)))
        api.create_commit(hc["repo_id"], ops, repo_type=hc["repo_type"],
                          commit_message=f"Upload {len(items)} frozen GATs")
    log.info("pushed %d GATs to https://huggingface.co/%s", len(items), hc["repo_id"])


def pull(cfg, hc):
    from huggingface_hub import hf_hub_download
    for k, remote, local in entries(cfg, hc):
        if local.exists():
            log.info("%s: %s already present, kept", k, local)
            continue
        with tempfile.TemporaryDirectory(dir=local.parent) as tmp:   # same filesystem, so the move is atomic
            os.replace(hf_hub_download(hc["repo_id"], remote, repo_type=hc["repo_type"], local_dir=tmp), local)
        log.info("%s: -> %s", k, local)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["status", "push", "pull"])
    ap.add_argument("--repo", default=None, help="hub repo id (default: REGAT_HF_REPO or configs/hub.yaml)")
    ap.add_argument("--private", action="store_true", help="push: create the repo as private")
    args = ap.parse_args()
    cfg = load_config()
    hc = hub_cfg(args.repo)
    if args.action == "status":
        for k, remote, local in entries(cfg, hc):
            print(f"{k:16s} {'ok     ' if local.exists() else 'MISSING'} {local} -> {remote}")
        return
    assert hc["repo_id"], "no hub repo: pass --repo, set REGAT_HF_REPO or fill repo_id in configs/hub.yaml"
    push(cfg, hc, args.private) if args.action == "push" else pull(cfg, hc)


if __name__ == "__main__":
    main()
