# ReGAT-BERT score-fusion ablations

ReGAT-BERT fuses a cross-encoder score `s_sem` with a GAT-contextualised score `s_ctx`:
`final = alpha * s_sem + (1 - alpha) * s_ctx`. This repo studies only that fusion step: 4 score normalisers ×
4 ways for `alpha` to depend on the query and the scores. It runs on MS MARCO (protocols A and B) and is
extended to multi-hop retrieval (MuSiQue, HotpotQA, BEIR HotpotQA, 2Wiki) to test whether multi-hop queries
need more context. The paper is in [paper/main.pdf](paper/main.pdf) and the slides in
[presentation/slides.pdf](presentation/slides.pdf).

## What is stored where

| What | Where | How to get it |
|---|---|---|
| Code, configs, tests, SLURM scripts | this repo (`src/`, `configs/`, `tests/`, `slurm/`) | |
| Results: tables, figures, per-run metrics, gates | this repo (`results/`) | |
| Job logs, paper, slides, plans, `NOTES.md` | this repo | |
| **Frozen GATs** (`A`, `B`, `musique`, `hotpotqa`, `beir_hotpotqa`, `twowiki`) | Hugging Face Hub model repo | `python -m src.hub pull --repo <user>/<repo>` |
| MS MARCO passage, TREC DL 19/20, BEIR HotpotQA | public, via `ir_datasets` | `python -m src.download` / pipeline `*:download` stages |
| MuSiQue, HotpotQA (distractor), 2Wiki | public HF dataset repos (`configs/multihop.yaml`) | `python -m src.datasets.sources --dataset <ds>` |
| Pyserini BM25 indexes | Pyserini prebuilt indexes | downloaded on first use |
| Cross-encoder / bi-encoder | public HF models (`configs/data.yaml`), never fine-tuned | downloaded on first use |
| Splits, candidates, `s_sem`/`s_bi`/`s_ctx` caches, embeddings | rebuilt locally (`data/`, `cache/`) | `python -m src.pipeline` |

Only the GATs go to the Hub. They are trained once and frozen (never retrained after Stage 3), and retraining
is not bit-exact. Everything else is either public or deterministically rebuilt by the pipeline from the
fixed seeds, so it is not stored anywhere. The H4 graph-type GATs (`<ds>@none|random|title_link|hybrid`) are
ablation controls and are not uploaded by default; add them to `configs/hub.yaml` to keep them too.

## Setup

```bash
git clone <this repo> Graph_Project && cd Graph_Project
bash slurm/setup_env.sh                                # ~/.conda/envs/regat (Python 3.11 + OpenJDK 21)
source slurm/env.sh
$PY -m src.hub pull --repo <user>/<repo>               # frozen GATs -> cache/**/gat_best.pt
```

## Reproducing

The committed `results/` already holds every number in the paper. The pipeline's done-markers live in
`results/`, so on a fresh clone it would consider most stages finished. To recompute from scratch into a
separate folder, using the pulled GATs:

```bash
export REGAT_RESULTS_DIR=results_repro
sbatch slurm/run_pipeline.sh          # MS MARCO: download -> BM25 -> CE / bi-encoder caches -> s_ctx -> fusion
sbatch slurm/run_multihop.sh          # multi-hop extension (after the first job has finished)
```

The `gat_*` / `<ds>:gat` stages are skipped because the pulled `gat_best.pt` files exist, so the frozen
models are reused rather than retrained. The H4 graph-type GATs and the graph controls are not on the Hub, so
those stages train them. In `results_repro` the golden-regression gate compares against the MS MARCO runs of
the first job. The primary normaliser stays the one fixed in the committed `results/primary_norm.json`.
Compare `results_repro/` with `results/` to check the reproduction. See [RUNNING.md](RUNNING.md) for stages, timings, cluster limits
and the smoke tests, [CLAUDE.md](CLAUDE.md) for conventions, and [NOTES.md](NOTES.md) for deviations and
verification outcomes.

## Publishing the GATs (maintainers)

```bash
hf auth login                                          # a write token
$PY -m src.hub status                                  # all listed GATs present locally?
$PY -m src.hub push --repo <user>/<repo> [--private]   # uploads them + a generated model card
```
