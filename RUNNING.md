# Running the pipeline on SLURM

Submit everything from the **repo root** (`~/Graph_Project`); jobs `cd` into `$SLURM_SUBMIT_DIR`.
Logs are written to `logs/<job-name>_<jobid>.out`.

**Cluster limits (QOS `normal`):** at most 3 submitted jobs and 1 running job per user, 1 GPU, 32 CPUs,
160 GB and 24 h per job. So the whole study runs as **one resumable job**, not a chain of jobs.

## 0. One-time setup (already done on this machine)

```bash
cd ~/Graph_Project
bash slurm/setup_env.sh      # ~/.conda/envs/regat: Python 3.11 + OpenJDK 21 + requirements.txt
```

`slurm/setup_env.sh` can also be submitted with `sbatch`, and re-running it is safe. If the env lives
somewhere else, `export ENV_PREFIX=/path/to/env` before running it and before every `sbatch`.

The MS MARCO files (`data/ir_datasets/`, md5-checked) and the Pyserini BM25 index (`data/pyserini/`)
are already downloaded, and so are both HF models (`~/.cache/huggingface`). The jobs only re-check them.

## 1. Optional: smoke test on a GPU (~15 min)

```bash
sbatch slurm/smoke.sbatch
tail -f logs/regat_smoke_<jobid>.out       # the last line is "SMOKE TEST PASSED"
```

This runs every stage on a tiny subset (200 train / 200 dev queries) into separate folders
(`data/smoke/`, `cache_smoke/`, `results_smoke/`), so it never touches the real outputs. I already ran it
on CPU; running it on the GPU checks the CUDA path before the long job.

## 2. Full run: one job

```bash
sbatch slurm/run_pipeline.sbatch
squeue -u $USER                              # PD = waiting for a GPU, R = running
tail -f logs/regat_pipeline_<jobid>.out
```

The job runs these stages in order. A stage whose outputs already exist is skipped.

| # | Stage | What it does | Device | Rough time |
|---|---|---|---|---|
| 1 | `download` | verify the MS MARCO files and the Pyserini index (no-op if present) | CPU | < 1 min |
| 2 | `data` | Stage 0: BM25 top-100 for 75k train + dev/small + TREC DL, splits A and B | CPU (32 threads) | 15-30 min |
| 3 | `cross_encoder` | Stage 1: `s_sem` raw logits for ~4.9M pairs | GPU | 20-40 min |
| 4 | `bi_encoder` | Stage 2: embeddings, `s_bi`, **acceptance check** (CE > BM25 and `s_bi` on val) | GPU | 10-20 min |
| 5 | `gat_A`, `ctx_A`, `gat_B`, `ctx_B` | Stages 3-4: train + freeze each GAT, cache `s_ctx`, **sanity gates** | GPU | 20-60 min |
| 6 | `tests` | `pytest tests` against the real caches | CPU | < 1 min |
| 7 | `fusion_A`, `fusion_B` | Stages 5-6: oracle, fixed grid, 16 configs × 5 seeds, paper row, λ=0 and BCE ablations, tables, plots | CPU (32 workers) | 15-40 min |
| 8 | `gat_controls` | Sec. 6.6: 11 graph controls on val (no_graph, random_graph, k_nn × min-cos) | GPU | 1-3 h |

Actual times are written to `results/pipeline_timing.json`. In total this should take a few hours, well
under the 24 h limit.

**If the job stops** (timeout, node issue, failed gate), fix the cause if needed and simply resubmit
`sbatch slurm/run_pipeline.sbatch`; it continues from the first unfinished stage. A failed acceptance
check or sanity gate stops the job with a message saying what to check (see `FUSION_ABLATION_PLAN.md`
Sec. 10). Fusion runs that already have a result file are never recomputed.

The GAT is trained only once per protocol: `train_gat` refuses to overwrite `cache/gat_best.pt` /
`cache/B/gat_best.pt`. To retrain anyway, delete the file explicitly. Deleting it also requires deleting
`results/{A,B}/stage4_sanity.json` and the fusion results (`results/{A,B}/`), since they depend on it.

## 3. Running single stages

```bash
sbatch slurm/run_pipeline.sbatch --only fusion_A fusion_B     # rerun named stages (ignores done-markers)
sbatch slurm/04_gat_controls.sbatch                           # all graph controls (skips finished ones)
sbatch slurm/04_gat_controls.sbatch random_graph no_graph     # only these controls
```

Per-stage scripts are also available (`slurm/01_data.sbatch`, `02_cache_encoders.sbatch`, `03_gat.sbatch`,
`05_fusion.sbatch`). With the 3-submission limit, chain at most three of them:
`sbatch --dependency=afterok:<jobid> slurm/05_fusion.sbatch`.

Interactive use (inside `srun --gres=gpu:1 -c 8 --mem=64G --pty bash`):

```bash
source slurm/env.sh
$PY -m src.train_fusion --protocol A --norm zscore --dep QS --seed 0      # a single fusion run
$PY -m src.oracle --protocol A                                            # alpha0 grid + oracle
$PY -m src.evaluate --protocol A && $PY -m src.analysis --protocol A      # tables + figures
$PY -m pytest -q tests                                                    # unit tests
REGAT_PROFILE=smoke $PY -m src.pipeline --fresh --workers 4               # the smoke test, by hand
```

## 4. Where the results are

- `results/{A,B}/main_table.csv`: reference rows (BM25, SBERT, CE alone, GAT-only), the 16 configs
  (`{norm}_fixedgrid` is the main fixed baseline, `{norm}_fixed_listwise` the learned scalar), the
  `paper` row, λ=0 and BCE ablations, and the oracle (upper bound, not a method). Values are MRR@10,
  NDCG@10 and MAP (mean and std over seeds) on test, TREC DL 2019 and 2020, plus the BM25 first-stage
  R@100 and R@1000.
- `results/{A,B}/significance.csv`: paired bootstrap (10k resamples) of each adaptive config against the best
  fixed config under the same normaliser.
- `results/{A,B}/oracle_gap.csv`, `oracle_summary.json`, `alpha0.json`: headroom analysis.
- `results/{A,B}/figures/`: heatmap, alpha histograms, alpha vs query length, alpha vs CE confidence, and the
  query-length buckets (PNG and PDF).
- `results/A/gat_controls.csv`: graph controls on val. `results/{A,B}/gat_train_summary.json` holds the GAT
  training curves.
- `results/{A,B}/summary.md`: auto summary (also copied into `NOTES.md`). `results/{A,B}/runs/` holds one
  json + npz per run (per-query metrics and alphas).
- `results/stage12_check.json`, `results/{A,B}/stage4_sanity.json`, `results/sem_gap_stats.json`,
  `results/bm25_first_stage.json`: acceptance checks and caveat statistics.
