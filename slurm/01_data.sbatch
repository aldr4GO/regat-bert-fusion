#!/bin/bash
# Stage 0: download MS MARCO via ir_datasets, BM25 top-100 (Pyserini prebuilt index), splits.
#SBATCH --job-name=regat_s0_data
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
source slurm/env.sh
$PY -m src.download          # parallel pre-fetch (md5-checked, skipped if present)
$PY -m src.data
$PY -m pytest -q tests/test_splits.py
