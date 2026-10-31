#!/bin/bash
# Stages 1-2: frozen cross-encoder logits (s_sem) and bi-encoder embeddings (+ s_bi). Needs 1 GPU.
#SBATCH --job-name=regat_s12_cache
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
source slurm/env.sh
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
$PY -m src.cache_semantic --num_workers 8
$PY -m src.cache_embeddings          # ends with the Stage 1-2 acceptance check on val
