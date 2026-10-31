#!/bin/bash
# Full study in ONE job (QOS "normal": 1 running job / 3 submitted per user, 1 GPU, 32 CPU, 160G, 24h).
# Stages: download -> data -> cross-encoder -> bi-encoder -> GAT A/B + s_ctx -> tests -> fusion A/B -> GAT controls.
# Resumable: finished stages are skipped, so after a timeout/failure just `sbatch slurm/run_pipeline.sbatch` again.
#SBATCH --job-name=regat_pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
source slurm/env.sh
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
$PY -m src.pipeline --workers ${SLURM_CPUS_PER_TASK:-16} "$@"
