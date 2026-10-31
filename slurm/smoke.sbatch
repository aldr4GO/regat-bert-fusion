#!/bin/bash
# Tiny end-to-end run of the whole pipeline (REGAT_PROFILE=smoke -> data/smoke, cache_smoke, results_smoke).
# Use it to verify the environment on the GPU before the full run (~15-30 min, mostly downloads on first use).
#SBATCH --job-name=regat_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
export REGAT_PROFILE=smoke
source slurm/env.sh
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
$PY -m src.pipeline --fresh --workers 4
echo "SMOKE TEST PASSED"
