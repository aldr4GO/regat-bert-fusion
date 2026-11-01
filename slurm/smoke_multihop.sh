#!/bin/bash
# Tiny end-to-end run of EVERYTHING (MS MARCO + multi-hop extension) with REGAT_PROFILE=smoke
# (data/smoke, cache_smoke, results_smoke; the real outputs are never touched). ~30-45 min on a GPU.
#SBATCH --job-name=regat_smoke_multihop
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
export REGAT_PROFILE=smoke
source slurm/env.sh
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
$PY -m src.pipeline --fresh --workers ${SLURM_CPUS_PER_TASK:-8}
echo "SMOKE TEST PASSED"
