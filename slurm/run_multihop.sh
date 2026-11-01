#!/bin/bash
# Multi-hop extension (MULTIHOP_EXTENSION_PLAN.md) in ONE resumable job:
#   golden regression A/B (+ new metrics) -> Step 0 diagnostics -> MuSiQue, HotpotQA (E1 + oracle) -> H4 graph
#   types + k_nn sweep -> BEIR HotpotQA (E2) -> tests -> transfer (E4) -> 2Wiki (optional) -> hypotheses (E3).
# Finished stages are skipped, so after a timeout or failure just submit it again.
# QOS "normal": 1 running / 3 submitted jobs per user, 1 GPU, 32 CPU, 160G, 24h.
#SBATCH --job-name=regat_multihop
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
source slurm/env.sh
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
$PY -m src.pipeline --part multihop --workers ${SLURM_CPUS_PER_TASK:-16} "$@"
