#!/bin/bash
# Sec. 6.6 graph controls (no_graph, random_graph, k_nn x min-cos sweep) evaluated on val, one after another.
# Already-finished controls are skipped. Run a subset with: sbatch slurm/04_gat_controls.sbatch random_graph no_graph
#SBATCH --job-name=regat_gat_controls
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
source slurm/env.sh
if [ $# -gt 0 ]; then
  for c in "$@"; do $PY -m src.train_gat --protocol A --control "$c"; done
  $PY -m src.train_gat --protocol A --summarize
else
  $PY -m src.pipeline --only gat_controls
fi
