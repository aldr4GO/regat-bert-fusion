#!/bin/bash
# Sec. 6.6 graph controls (no_graph, random_graph, k_nn x min-cos sweep), evaluated on val.
# One array task per control (11 total, see configs/gat.yaml). Independent of Stage 5.
#SBATCH --job-name=regat_gat_controls
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-10%2
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
source slurm/env.sh
NAME=$($PY -c "import yaml;print(yaml.safe_load(open('configs/gat.yaml'))['controls'][${SLURM_ARRAY_TASK_ID:-0}]['name'])")
echo "control: $NAME"
$PY -m src.train_gat --protocol A --control "$NAME"
$PY -m src.train_gat --protocol A --summarize
