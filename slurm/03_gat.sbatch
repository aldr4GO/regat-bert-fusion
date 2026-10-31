#!/bin/bash
# Stages 3-4: train the GAT once per protocol, freeze it, cache s_ctx, run the sanity gates. Needs 1 GPU.
#SBATCH --job-name=regat_s34_gat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
source slurm/env.sh
for P in A B; do
  CK=$($PY -c "from src.common import load_config, cache_path; print(cache_path(load_config(), 'gat_best.pt', '$P'))")
  if [ -f "$CK" ]; then
    echo "GAT for protocol $P already trained and frozen ($CK); skipping training"
  else
    $PY -m src.train_gat --protocol $P
  fi
  $PY -m src.cache_contextual --protocol $P
done
