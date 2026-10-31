#!/bin/bash
# Stages 5-6: unit tests, oracle + fixed grid, 16-config grid x 5 seeds (+ paper / loss / lambda ablations),
# tables, significance, plots — for both protocols. CPU only.
#SBATCH --job-name=regat_s56_fusion
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
source slurm/env.sh
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
$PY -m pytest -q tests
for P in A B; do
  $PY -m src.run_all --protocol $P --workers ${SLURM_CPUS_PER_TASK:-16}
done
