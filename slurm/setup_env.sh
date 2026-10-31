#!/bin/bash
# Creates the conda env used by every job. Run once:  bash slurm/setup_env.sh
# (or: sbatch slurm/setup_env.sh). Safe to re-run.
#SBATCH --job-name=regat_setup
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
ENV_PREFIX=${ENV_PREFIX:-$HOME/.conda/envs/regat}
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  conda create -y -p "$ENV_PREFIX" -c conda-forge python=3.11 openjdk=21
fi
PIP="$ENV_PREFIX/bin/python -m pip"
$PIP install --upgrade pip
if [ -f requirements.txt ]; then
  $PIP install -r requirements.txt
else
  $PIP install -r requirements.in
fi
"$ENV_PREFIX/bin/python" -c "import torch, torch_geometric, pyserini, ir_datasets, ir_measures; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"$ENV_PREFIX/bin/java" -version
