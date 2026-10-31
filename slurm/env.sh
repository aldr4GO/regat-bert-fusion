# Sourced by every job script. Override ENV_PREFIX if your env lives elsewhere.
ENV_PREFIX=${ENV_PREFIX:-$HOME/.conda/envs/regat}
export PATH="$ENV_PREFIX/bin:$PATH"
export JAVA_HOME="$ENV_PREFIX/lib/jvm"          # conda-forge openjdk (needed by pyserini)
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PY="$ENV_PREFIX/bin/python"
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
mkdir -p logs
echo "[$(date)] job=${SLURM_JOB_ID:-local} host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-none} profile=${REGAT_PROFILE:-full}"
