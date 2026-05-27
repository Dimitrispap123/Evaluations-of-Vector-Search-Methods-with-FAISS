#!/bin/bash
#SBATCH --job-name=faiss_lsh
#SBATCH --output=logs/lsh_%j.out
#SBATCH --error=logs/lsh_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G

# ── TODO: uncomment ONE of the queue blocks below ────────────────────────────

# --- GPU queue (gpu / ampere) ---
##SBATCH --partition=gpu
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=4

# --- CPU-only queue (batch / rome) ---
##SBATCH --partition=batch
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=8

# ── Environment setup ────────────────────────────────────────────────────────

echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Started      : $(date)"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"

if command -v nvidia-smi &>/dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    pip install faiss-gpu --quiet 2>/dev/null || true
fi

# ── Run ───────────────────────────────────────────────────────────────────────

mkdir -p "$SCRIPT_DIR/logs"
cd "$SCRIPT_DIR"

python run_experiment.py \
    --method lsh \
    --profile full \
    --data_root ./data \
    --results_dir ./results

echo "Finished : $(date)"
