#!/bin/bash
#SBATCH --job-name=faiss_ivfpq
#SBATCH --output=logs/ivfpq_%j.out
#SBATCH --error=logs/ivfpq_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G

# ── TODO: uncomment ONE of the queue blocks below based on your allocation ──

# --- GPU queue (gpu / ampere) — fastest for IVFPQ ---
##SBATCH --partition=gpu
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=8

# --- CPU-only queue (rome / batch) ---
##SBATCH --partition=rome
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=32

# ── Environment setup ────────────────────────────────────────────────────────

echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Started      : $(date)"

# Activate venv — adjust path if your venv is elsewhere
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"

# If on a GPU node, install faiss-gpu if not already present
if command -v nvidia-smi &>/dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    pip install faiss-gpu --quiet 2>/dev/null || true
fi

# ── Run ───────────────────────────────────────────────────────────────────────

mkdir -p "$SCRIPT_DIR/logs"
cd "$SCRIPT_DIR"

python run_experiment.py \
    --method ivfpq \
    --profile full \
    --data_root ./data \
    --results_dir ./results

echo "Finished : $(date)"
