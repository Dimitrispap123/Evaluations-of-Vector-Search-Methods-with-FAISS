#!/bin/bash
#SBATCH --job-name=faiss_hnsw
#SBATCH --output=logs/hnsw_%j.out
#SBATCH --error=logs/hnsw_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G

# ── HNSW is CPU-only in FAISS — use a high-core-count CPU queue ─────────────

# --- rome queue (128 cores, best for HNSW OpenMP) ---
##SBATCH --partition=rome
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=64

# --- batch queue (fallback) ---
##SBATCH --partition=batch
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=20

# ── Environment setup ────────────────────────────────────────────────────────

echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "CPU cores    : $SLURM_CPUS_PER_TASK"
echo "Started      : $(date)"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"

# ── Run ───────────────────────────────────────────────────────────────────────

mkdir -p "$SCRIPT_DIR/logs"
cd "$SCRIPT_DIR"

# HNSW builds benefit a lot from many cores (OpenMP)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

python run_experiment.py \
    --method hnsw \
    --profile full \
    --data_root ./data \
    --results_dir ./results

echo "Finished : $(date)"
