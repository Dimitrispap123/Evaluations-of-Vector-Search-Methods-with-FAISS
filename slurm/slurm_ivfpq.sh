#!/bin/bash
#SBATCH --job-name=faiss_ivfpq
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ivfpq_%j.out
#SBATCH --error=logs/ivfpq_%j.err

# ── IVFPQ on Aristotelis: batch partition (Intel Xeon, 20 cores/node) ───────
# 16 cores leaves headroom on the node; k-means training + search both scale.

echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Partition : $SLURM_JOB_PARTITION"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "Started   : $(date)"

module load gcc/14.2.0 python/3.13.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
cd "$SCRIPT_DIR"

mkdir -p logs results

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python run_experiment.py \
    --method ivfpq \
    --profile full \
    --data_root ./data \
    --results_dir ./results

echo "Finished  : $(date)"
