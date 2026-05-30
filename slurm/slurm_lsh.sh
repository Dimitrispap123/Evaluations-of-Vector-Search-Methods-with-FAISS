#!/bin/bash
#SBATCH --job-name=faiss_lsh
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/lsh_%j.out
#SBATCH --error=logs/lsh_%j.err

# ── LSH on Aristotelis: batch partition ─────────────────────────────────────
# LSH is the cheapest of the three. 8 cores is plenty.

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
    --method lsh \
    --profile full \
    --data_root ./data \
    --results_dir ./results

echo "Finished  : $(date)"
