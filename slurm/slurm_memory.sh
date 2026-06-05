#!/bin/bash
#SBATCH --job-name=faiss_memory
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/memory_%j.out
#SBATCH --error=logs/memory_%j.err

# ── measure_memory.py on Aristotelis: batch partition (Intel Xeon) ──────────
# Builds each (method, dataset, structural-param) index ONCE and records the
# serialised byte size. No search, no sweep — fast. 8 cores is plenty; the
# bottleneck is k-means training for IVFPQ on GIST, which uses few cores.

echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Partition : $SLURM_JOB_PARTITION"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "Started   : $(date)"

module load gcc/14.2.0 python/3.13.0

SCRIPT_DIR="$HOME/project"
source "$SCRIPT_DIR/venv/bin/activate"
cd "$SCRIPT_DIR"

mkdir -p logs results

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python measure_memory.py \
    --data_root ./data \
    --results_dir ./results

echo "Finished  : $(date)"