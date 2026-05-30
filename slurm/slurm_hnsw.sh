#!/bin/bash
#SBATCH --job-name=faiss_hnsw
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/hnsw_%j.out
#SBATCH --error=logs/hnsw_%j.err

# ── HNSW on Aristotelis: rome partition (AMD EPYC, 128 cores/node) ──────────
# rome is parallel-only and kills jobs with <60% utilisation.
# 32 cores saturates faiss HNSW build/search well without overshooting.

echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Partition : $SLURM_JOB_PARTITION"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "Started   : $(date)"

# Load the same toolchain you used to create the venv
module load gcc/14.2.0 python/3.13.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
cd "$SCRIPT_DIR"

mkdir -p logs results

# Let FAISS use all allocated cores (cgroups will pin us to them)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python run_experiment.py \
    --method hnsw \
    --profile full \
    --data_root ./data \
    --results_dir ./results

echo "Finished  : $(date)"

# After the job ends, check resource use with:  seff $SLURM_JOB_ID
