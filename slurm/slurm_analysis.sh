#!/bin/bash
#SBATCH --job-name=faiss_analysis
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err

# ── Analysis (§6 RC, §7 perturbation, §8 adaptation, §9 modified queries) ───
# Same partition as HNSW yesterday — AMD EPYC, 128 cores, parallel-only queue.
# 32 cores keeps efficiency safely above rome's 60% threshold.

echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Partition : $SLURM_JOB_PARTITION"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "Started   : $(date)"

module load gcc/14.2.0 python/3.13.0

SCRIPT_DIR="$HOME/project"
source "$SCRIPT_DIR/venv/bin/activate"
cd "$SCRIPT_DIR"

mkdir -p logs results modified_queries

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python run_analysis.py \
    --data_root ./data \
    --results_dir ./results \
    --query_dir ./modified_queries

echo "Finished  : $(date)"

# After completion, check resource use with:  seff $SLURM_JOB_ID
