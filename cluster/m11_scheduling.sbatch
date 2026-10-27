#!/bin/bash
#SBATCH --job-name=renly-active-m11
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:30:00
set -euo pipefail
cd "$1"
root="$2"
mode="$3"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
weights="$root/frozen-estimator.pt"
manifest="$PWD/research/2026-10-27-m11-scheduling/frozen_estimator.json"
if [ "$mode" = smoke ]; then
    python -m unittest discover -s tests -v
    python -m patrol_rsi.m11 capacity --output "$root/capacity-$SLURM_JOB_ID"
    python -m patrol_rsi.m11 train --smoke --estimator history --seed 0 --estimator-weights "$weights" --estimator-manifest "$manifest" --output "$root/smoke-model-$SLURM_JOB_ID"
elif [ "$mode" = train ]; then
    test -f "$root/development-gate-v2.json"
    index="$SLURM_ARRAY_TASK_ID"
    seed=$((index % 3))
    estimator=nominal
    if [ "$index" -ge 3 ]; then estimator=history; fi
    python -m patrol_rsi.m11 train --estimator "$estimator" --seed "$seed" --estimator-weights "$weights" --estimator-manifest "$manifest" --output "$root/policies-v2/$estimator-seed-$seed"
elif [ "$mode" = evaluate ]; then
    test -f "$root/training-gate-v2.json"
    python -m patrol_rsi.m11 evaluate --estimator-weights "$weights" --estimator-manifest "$manifest" --models "$root/policies-v2" --output "$root/evaluation-$SLURM_JOB_ID"
else
    exit 2
fi
