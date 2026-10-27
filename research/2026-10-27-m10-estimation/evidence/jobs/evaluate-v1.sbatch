#!/bin/bash
#SBATCH --job-name=renly-m10-test
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
set -euo pipefail
cd "$1"
root="$2"
models="$3"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
test -f "$root/training-gate.json"
python -m patrol_rsi.estimation_data --phase test --output "$root/test-data-$SLURM_JOB_ID"
python -m patrol_rsi.m10 evaluate --data "$root/test-data-$SLURM_JOB_ID" --models "$models" --output "$root/evaluation-$SLURM_JOB_ID"
