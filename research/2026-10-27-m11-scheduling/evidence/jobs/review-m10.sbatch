#!/bin/bash
#SBATCH --job-name=renly-m11-review-m10
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
root=/share/home/renly/renly-patrol-rsi/runs/20261027-m11
old=/share/home/renly/renly-patrol-rsi/runs/20261027-m10
cd "$old/code-5696d2a1"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python "$old/analysis-code-v1/scripts/audit_m10.py" --train-data "$old/training-data-129923" --test-data "$old/test-data-129925" --models "$old/models-129923" --results "$old/evaluation-129925" --output "$root/m10-full-review-$SLURM_JOB_ID.json"
