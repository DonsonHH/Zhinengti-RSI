#!/bin/bash
#SBATCH --job-name=renly-m10-audit-report
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
root=/share/home/renly/renly-patrol-rsi/runs/20261027-m10
cd "$root/code-5696d2a1"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python "$root/analysis-code-v1/scripts/audit_m10.py" --train-data "$root/training-data-129923" --test-data "$root/test-data-129925" --models "$root/models-129923" --results "$root/evaluation-129925" --output "$root/full-audit-$SLURM_JOB_ID.json"
python "$root/analysis-code-v1/scripts/report_m10.py" --results "$root/evaluation-129925" --data "$root/test-data-129925" --models "$root/models-129923" --output "$root/report-$SLURM_JOB_ID"
