#!/bin/bash
#SBATCH --job-name=renly-m11-audit-report
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
root=/share/home/renly/renly-patrol-rsi/runs/20261027-m11
cd "$root/code-4538d789"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python "$root/analysis-code-v1/scripts/audit_m11.py" --results "$root/evaluation-129953" --models "$root/policies-v2" --estimator-weights "$root/frozen-estimator.pt" --estimator-manifest "$PWD/research/2026-10-27-m11-scheduling/frozen_estimator.json" --output "$root/full-audit-$SLURM_JOB_ID.json"
python "$root/analysis-code-v1/scripts/report_m11.py" --results "$root/evaluation-129953" --capacity "$root/capacity-129946" --output "$root/report-$SLURM_JOB_ID"
