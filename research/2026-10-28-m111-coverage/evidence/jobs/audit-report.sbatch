#!/bin/bash
#SBATCH --job-name=renly-m111-audit-report
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
root=/share/home/renly/renly-patrol-rsi/runs/20261028-m111
cd "$root/code-d3008cf4"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python "$root/analysis-code-v1/scripts/audit_m111.py" --results "$root/pilot-130101" --estimator-weights "$root/frozen-estimator.pt" --estimator-manifest "$PWD/research/2026-10-28-m111-coverage/frozen_estimator.json" --output "$root/full-audit-$SLURM_JOB_ID.json"
python "$root/analysis-code-v1/scripts/report_m111.py" --results "$root/pilot-130101" --output "$root/report-$SLURM_JOB_ID"
