#!/usr/bin/env bash
#SBATCH --job-name=renly-b1-audit
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
set -euo pipefail
root=/share/home/renly/renly-patrol-rsi
run=$root/runs/20261029-m12c1b1
cd "$run/analysis-v1"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
phase="${1:?phase}"
python scripts/audit_m12c1b1.py --results "$run/$phase" --actors "$root/runs/20261027-m11/policies-v2" --estimator-weights "$root/runs/20261028-m111/frozen-estimator.pt" --estimator-manifest "$PWD/research/2026-10-28-m111-coverage/frozen_estimator.json" --validation "$run/validation" --output "$run/$phase-audit.json"
