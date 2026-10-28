#!/usr/bin/env bash
#SBATCH --job-name=renly-c1b-quality
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:12:00
set -euo pipefail
root=/share/home/renly/renly-patrol-rsi
run=$root/runs/20261028-m12c1b
cd "$run/analysis-v2"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/diagnose_m12c1b_quality.py --run-root "$run" --actors "$root/runs/20261027-m11/policies-v2" --estimator-weights "$root/runs/20261028-m111/frozen-estimator.pt" --estimator-manifest "$PWD/research/2026-10-28-m111-coverage/frozen_estimator.json" --output "$run/quality-details-audit.json"
