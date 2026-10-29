#!/usr/bin/env bash
#SBATCH --job-name=causal-preflight
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
python scripts/preflight_causal_stopping.py \
  --output "$1" \
  --actors /share/home/renly/renly-patrol-rsi/runs/20261027-m11/policies-v2 \
  --estimator-weights /share/home/renly/renly-patrol-rsi/runs/20261028-m111/frozen-estimator.pt \
  --estimator-manifest research/2026-10-28-m111-coverage/frozen_estimator.json \
  --protocol research/2026-10-29-m12c1b31-interface/正式实验冻结方案.json
