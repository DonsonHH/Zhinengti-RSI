#!/usr/bin/env bash
#SBATCH --job-name=causal-audit
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
python scripts/test_causal_torch.py --output "$1"
python scripts/audit_causal_preflight.py \
  --pilot "$2" --output "$1/pilot-audit.json" \
  --actors /share/home/renly/renly-patrol-rsi/runs/20261027-m11/policies-v2 \
  --estimator-weights /share/home/renly/renly-patrol-rsi/runs/20261028-m111/frozen-estimator.pt \
  --estimator-manifest research/2026-10-28-m111-coverage/frozen_estimator.json
