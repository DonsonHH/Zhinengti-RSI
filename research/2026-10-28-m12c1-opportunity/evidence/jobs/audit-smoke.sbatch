#!/usr/bin/env bash
#SBATCH --job-name=c1a-smoke-audit
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
r=/share/home/renly/renly-patrol-rsi
b=$r/runs/20261028-m12c1
export PYTHONPATH="$b/code-b69432b6/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python "$b/analysis-code-v1/scripts/audit_m12c1.py" --results "$b/smoke-v1" --actors "$r/runs/20261027-m11/policies-v2" --estimator-weights "$r/runs/20261028-m111/frozen-estimator.pt" --estimator-manifest "$b/code-b69432b6/research/2026-10-28-m111-coverage/frozen_estimator.json" --output "$b/smoke-audit.json"
