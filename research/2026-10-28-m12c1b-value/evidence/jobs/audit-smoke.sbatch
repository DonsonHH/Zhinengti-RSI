#!/usr/bin/env bash
#SBATCH --job-name=renly-c1b-audit-smoke
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
root=/share/home/renly/renly-patrol-rsi
run=$root/runs/20261028-m12c1b
cd "$run/analysis-v1"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
common=(--actors "$root/runs/20261027-m11/policies-v2" --estimator-weights "$root/runs/20261028-m111/frozen-estimator.pt" --estimator-manifest "$PWD/research/2026-10-28-m111-coverage/frozen_estimator.json")
python scripts/audit_m12c1b.py "${common[@]}" --results "$run/smoke-data" --output "$run/smoke-data-audit.json"
python scripts/audit_m12c1b.py "${common[@]}" --results "$run/smoke-fit" --data "$run/smoke-data" --output "$run/smoke-fit-audit.json"
