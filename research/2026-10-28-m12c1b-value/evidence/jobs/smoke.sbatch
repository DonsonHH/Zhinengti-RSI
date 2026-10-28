#!/usr/bin/env bash
#SBATCH --job-name=renly-c1b-smoke
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:12:00
set -euo pipefail
root=/share/home/renly/renly-patrol-rsi
run=$root/runs/20261028-m12c1b
cd "$run/code-v1"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s tests -v
common=(--actors "$root/runs/20261027-m11/policies-v2" --estimator-weights "$root/runs/20261028-m111/frozen-estimator.pt" --estimator-manifest "$PWD/research/2026-10-28-m111-coverage/frozen_estimator.json")
python -m patrol_rsi.m12c1b "${common[@]}" --stage data --smoke --output "$run/smoke-data"
python -m patrol_rsi.m12c1b "${common[@]}" --stage fit --smoke --data "$run/smoke-data" --output "$run/smoke-fit" --trial 0
