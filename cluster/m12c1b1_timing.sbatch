#!/usr/bin/env bash
#SBATCH --job-name=renly-b1-timing
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:35:00
set -euo pipefail
cd "${1:?source}"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
phase="${2:?phase}";output="${3:?output}";validation="${4:-}"
root=/share/home/renly/renly-patrol-rsi
if [ "$phase" = smoke ]; then python -m unittest discover -s tests -v; fi
args=(--phase "$phase" --output "$output" --actors "$root/runs/20261027-m11/policies-v2" --estimator-weights "$root/runs/20261028-m111/frozen-estimator.pt" --estimator-manifest "$PWD/research/2026-10-28-m111-coverage/frozen_estimator.json")
if [ -n "$validation" ]; then args+=(--validation "$validation"); fi
python -m patrol_rsi.m12c1b1 "${args[@]}"
