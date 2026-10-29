#!/usr/bin/env bash
#SBATCH --job-name=renly-b2-confirm
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
cd "${1:?frozen source}"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
root=/share/home/renly/renly-patrol-rsi
run="${2:?run root}";mode="${3:?smoke or formal}"
common=(--actors "$root/runs/20261027-m11/policies-v2" --estimator-weights "$root/runs/20261028-m111/frozen-estimator.pt" --estimator-manifest "$PWD/research/2026-10-28-m111-coverage/frozen_estimator.json")
if [ "$mode" = smoke ]; then
    python -m unittest discover -s tests -v
    python -m patrol_rsi.m12c1b2 "${common[@]}" --smoke --output "$run/smoke"
else
    python -m patrol_rsi.m12c1b2 "${common[@]}" --chunk "$SLURM_ARRAY_TASK_ID" --output "$run/chunk-$SLURM_ARRAY_TASK_ID"
fi
