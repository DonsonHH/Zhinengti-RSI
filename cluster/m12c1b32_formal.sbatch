#!/usr/bin/env bash
#SBATCH --job-name=b32-formal
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
phase="$1"
run="$2"
selection=()
if [[ "$phase" == validation ]]; then selection=(--selection "$run/development-selection.json"); fi
if [[ "$phase" == test ]]; then selection=(--selection "$run/validation-selection.json"); fi
python -m patrol_rsi.formal_run --phase "$phase" --chunk "$SLURM_ARRAY_TASK_ID" \
  --output "$run/$phase-$SLURM_ARRAY_TASK_ID" --gate "$run/gate/full-tests.json" \
  --protocol research/2026-10-29-m12c1b31-interface/正式实验冻结方案.json \
  --actors /share/home/renly/renly-patrol-rsi/runs/20261027-m11/policies-v2 \
  --estimator-weights /share/home/renly/renly-patrol-rsi/runs/20261028-m111/frozen-estimator.pt \
  --estimator-manifest research/2026-10-28-m111-coverage/frozen_estimator.json "${selection[@]}"
