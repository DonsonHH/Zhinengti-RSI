#!/usr/bin/env bash
#SBATCH --job-name=rsi-b4a-audit
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
set -euo pipefail
source_root="$1"
formal_root="$2"
output_root="$3"
mode="$4"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$source_root/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$source_root"
folders=(development-0 development-1 validation-0 validation-1 test-0 test-1 test-2 test-3)
extra=()
if [[ "$mode" == pilot ]]; then
    folder=development-0
    destination="$output_root/pilot"
    extra=(--pilot)
else
    folder="${folders[$SLURM_ARRAY_TASK_ID]}"
    destination="$output_root/$folder"
fi
python scripts/audit_information.py --formal-root "$formal_root" --folder "$folder" \
    --output "$destination" --estimator-weights /share/home/renly/renly-patrol-rsi/runs/20261028-m111/frozen-estimator.pt \
    --estimator-manifest "$source_root/research/2026-10-28-m111-coverage/frozen_estimator.json" "${extra[@]}"
