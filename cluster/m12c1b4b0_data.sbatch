#!/usr/bin/env bash
#SBATCH --job-name=rsi-b4b0-data
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
source_root="$1"
run_root="$2"
mode="$3"
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
    output="$run_root/pilot"
    extra=(--pilot)
else
    folder="${folders[$SLURM_ARRAY_TASK_ID]}"
    output="$run_root/full/$folder"
fi
python scripts/build_opportunity_dataset.py \
    --source-root /share/home/renly/renly-patrol-rsi/runs/20261029-m12c1b32 \
    --b4a-root /share/home/renly/renly-patrol-rsi/runs/20261029-m12c1b4a/formal \
    --source-inventory "$source_root/research/2026-10-29-m12c1b4a-information/evidence/raw-storage.json" \
    --folder "$folder" --output "$output" "${extra[@]}"
