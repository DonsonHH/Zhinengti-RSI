#!/usr/bin/env bash
#SBATCH --job-name=rsi-b4b0-verify
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
source_root="$1"
run_root="$2"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$source_root/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$source_root"
python scripts/verify_opportunity_data.py --data-root "$run_root/full" \
    --b4a-root /share/home/renly/renly-patrol-rsi/runs/20261029-m12c1b4a/formal \
    --source-root /share/home/renly/renly-patrol-rsi/runs/20261029-m12c1b32 \
    --output "$run_root/independent-verification.json"
python scripts/report_opportunity_data.py --root "$run_root/full" --output "$run_root/report"
cp "$run_root/independent-verification.json" "$run_root/report/independent-verification.json"
