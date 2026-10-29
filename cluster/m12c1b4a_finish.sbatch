#!/usr/bin/env bash
#SBATCH --job-name=rsi-b4a-finish
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
set -euo pipefail
source_root="$1"
run_root="$2"
formal_root="$3"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$source_root/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$source_root"
python scripts/verify_information.py --root "$run_root/formal" --formal-root "$formal_root" \
    --output "$run_root/independent-verification.json"
python scripts/report_information.py --root "$run_root/formal" --output "$run_root/report"
cp "$run_root/independent-verification.json" "$run_root/report/independent-verification.json"
python scripts/package_information.py --root "$run_root" --output "$run_root/public-evidence"
