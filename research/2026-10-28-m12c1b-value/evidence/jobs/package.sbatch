#!/usr/bin/env bash
#SBATCH --job-name=renly-c1b-package
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
run=/share/home/renly/renly-patrol-rsi/runs/20261028-m12c1b
cd "$run/report-v1"
module purge
module load python/3.11-pytorch2.6-cu124
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/package_m12c1b.py --run-root "$run"
