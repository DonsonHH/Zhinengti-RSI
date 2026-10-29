#!/usr/bin/env bash
#SBATCH --job-name=b32-report
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run=/share/home/renly/renly-patrol-rsi/runs/20261029-m12c1b32
cd "$run/report-v2"
export PYTHONPATH="$PWD/src"
python scripts/report_formal_study.py --run-root "$run" --protocol research/2026-10-29-m12c1b31-interface/正式实验冻结方案.json --output "$run/report"
