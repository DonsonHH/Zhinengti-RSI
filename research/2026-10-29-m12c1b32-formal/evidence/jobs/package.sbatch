#!/usr/bin/env bash
#SBATCH --job-name=b32-package
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
run=/share/home/renly/renly-patrol-rsi/runs/20261029-m12c1b32
cd "$run/report-v2"
python scripts/package_formal_study.py --run-root "$run"
