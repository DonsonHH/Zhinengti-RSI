#!/usr/bin/env bash
#SBATCH --job-name=renly-b1-package
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:08:00
set -euo pipefail
run=/share/home/renly/renly-patrol-rsi/runs/20261029-m12c1b1
cd "$run/report-v1"
module purge
module load python/3.11-pytorch2.6-cu124
python scripts/package_m12c1b1.py --run "$run"
