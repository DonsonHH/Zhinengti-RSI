#!/bin/bash
#SBATCH --job-name=renly-m111-delivery
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
root=/share/home/renly/renly-patrol-rsi/runs/20261028-m111
python "$root/package-code-v1/scripts/package_m111.py" --root "$root"
