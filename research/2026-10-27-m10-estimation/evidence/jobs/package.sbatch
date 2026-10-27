#!/bin/bash
#SBATCH --job-name=renly-m10-delivery
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
root=/share/home/renly/renly-patrol-rsi/runs/20261027-m10
python "$root/package-code-v1/scripts/package_m10_evidence.py" --root "$root"
