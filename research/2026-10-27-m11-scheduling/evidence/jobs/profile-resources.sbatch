#!/bin/bash
#SBATCH --job-name=renly-m11-resource-profile
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
root=/share/home/renly/renly-patrol-rsi/runs/20261027-m11
export PYTHONPATH="$root/code-56cafb4f/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python "$root/profile-resources.py"
