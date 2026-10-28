#!/usr/bin/env bash
#SBATCH --job-name=m12b-package
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
b=/share/home/renly/renly-patrol-rsi/runs/20261028-m12b
python "$b/package-code-v1/scripts/package_m12b.py" --run "$b" --jobs "$b/job-identities.json"
