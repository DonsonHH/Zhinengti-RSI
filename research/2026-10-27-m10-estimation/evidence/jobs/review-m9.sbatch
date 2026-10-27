#!/bin/bash
#SBATCH --job-name=renly-m10-review-m9
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
root=/share/home/renly/renly-patrol-rsi/runs/20261027-m10
old=/share/home/renly/renly-patrol-rsi/runs/20261027-m9
cd "$old/code-dd45056c"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
python "$old/report-code-v1/scripts/audit_m9.py" --results "$PWD/results/full-129910" --output "$root/m9-full-review-$SLURM_JOB_ID.json"
