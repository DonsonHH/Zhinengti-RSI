#!/usr/bin/env bash
#SBATCH --job-name=renly-m6-rereview
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH=/share/home/renly/renly-patrol-rsi/runs/20261027-m6/code-2b468f92/src
python /share/home/renly/renly-patrol-rsi/runs/20261027-m6/final-review-code-v2/scripts/audit_m6.py \
 --results /share/home/renly/renly-patrol-rsi/runs/20261027-m6/delivery-evidence \
 --output /share/home/renly/renly-patrol-rsi/runs/20261027-m7/m6-review-reaudit.json
