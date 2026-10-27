#!/usr/bin/env bash
#SBATCH --job-name=renly-m7-rereview
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH=/share/home/renly/renly-patrol-rsi/runs/20261027-m7/code-6809bdcc/src
python /share/home/renly/renly-patrol-rsi/runs/20261027-m7/report-code-v1/scripts/audit_m7.py --results /share/home/renly/renly-patrol-rsi/runs/20261027-m7/code-6809bdcc/results --fingerprints /share/home/renly/renly-patrol-rsi/runs/20261027-m7/code-6809bdcc/research/2026-10-27-m7-stress/frozen_m6_weights.json --output /share/home/renly/renly-patrol-rsi/runs/20261027-m8/m7-review-reaudit.json
