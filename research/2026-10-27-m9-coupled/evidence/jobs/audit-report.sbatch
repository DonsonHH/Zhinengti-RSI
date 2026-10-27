#!/usr/bin/env bash
#SBATCH --job-name=renly-m9-audit-report
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1
root=/share/home/renly/renly-patrol-rsi/runs/20261027-m9
export PYTHONPATH="$root/code-dd45056c/src"
python "$root/report-code-v1/scripts/audit_m9.py" --results "$root/code-dd45056c/results/full-129910" --output "$root/full-chain-audit.json"
python "$root/report-code-v1/scripts/report_m9.py" --results "$root/code-dd45056c/results/full-129910" --output "$root/final-report"
