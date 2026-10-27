#!/usr/bin/env bash
#SBATCH --job-name=renly-m8-audit-report
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
root=/share/home/renly/renly-patrol-rsi/runs/20261027-m8
export PYTHONPATH="$root/code-96b58422/src"
results="$root/code-96b58422/results"
python "$root/report-code-v1/scripts/audit_m8.py" --results "$results" --output "$root/full-chain-audit.json"
python "$root/report-code-v1/scripts/check_m8_checkpoints.py" --results "$results" --output "$root/actual-weight-audit.json"
python "$root/report-code-v1/scripts/report_m8.py" --results "$results" --output "$root/final-report"
