#!/usr/bin/env bash
#SBATCH --job-name=renly-m7-audit-report
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:25:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
root=/share/home/renly/renly-patrol-rsi/runs/20261027-m7
export PYTHONPATH="$root/code-6809bdcc/src"
python "$root/report-code-v1/scripts/audit_m7.py" --results "$root/code-6809bdcc/results" --fingerprints "$root/code-6809bdcc/research/2026-10-27-m7-stress/frozen_m6_weights.json" --output "$root/full-chain-audit.json"
python "$root/report-code-v1/scripts/report_m7.py" --results "$root/code-6809bdcc/results" --output "$root/final-report"
