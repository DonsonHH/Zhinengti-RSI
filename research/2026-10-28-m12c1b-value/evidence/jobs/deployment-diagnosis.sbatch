#!/usr/bin/env bash
#SBATCH --job-name=renly-c1b-deployment
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:08:00
set -euo pipefail
run=/share/home/renly/renly-patrol-rsi/runs/20261028-m12c1b
cd "$run/analysis-v3"
module purge
module load python/3.11-pytorch2.6-cu124
python scripts/diagnose_m12c1b_deployment.py --run-root "$run" --output "$run/deployment-diagnosis-audit.json"
