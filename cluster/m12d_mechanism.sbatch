#!/usr/bin/env bash
#SBATCH --job-name=rsi-m12d-behavior-audit
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
set -euo pipefail
python "$1/diagnose_m12d_mechanism.py" --run-root "$1"
