#!/usr/bin/env bash
#SBATCH --job-name=renly-m9-delivery
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
root=/share/home/renly/renly-patrol-rsi/runs/20261027-m9
python "$root/package-code-v1/scripts/package_m9_evidence.py" --root "$root" --results "$root/code-dd45056c/results/full-129910" --smoke "$root/code-dd45056c/results/smoke-129909" --ledger "$root/slurm-outcomes-before-packaging.psv" --required-jobs 129906,129909,129910,129911
