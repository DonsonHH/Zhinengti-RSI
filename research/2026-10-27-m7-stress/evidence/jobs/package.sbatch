#!/usr/bin/env bash
#SBATCH --job-name=renly-m7-delivery
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
root=/share/home/renly/renly-patrol-rsi/runs/20261027-m7
python "$root/package-code-v1/scripts/package_m7_evidence.py" --root "$root" --code "$root/code-6809bdcc" --ledger "$root/slurm-outcomes-before-packaging.psv" --array-id 129856 --review-id 129853 --smoke-id 129855 --report-id 129878
