#!/usr/bin/env bash
#SBATCH --job-name=renly-m8-delivery
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
root=/share/home/renly/renly-patrol-rsi/runs/20261027-m8
required=129885,129886,129887_0,129887_1,129890,129897
for index in {0..11}; do required="$required,129889_$index"; done
python "$root/package-code-v1/scripts/package_m8_evidence.py" --root "$root" --code "$root/code-96b58422" --ledger "$root/slurm-outcomes-before-packaging.psv" --required-jobs "$required"
