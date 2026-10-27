#!/usr/bin/env bash
#SBATCH --job-name=renly-m8-rereview
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
root=/share/home/renly/renly-patrol-rsi/runs/20261027-m9
previous=/share/home/renly/renly-patrol-rsi/runs/20261027-m8
export PYTHONPATH="$previous/code-96b58422/src"
python "$previous/report-code-v1/scripts/audit_m8.py" --results "$previous/code-96b58422/results" --output "$root/m8-review-reaudit.json"
python "$previous/report-code-v1/scripts/check_m8_checkpoints.py" --results "$previous/code-96b58422/results" --output "$root/m8-review-actual-weights.json"
