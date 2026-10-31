#!/usr/bin/env bash
#SBATCH --job-name=rsi-m12f-confirmation
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
source_root="$1"
run_root="$2"
mode="$3"
batch="$4"
[[ "$mode" = F1_validation || "$mode" = F1_final || "$mode" = F2_validation || "$mode" = F2_final ]]
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$source_root/src:$source_root/scripts"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$source_root"
/share/home/renly/renly-patrol-rsi/runs/20261026-m1/model-env/bin/python scripts/run_m12f_campaign.py --source-root "$source_root" --run-root "$run_root" --mode "$mode" --batch "$batch"
