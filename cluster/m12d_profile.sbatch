#!/usr/bin/env bash
#SBATCH --job-name=rsi-m12d-cost-profile
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
set -euo pipefail
source_root="$1"
run_root="$2"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$source_root/src:$source_root/scripts"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$source_root"
python /share/home/renly/renly-patrol-rsi/runs/20261030-m12d/profile_m12d_rsi.py --source-root "$source_root" --run-root "$run_root"
