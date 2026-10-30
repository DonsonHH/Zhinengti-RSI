#!/usr/bin/env bash
#SBATCH --job-name=rsi-b4b1a-package
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
source_root="$1"
run_root="$2"
umask 077
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$source_root/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$source_root"
bash cluster/m12c1b4b0_gate.sbatch "$source_root" "$run_root/final"
python scripts/package_fresh_evidence.py --run "$run_root" --output "$run_root/delivery"
