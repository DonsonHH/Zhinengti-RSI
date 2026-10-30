#!/usr/bin/env bash
#SBATCH --job-name=rsi-b4b1a-recheck
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
source_root="$1"
run_root="$2"
umask 077
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$source_root/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$source_root"
python scripts/verify_fresh_opportunity.py --root "$run_root/data" \
  --protocol "$source_root/research/2026-10-30-m12c1b4b0-data/B4-B1冻结协议.json" \
  --manifest "$source_root/research/2026-10-30-m12c1b4b0-data/B4-B1新数据清单.json" \
  --group-index "$SLURM_ARRAY_TASK_ID" --output "$run_root/audits-v4/group-$SLURM_ARRAY_TASK_ID.json"
