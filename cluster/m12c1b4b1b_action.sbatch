#!/usr/bin/env bash
#SBATCH --job-name=rsi-b1b-action
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
source_root="$1"
run_root="$2"
seed_index="${3:-${SLURM_ARRAY_TASK_ID}}"
umask 077
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$source_root/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$source_root"
stage="$source_root/research/2026-10-30-m12c1b4b1b-limited"
old="$source_root/research/2026-10-30-m12c1b4b0-data"
prior=/share/home/renly/renly-patrol-rsi/runs/20261030-m12c1b4b1a
python scripts/diagnose_action_indexed.py --protocol "$stage/B4-B1b-Limited-v1.json" \
  --review "$stage/evidence/protocol-freeze.json" --old-protocol "$old/B4-B1冻结协议.json" \
  --old-manifest "$old/B4-B1新数据清单.json" --source-root "$prior/data" \
  --source-inventory "$prior/report/raw-storage.json" --seed-index "$seed_index" \
  --output "$run_root/action/seed-index-$seed_index"
