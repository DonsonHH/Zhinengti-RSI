#!/usr/bin/env bash
#SBATCH --job-name=rsi-b4b1a-gen
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
set -euo pipefail
source_root="$1"
run_root="$2"
group_index="${3:-${SLURM_ARRAY_TASK_ID}}"
umask 077
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$source_root/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$source_root"
protocol="$source_root/research/2026-10-30-m12c1b4b0-data/B4-B1冻结协议.json"
manifest="$source_root/research/2026-10-30-m12c1b4b0-data/B4-B1新数据清单.json"
python scripts/report_fresh_opportunity.py --root "$run_root/data" --protocol "$protocol" \
  --manifest "$manifest" --partial --output "$run_root/audits/before-$group_index.json"
python scripts/generate_fresh_opportunity.py --protocol "$protocol" --manifest "$manifest" \
  --output "$run_root/data" --group-index "$group_index" --gate "$run_root/gate/full-tests.json" \
  --estimator-weights /share/home/renly/renly-patrol-rsi/runs/20261028-m111/frozen-estimator.pt \
  --estimator-manifest "$source_root/research/2026-10-28-m111-coverage/frozen_estimator.json"
python scripts/verify_fresh_opportunity.py --root "$run_root/data" --protocol "$protocol" \
  --manifest "$manifest" --group-index "$group_index" --output "$run_root/audits/group-$group_index.json"
python scripts/report_fresh_opportunity.py --root "$run_root/data" --protocol "$protocol" \
  --manifest "$manifest" --partial --output "$run_root/audits/after-$group_index.json"
