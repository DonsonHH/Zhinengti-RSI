#!/usr/bin/env bash
#SBATCH --job-name=rsi-b4b1a-audit
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
source_root="$1"
run_root="$2"
umask 077
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$source_root/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$source_root"
protocol="$source_root/research/2026-10-30-m12c1b4b0-data/B4-B1冻结协议.json"
manifest="$source_root/research/2026-10-30-m12c1b4b0-data/B4-B1新数据清单.json"
python scripts/report_fresh_opportunity.py --root "$run_root/data" --protocol "$protocol" \
  --manifest "$manifest" --audit-root "$run_root/audits-v4" --output "$run_root/report"
python scripts/audit_fresh_counts.py --root "$run_root/data" --protocol "$protocol" \
  --manifest "$manifest" --report "$run_root/sealed-full-coverage.json" --output "$run_root/report/counts-independent.json"
python scripts/audit_fresh_sources.py --root "$run_root/data" --protocol "$protocol" \
  --manifest "$manifest" --storage "$run_root/report/raw-storage.json" \
  --weights /share/home/renly/renly-patrol-rsi/runs/20261028-m111/frozen-estimator.pt \
  --output "$run_root/report/source-replay.json"
