#!/usr/bin/env bash
#SBATCH --job-name=rsi-b1b-analysis
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
stage="$source_root/research/2026-10-30-m12c1b4b1b-limited"
old="$source_root/research/2026-10-30-m12c1b4b0-data"
prior=/share/home/renly/renly-patrol-rsi/runs/20261030-m12c1b4b1a
common=(--protocol "$stage/B4-B1b-Limited-v1.json" --review "$stage/evidence/protocol-freeze.json")
python scripts/report_limited_results.py "${common[@]}" --selection "$run_root/selection/selection-receipt.json" \
  --final-root "$run_root/final" --output "$run_root/report"
python scripts/diagnose_limited_predictions.py "${common[@]}" --selection "$run_root/selection/selection-receipt.json" \
  --cache "$run_root/cache" --final-root "$run_root/final" --output "$run_root/diagnostics"
python scripts/report_action_indexed.py "${common[@]}" --root "$run_root/action" \
  --B1a-coverage "$prior/report/coverage.json" --output "$run_root/action-report"
python scripts/audit_limited_evidence.py "${common[@]}" --old-manifest "$old/B4-B1新数据清单.json" \
  --source-root "$prior/data" --source-inventory "$prior/report/raw-storage.json" --cache "$run_root/cache" \
  --selection "$run_root/selection/selection-receipt.json" --final-root "$run_root/final" --output "$run_root/independent-audit"
