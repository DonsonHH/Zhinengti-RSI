#!/usr/bin/env bash
#SBATCH --job-name=rsi-b1b-limited
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
source_root="$1"
run_root="$2"
phase="$3"
cell="${4:-${SLURM_ARRAY_TASK_ID:-0}}"
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
case "$phase" in
  prepare)
    for split in development validation_calibration validation_selection;do
      python scripts/prepare_limited_data.py "${common[@]}" --old-protocol "$old/B4-B1冻结协议.json" \
        --old-manifest "$old/B4-B1新数据清单.json" --source-root "$prior/data" \
        --source-inventory "$prior/report/raw-storage.json" --coverage "$prior/report/coverage.json" \
        --output "$run_root/cache" --phase "$split"
    done
    ;;
  fit)
    python scripts/fit_limited_predictors.py "${common[@]}" --gate "$run_root/reviewed/gate/full-tests.json" \
      --cache "$run_root/cache" --output "$run_root/fits" --cell "$cell"
    ;;
  select)
    python scripts/select_limited_predictors.py "${common[@]}" --cache "$run_root/cache" \
      --fits "$run_root/fits" --output "$run_root/selection"
    ;;
  final-cache)
    python scripts/prepare_limited_data.py "${common[@]}" --old-protocol "$old/B4-B1冻结协议.json" \
      --old-manifest "$old/B4-B1新数据清单.json" --source-root "$prior/data" \
      --source-inventory "$prior/report/raw-storage.json" --coverage "$prior/report/coverage.json" \
      --output "$run_root/cache" --phase final_test --selection-receipt "$run_root/selection/selection-receipt.json"
    ;;
  final-score)
    python scripts/evaluate_limited_final.py "${common[@]}" --cache "$run_root/cache" \
      --selection "$run_root/selection/selection-receipt.json" --output "$run_root/final" --cell "$cell"
    ;;
  *) printf 'Unknown stage\n' >&2;exit 2;;
esac
