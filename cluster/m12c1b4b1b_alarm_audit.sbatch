#!/usr/bin/env bash
#SBATCH --job-name=rsi-b1b-alarm-audit
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
stage="$source_root/research/2026-10-30-m12c1b4b1b-limited"
python scripts/audit_limited_alarm_counts.py --protocol "$stage/B4-B1b-Limited-v1.json" \
  --review "$stage/evidence/protocol-freeze.json" --selection "$run_root/selection/selection-receipt.json" \
  --cache "$run_root/cache" --final-root "$run_root/final" --diagnostics "$run_root/diagnostics" --output "$run_root/alarm-audit"
