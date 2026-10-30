#!/usr/bin/env bash
#SBATCH --job-name=rsi-b2-interface
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
python - "$run_root/reviewed/gate/full-tests.json" <<'PY'
import json,sys
g=json.load(open(sys.argv[1]));assert g['tests']>=276 and g['success'] and not(g['failures'] or g['errors'] or g['skipped'])
PY
python scripts/b2_archived_interface_smoke.py \
  --source-root /share/home/renly/renly-patrol-rsi/runs/20261030-m12c1b4b1a/data \
  --source-inventory /share/home/renly/renly-patrol-rsi/runs/20261030-m12c1b4b1a/report/raw-storage.json \
  --actors /share/home/renly/renly-patrol-rsi/runs/20261027-m11/policies-v2 \
  --estimator-weights /share/home/renly/renly-patrol-rsi/runs/20261028-m111/frozen-estimator.pt \
  --estimator-manifest "$source_root/research/2026-10-28-m111-coverage/frozen_estimator.json" \
  --output "$run_root/interface-smoke"
