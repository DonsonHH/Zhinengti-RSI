#!/usr/bin/env bash
#SBATCH --job-name=rsi-m12d-phase
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
seed_index="$4"
umask 077
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$source_root/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$source_root"
stage="$source_root/research/2026-10-30-m12d-formal"
prior="$source_root/research/2026-10-30-m12c1b4b2-alignment"
python scripts/run_m12d_paired.py --paired "$prior/MultiAgent-Patrol-Paired-v1.json" --rsi "$prior/RSI-Controlled-Meta-v1.json" --manifest "$stage/process-manifest.json" --review "$stage/evidence/execution-review.json" --gate "$run_root/reviewed/gate/full-tests.json" --run-root "$run_root" --actors /share/home/renly/renly-patrol-rsi/runs/20261027-m11/policies-v2 --estimator-weights /share/home/renly/renly-patrol-rsi/runs/20261028-m111/frozen-estimator.pt --estimator-manifest "$source_root/research/2026-10-28-m111-coverage/frozen_estimator.json" --phase "$phase" --seed-index "$seed_index"
