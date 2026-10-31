#!/usr/bin/env bash
#SBATCH --job-name=rsi-m12d-phase-audit
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
source_root="$1"
run_root="$2"
phase="$3"
umask 077
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$source_root/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$source_root"
prior="$source_root/research/2026-10-30-m12c1b4b2-alignment"
python scripts/finalize_m12d_phase.py --paired "$prior/MultiAgent-Patrol-Paired-v1.json" --rsi "$prior/RSI-Controlled-Meta-v1.json" --manifest "$source_root/research/2026-10-30-m12d-formal/process-manifest.json" --run-root "$run_root" --phase "$phase"
