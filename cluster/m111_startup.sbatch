#!/bin/bash
#SBATCH --job-name=renly-covered-m111
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
cd "$1"
root="$2"
mode="$3"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [ "$mode" = development ]; then
    python -m unittest discover -s tests -v
elif [ "$mode" = pilot ]; then
    test -f "$root/development-gate.json"
else
    exit 2
fi
python -m patrol_rsi.m111 --phase "$mode" --estimator-weights "$root/frozen-estimator.pt" --estimator-manifest "$PWD/research/2026-10-28-m111-coverage/frozen_estimator.json" --output "$root/$mode-$SLURM_JOB_ID"
