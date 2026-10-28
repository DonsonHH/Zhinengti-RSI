#!/usr/bin/env bash
#SBATCH --job-name=renly-c1b-value
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:50:00
set -euo pipefail
cd "${1:?frozen source}"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
actors="${2:?actors}";estimator="${3:?estimator}";manifest="${4:?manifest}";shift 4
python -m patrol_rsi.m12c1b --actors "$actors" --estimator-weights "$estimator" --estimator-manifest "$manifest" "$@"
