#!/usr/bin/env bash
#SBATCH --job-name=renly-facility-m9
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
cd "${1:?source snapshot}"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1
if [ "${2:-full}" = smoke ]; then
 python -m unittest discover -s tests -v
 python -m patrol_rsi.m9 --output "$PWD/results/smoke-$SLURM_JOB_ID" --smoke
else
 python -m patrol_rsi.m9 --output "$PWD/results/full-$SLURM_JOB_ID"
fi
