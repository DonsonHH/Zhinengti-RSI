#!/bin/bash
#SBATCH --job-name=renly-temperature-m10
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:V100S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
cd "$1"
root="$2"
mode="$3"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [ "$mode" = smoke ]; then
    python -m unittest discover -s tests -v
    python -m patrol_rsi.estimation_data --phase development --output "$root/development-$SLURM_JOB_ID"
    python -m patrol_rsi.m10 train --smoke --data "$root/development-$SLURM_JOB_ID" --output "$root/smoke-models-$SLURM_JOB_ID"
elif [ "$mode" = train ]; then
    test -f "$root/development-gate.json"
    python -m patrol_rsi.estimation_data --phase train --output "$root/training-data-$SLURM_JOB_ID"
    python -m patrol_rsi.m10 train --data "$root/training-data-$SLURM_JOB_ID" --output "$root/models-$SLURM_JOB_ID"
else
    exit 2
fi
