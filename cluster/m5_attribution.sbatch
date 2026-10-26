#!/usr/bin/env bash
#SBATCH --job-name=renly-patrol-m5
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:V100S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
set -euo pipefail
cd "${1:?source snapshot}"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
nvidia-smi --query-gpu=name,memory.total --format=csv
estimator="${2:?frozen VAE checkpoint}"
variants=(deadline_vae no_vae no_deadline no_shaping)
if [ "${3:-full}" = smoke ]; then
 python -m unittest discover -s tests -v
 python -m patrol_rsi.m5 --output "$PWD/results/smoke-$SLURM_JOB_ID" --estimator "$estimator" --seed 0 --variant deadline_vae --smoke
else
 index="$SLURM_ARRAY_TASK_ID"
 variant="${variants[$((index / 3))]}"
 seed="$((index % 3))"
 python -m patrol_rsi.m5 --output "$PWD/results/$variant-seed-$seed" --estimator "$estimator" --seed "$seed" --variant "$variant"
fi
