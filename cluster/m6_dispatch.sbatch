#!/usr/bin/env bash
#SBATCH --job-name=renly-patrol-m6
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:V100S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:15:00
set -euo pipefail
cd "${1:?source snapshot}"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
nvidia-smi --query-gpu=name,memory.total --format=csv
estimator="${2:?fixed VAE}"
stage="${3:-full}"
if [ "$stage" = smoke ]; then
 python -m unittest discover -s tests -v
 python -m patrol_rsi.m6 --output "$PWD/results/smoke-$SLURM_JOB_ID" --estimator "$estimator" --seed 0 --variant joint_vae --smoke
elif [ "$stage" = preflight ]; then
 python -m patrol_rsi.m6 --output "$PWD/results/preflight-$SLURM_JOB_ID" --estimator "$estimator" --seed 0 --variant joint_vae --preflight
else
 variants=(joint_vae no_vae independent_targets)
 variant="${variants[$((SLURM_ARRAY_TASK_ID / 3))]}"
 seed="$((SLURM_ARRAY_TASK_ID % 3))"
 python -m patrol_rsi.m6 --output "$PWD/results/$variant-seed-$seed" --estimator "$estimator" --seed "$seed" --variant "$variant"
fi
