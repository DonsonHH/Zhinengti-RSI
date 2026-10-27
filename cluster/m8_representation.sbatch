#!/usr/bin/env bash
#SBATCH --job-name=renly-patrol-m8
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:V100S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
set -euo pipefail
cd "${1:?source snapshot}"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
nvidia-smi --query-gpu=name,memory.total --format=csv
estimator="${2:?frozen VAE}"
mode="${3:-full}"
if [ "$mode" = smoke ]; then
 python -m unittest discover -s tests -v
 for variant in mixed_local mixed_relation; do
  python -m patrol_rsi.m8 --output "$PWD/results/smoke-$SLURM_JOB_ID-$variant" --estimator "$estimator" --variant "$variant" --seed 0 --smoke
 done
elif [ "$mode" = preflight ]; then
 variants=(mixed_local mixed_relation)
 variant="${variants[$SLURM_ARRAY_TASK_ID]}"
 python -m patrol_rsi.m8 --output "$PWD/results/preflight-$SLURM_ARRAY_JOB_ID-$variant" --estimator "$estimator" --variant "$variant" --seed 0 --preflight
elif [ "$mode" = planners ]; then
 python -m patrol_rsi.m8 --output "$PWD/results/common-planners" --estimator "$estimator" --planners
else
 variants=(rack_local rack_relation mixed_local mixed_relation)
 variant="${variants[$((SLURM_ARRAY_TASK_ID / 3))]}"
 seed="$((SLURM_ARRAY_TASK_ID % 3))"
 python -m patrol_rsi.m8 --output "$PWD/results/$variant-seed-$seed" --estimator "$estimator" --variant "$variant" --seed "$seed"
fi
