#!/usr/bin/env bash
#SBATCH --job-name=renly-patrol-m1
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:V100S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
set -euo pipefail
cd "${1:?project directory}"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
stage="${2:?vae or policy}"
if [ "$stage" = vae ]; then
 python -m unittest discover -s tests -v
 python -m patrol_rsi.train vae --weights "$PWD/assets/vgg16_bn-6c64b313.pth" --output "$PWD/results/vae"
else
 python -m patrol_rsi.train policy --seed "$SLURM_ARRAY_TASK_ID" --estimator "$PWD/results/vae/best-vae.pt" --output "$PWD/results/seed-$SLURM_ARRAY_TASK_ID"
fi
