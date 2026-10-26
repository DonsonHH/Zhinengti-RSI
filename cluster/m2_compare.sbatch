#!/usr/bin/env bash
#SBATCH --job-name=renly-patrol-m2
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:V100S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
set -euo pipefail
cd "${1:?source snapshot}"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
runtime="${2:?model venv}"
model="${3:?verified model}"
checkpoints="${4:?frozen M1 checkpoints}"
nvidia-smi --query-gpu=name,memory.total --format=csv
if [ "${5:-full}" = smoke ]; then
 "$runtime/bin/python" -m unittest discover -s tests -v
 "$runtime/bin/python" -m patrol_rsi.m2 --output "$PWD/results/smoke-$SLURM_JOB_ID" --model "$model" --checkpoints "$checkpoints" --trial 0 --smoke
else
 "$runtime/bin/python" -m patrol_rsi.m2 --output "$PWD/results/trial-$SLURM_ARRAY_TASK_ID" --model "$model" --checkpoints "$checkpoints" --trial "$SLURM_ARRAY_TASK_ID"
fi
