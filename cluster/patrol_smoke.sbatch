#!/usr/bin/env bash
#SBATCH --job-name=renly-patrol-smoke
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:V100S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
set -euo pipefail
cd "${1:?Pass the staged project directory}"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -m unittest discover -s tests -v
python -m patrol_rsi.smoke --device cuda:0 --steps 48 --seed 7 --output "$PWD/results/$SLURM_JOB_ID"
