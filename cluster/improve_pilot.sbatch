#!/usr/bin/env bash
#SBATCH --job-name=renly-rsi-pilot
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:V100S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
set -euo pipefail
cd "${1:?source snapshot}"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
runtime="${2:?model venv}"
model="${3:?verified local model}"
vae="${4:?frozen estimator}"
policy="${5:?frozen policy seed 0}"
test -s "$model/verified_manifest.json"
nvidia-smi --query-gpu=name,memory.total --format=csv
"$runtime/bin/python" -c 'import torch,transformers,accelerate; print(torch.__version__,transformers.__version__,accelerate.__version__); assert torch.cuda.is_available()'
"$runtime/bin/python" -m patrol_rsi.improve --output "$PWD/results/rsi-pilot-$SLURM_JOB_ID" --model "$model" --estimator "$vae" --policy "$policy"
