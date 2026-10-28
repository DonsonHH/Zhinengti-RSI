#!/usr/bin/env bash
#SBATCH --job-name=renly-m12c-events
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:V100S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
set -euo pipefail
cd "${1:?frozen source}"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
runtime="${2:?runtime}";model="${3:?model}";actors="${4:?actors}"
estimator="${5:?estimator}";manifest="${6:?manifest}";mode="${7:?mode}";output="${8:?output root}"
nvidia-smi --query-gpu=name,memory.total --format=csv
if [ "$mode" = smoke ]; then
    "$runtime/bin/python" -m unittest discover -s tests -v
    "$runtime/bin/python" -m patrol_rsi.m12c --smoke --output "$output/smoke-$SLURM_JOB_ID" --model "$model" \
        --actors "$actors" --estimator-weights "$estimator" --estimator-manifest "$manifest"
else
    "$runtime/bin/python" -m patrol_rsi.m12c --trial "$SLURM_ARRAY_TASK_ID" --output "$output/trial-$SLURM_ARRAY_TASK_ID" \
        --model "$model" --actors "$actors" --estimator-weights "$estimator" --estimator-manifest "$manifest"
fi
