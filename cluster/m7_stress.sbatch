#!/usr/bin/env bash
#SBATCH --job-name=renly-patrol-m7
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:V100S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -euo pipefail
cd "${1:?source snapshot}"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
nvidia-smi --query-gpu=name,memory.total --format=csv
weights="${2:?M6 frozen weight directory}"
fingerprints="${3:?M6 fingerprints}"
estimator="${4:?frozen VAE}"
if [ "${5:-full}" = smoke ]; then
 python -m unittest discover -s tests -v
 python -m patrol_rsi.m7 --output "$PWD/results/smoke-$SLURM_JOB_ID" --weights-root "$weights" --fingerprints "$fingerprints" --estimator "$estimator" --smoke
else
 settings=(joint_vae joint_vae_tool_off no_vae no_vae_tool_on independent_targets joint_vae_mask_off independent_mask_on)
 planners=(risk_vae risk_hold matching_vae matching_hold slack pair_cycle)
 if [ "$SLURM_ARRAY_TASK_ID" -lt 21 ]; then
  setting="${settings[$((SLURM_ARRAY_TASK_ID / 3))]}"
  seed="$((SLURM_ARRAY_TASK_ID % 3))"
 else
  setting="${planners[$((SLURM_ARRAY_TASK_ID - 21))]}"
  seed=0
 fi
 python -m patrol_rsi.m7 --output "$PWD/results/$setting-seed-$seed" --weights-root "$weights" --fingerprints "$fingerprints" --estimator "$estimator" --setting "$setting" --seed "$seed"
fi
