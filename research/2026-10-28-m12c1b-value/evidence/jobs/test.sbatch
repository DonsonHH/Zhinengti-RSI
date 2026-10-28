#!/usr/bin/env bash
#SBATCH --job-name=renly-c1b-test
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
set -euo pipefail
root=/share/home/renly/renly-patrol-rsi
run=$root/runs/20261028-m12c1b
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then stage=test-common;trial=0;name=test-common; else stage=test-trial;trial=$((SLURM_ARRAY_TASK_ID-1));name=test-$trial; fi
bash "$run/code-v1/cluster/m12c1b_value.sbatch" "$run/code-v1" "$root/runs/20261027-m11/policies-v2" "$root/runs/20261028-m111/frozen-estimator.pt" "$run/code-v1/research/2026-10-28-m111-coverage/frozen_estimator.json" --stage "$stage" --run-root "$run" --trial "$trial" --output "$run/$name"
