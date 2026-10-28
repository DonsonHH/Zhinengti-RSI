#!/usr/bin/env bash
#SBATCH --job-name=m12c-full-audit
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
r=/share/home/renly/renly-patrol-rsi
b=$r/runs/20261028-m12c
t=$SLURM_ARRAY_TASK_ID
export PYTHONPATH="$b/code-df540969/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python "$b/analysis-code-v1/scripts/audit_m12c.py" --metadata-only --results "$b/trial-$t" --actors "$r/runs/20261027-m11/policies-v2" --estimator-weights "$r/runs/20261028-m111/frozen-estimator.pt" --estimator-manifest "$b/code-df540969/research/2026-10-28-m111-coverage/frozen_estimator.json" --output "$b/metadata-$t.json"
python "$b/analysis-code-v1/scripts/audit_m12c.py" --results "$b/trial-$t" --actors "$r/runs/20261027-m11/policies-v2" --estimator-weights "$r/runs/20261028-m111/frozen-estimator.pt" --estimator-manifest "$b/code-df540969/research/2026-10-28-m111-coverage/frozen_estimator.json" --output "$b/audit-$t.json"
