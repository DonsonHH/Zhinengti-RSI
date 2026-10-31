#!/usr/bin/env bash
#SBATCH --job-name=rsi-m12g-result-audit
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
source_root="$1"
run_root="$2"
module purge
module load python/3.11-pytorch2.6-cu124
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$source_root/src:$source_root/scripts"
cd "$run_root"
python_exe=/share/home/renly/renly-patrol-rsi/runs/20261026-m1/model-env/bin/python
"$python_exe" -B "$source_root/scripts/finish_m12g_g0.py" --run-root "$run_root"
"$python_exe" -B "$run_root/audit-tools/derive_m12g_feedback.py" --run-root "$run_root" --feedback-module "$run_root/audit-tools/g_feedback_results.py"
export PYTHONPATH=""
"$python_exe" -B "$run_root/audit-tools/audit_m12g_results.py" --run-root "$run_root" --source-root "$source_root" --f-audit-kernel "$run_root/audit-tools/audit_m12f_actual.py" --accounting "$run_root/resource-sacct-G0.psv" --output "$run_root/G0-independent-result-audit-v1.json"
