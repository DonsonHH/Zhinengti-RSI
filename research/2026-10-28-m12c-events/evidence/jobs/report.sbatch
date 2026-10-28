#!/usr/bin/env bash
#SBATCH --job-name=m12c-report
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:08:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
b=/share/home/renly/renly-patrol-rsi/runs/20261028-m12c
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python - <<'PY'
from pathlib import Path
import json
b=Path('/share/home/renly/renly-patrol-rsi/runs/20261028-m12c')
for t in range(3):
    a=json.loads((b/f'audit-{t}.json').read_text());s=json.loads((b/f'trial-{t}/summary.json').read_text())
    assert a['actual_closed_loops_replayed']==s['unique_cases'] and a['actual_model_calls']==15 and a['maximum_exact_gap_ratio']<=1
PY
python "$b/analysis-code-v1/scripts/report_m12c.py" --results "$b" --output "$b/report"
