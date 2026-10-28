#!/usr/bin/env bash
#SBATCH --job-name=c1a-report
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:08:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
b=/share/home/renly/renly-patrol-rsi/runs/20261028-m12c1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python - <<'PY'
from pathlib import Path
import json
b=Path('/share/home/renly/renly-patrol-rsi/runs/20261028-m12c1')
a=json.loads((b/'formal-audit.json').read_text());s=json.loads((b/'formal-v1/summary.json').read_text())
assert a['all_actual_cases_replayed']==s['all_actual_cases'] and a['maximum_exact_gap_ratio']<=1 and a['main_cases_replayed']==192
PY
python "$b/report-code-v1/scripts/report_m12c1.py" --results "$b/formal-v1" --output "$b/report"
