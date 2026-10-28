#!/usr/bin/env bash
#SBATCH --job-name=m12b-report
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
b=/share/home/renly/renly-patrol-rsi/runs/20261028-m12b
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python - <<'PY'
from pathlib import Path
import json
b=Path('/share/home/renly/renly-patrol-rsi/runs/20261028-m12b')
for t in range(3):
    a=json.loads((b/f'audit-{t}.json').read_text());s=json.loads((b/f'trial-{t}/summary.json').read_text())
    assert a['all_actual_cases_replayed']==s['unique_cases'] and a['actual_model_calls']==39 and a['maximum_exact_gap_ratio']<=1
PY
python "$b/analysis-code-v1/scripts/report_m12b.py" --results "$b" --old-review "$b/m12a-objective-review.json" --output "$b/report"
