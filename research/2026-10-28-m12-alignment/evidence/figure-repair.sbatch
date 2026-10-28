#!/usr/bin/env bash
#SBATCH --job-name=m12-figure-layout
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:03:00
set -euo pipefail
module purge
module load python/3.11-pytorch2.6-cu124
b=/share/home/renly/renly-patrol-rsi/runs/20261028-m12
python "$b/figure-code-v1/scripts/repair_m12_attribution.py" --aggregate "$b/report/aggregate.json" --output "$b/report-corrected"
python - <<'PY'
from pathlib import Path
import hashlib,json,os
b=Path('/share/home/renly/renly-patrol-rsi/runs/20261028-m12')
files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (b/'report-corrected').iterdir()}
value={'job_id':int(os.environ['SLURM_JOB_ID']),'source_aggregate_sha256':hashlib.sha256((b/'report/aggregate.json').read_bytes()).hexdigest(),'method':'Legend margins only, original figure retained; no case simulation or statistic change','files':files}
(b/'figure-repair.json').write_text(json.dumps(value,indent=2)+'\n')
PY
