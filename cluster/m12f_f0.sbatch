#!/usr/bin/env bash
#SBATCH --job-name=rsi-m12f-F0-parity
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
source_root="$1"
run_root="$2"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$source_root/src:$source_root/scripts"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$source_root"
/share/home/renly/renly-patrol-rsi/runs/20261026-m1/model-env/bin/python - "$run_root" <<'PY'
import json,os,sys,time
from pathlib import Path
with (Path(sys.argv[1])/'F0-launch.json').open('x') as f:
    json.dump({'job_id':os.environ['SLURM_JOB_ID'],'unix':time.time(),'one_launch_only':True},f)
PY
/share/home/renly/renly-patrol-rsi/runs/20261026-m1/model-env/bin/python scripts/run_m12f_f0.py --source-root "$source_root" --run-root "$run_root"
