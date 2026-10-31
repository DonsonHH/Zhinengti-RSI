#!/usr/bin/env bash
#SBATCH --job-name=rsi-m12e-E1-timing
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
source_root="$1"
run_root="$2"
seed_index="$3"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$source_root/src:$source_root/scripts"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$source_root"
set +e
python scripts/run_m12e_timing.py --source-root "$source_root" --run-root "$run_root" --seed-index "$seed_index"
result=$?
set -e
if [[ "$result" -ne 0 ]];then
  python - "$run_root" "$SLURM_JOB_ID" "$result" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1])/'E1-STOP.json'
try:
    with p.open('x') as f:json.dump({'stop':True,'failed_job':sys.argv[2],'exit_code':int(sys.argv[3]),'unix':time.time(),'no_free_retry':True},f,indent=2)
except FileExistsError:pass
PY
fi
exit "$result"
