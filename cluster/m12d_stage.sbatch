#!/usr/bin/env bash
#SBATCH --job-name=rsi-m12d
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
source_root="$1"
run_root="$2"
phase="$3"
umask 077
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$source_root/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$source_root"
stage="$source_root/research/2026-10-30-m12d-formal"
prior="$source_root/research/2026-10-30-m12c1b4b2-alignment"
if [[ "$phase" == gate ]];then
  mkdir "$run_root/gate"
  python - "$run_root/gate" <<'PY'
import json,sys,unittest,torch
from pathlib import Path
torch.set_num_threads(1);root=Path(sys.argv[1])
with (root/'full-tests.txt').open('x') as f:r=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.discover('tests'))
v={'tests':r.testsRun,'skipped':len(r.skipped),'errors':len(r.errors),'failures':len(r.failures),'success':r.wasSuccessful(),'torch':torch.__version__}
(root/'full-tests.json').write_text(json.dumps(v,indent=2))
print(json.dumps(v),flush=True)
assert v['success'] and v['tests']>=286 and not(v['skipped'] or v['errors'] or v['failures'])
PY
  exit
fi
python scripts/run_m12d_paired.py --paired "$prior/MultiAgent-Patrol-Paired-v1.json" --rsi "$prior/RSI-Controlled-Meta-v1.json" --manifest "$stage/process-manifest.json" --review "$stage/evidence/execution-review.json" --gate "$run_root/gate/full-tests.json" --run-root "$run_root" --actors /share/home/renly/renly-patrol-rsi/runs/20261027-m11/policies-v2 --estimator-weights /share/home/renly/renly-patrol-rsi/runs/20261028-m111/frozen-estimator.pt --estimator-manifest "$source_root/research/2026-10-28-m111-coverage/frozen_estimator.json" --phase "$phase" --seed-index "$4"
