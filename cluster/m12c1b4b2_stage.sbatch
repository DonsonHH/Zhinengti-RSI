#!/usr/bin/env bash
#SBATCH --job-name=rsi-b2-development
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
stage="$source_root/research/2026-10-30-m12c1b4b2-alignment"
old="$source_root/research/2026-10-30-m12c1b4b0-data"
prior=/share/home/renly/renly-patrol-rsi/runs/20261030-m12c1b4b1a
limited=/share/home/renly/renly-patrol-rsi/runs/20261030-m12c1b4b1b
if [[ "$phase" == gate ]];then
  mkdir "$run_root/gate"
  python - "$run_root/gate" <<'PY'
import json,sys,unittest,torch
from pathlib import Path
torch.set_num_threads(1);root=Path(sys.argv[1])
with (root/'full-tests.txt').open('x') as stream:
 r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover('tests'))
result={'tests':r.testsRun,'skipped':len(r.skipped),'failures':len(r.failures),'errors':len(r.errors),'success':r.wasSuccessful(),'torch':torch.__version__}
(root/'full-tests.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
if not result['success'] or result['tests']<274 or result['skipped']:raise SystemExit(1)
PY
  exit
fi
python - "$run_root/gate/full-tests.json" <<'PY'
import json,sys
g=json.load(open(sys.argv[1]));assert g['success'] and g['tests']>=274 and not(g['skipped'] or g['failures'] or g['errors'])
PY
common=(--protocol "$stage/B4-B2-development-v1.json" --review "$stage/evidence/protocol-review.json")
case "$phase" in
  calibration)
    python scripts/diagnose_b2_calibration.py "${common[@]}" \
      --old-protocol "$source_root/research/2026-10-30-m12c1b4b1b-limited/B4-B1b-Limited-v1.json" \
      --cache "$limited/cache" --fits "$limited/fits" --output "$run_root/calibration"
    ;;
  actions)
    python scripts/diagnose_b2_actions.py "${common[@]}" --old-manifest "$old/B4-B1新数据清单.json" \
      --source-root "$prior/data" --source-inventory "$prior/report/raw-storage.json" --seed-index "${4:-${SLURM_ARRAY_TASK_ID}}" \
      --output "$run_root/actions/seed-index-${4:-${SLURM_ARRAY_TASK_ID}}"
    ;;
  *) exit 2;;
esac
