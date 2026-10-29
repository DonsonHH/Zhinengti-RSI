#!/usr/bin/env bash
#SBATCH --job-name=rsi-b4b0-gate
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
source_root="$1"
output_root="$2"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$source_root/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$source_root"
test ! -e "$output_root/gate"
mkdir "$output_root/gate"
python - "$output_root/gate" <<'PY'
import json,sys,unittest,torch
from pathlib import Path
torch.set_num_threads(1)
root=Path(sys.argv[1])
with (root/'full-tests.txt').open('x') as stream:
    r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover('tests'))
result={'tests':r.testsRun,'skipped':len(r.skipped),'failures':len(r.failures),'errors':len(r.errors),'success':r.wasSuccessful(),'torch':torch.__version__}
(root/'full-tests.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
if not result['success'] or result['tests']<235 or result['skipped']:raise SystemExit(1)
PY
