#!/usr/bin/env bash
#SBATCH --job-name=rsi-m12e-Torch-gate
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
source_root="$1"
output="$2"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$source_root/src:$source_root/scripts"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$source_root"
mkdir "$output"
/share/home/renly/renly-patrol-rsi/runs/20261026-m1/model-env/bin/python - "$output" <<'PY'
import hashlib,json,sys,unittest,torch,transformers
from pathlib import Path
torch.set_num_threads(1);root=Path.cwd().resolve();dest=Path(sys.argv[1]);suite=unittest.defaultTestLoader.discover('tests')
def ids(s):
    if isinstance(s,unittest.TestSuite):return [x for child in s for x in ids(child)]
    return [s.id()]
names=ids(suite)
with (dest/'full-tests.txt').open('x') as f:result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
value={'tests':result.testsRun,'success':result.wasSuccessful(),'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
    'torch':torch.__version__,'transformers':transformers.__version__,'source_root':str(root),
    'E_test_sources':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('tests/test_meta_schema.py','tests/test_patrol_timing.py')},
    'E_test_ids':[v for v in names if 'test_meta_schema' in v or 'test_patrol_timing' in v]}
(dest/'full-tests.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps(value),flush=True)
assert value['success'] and value['tests']>=316 and len(value['E_test_ids'])==23 and not(value['failures'] or value['errors'] or value['skipped'])
PY
