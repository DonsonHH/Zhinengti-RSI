#!/usr/bin/env bash
#SBATCH --job-name=rsi-m12g-integration
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
/share/home/renly/renly-patrol-rsi/runs/20261026-m1/model-env/bin/python - "$output" <<'PY'
import pathlib,json,hashlib,unittest,sys,torch,transformers
torch.set_num_threads(1);root=pathlib.Path.cwd();dest=pathlib.Path(sys.argv[1]);dest.mkdir()
names=('timing_skill','timing_semantics','timing_skill_features','timing_learning','g_timing_patch','g_bridge','g_feedback')
suite=unittest.TestSuite()
for name in names:suite.addTests(unittest.defaultTestLoader.discover('tests',pattern=f'test_{name}.py'))
with (dest/'tests.txt').open('x') as f:result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
import run_m12g_g0,run_m12g_precheck,g_timing_execution,finish_m12g_g0
value={'success':result.wasSuccessful(),'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'skipped':len(result.skipped),
'source_root':str(root),'test_source_SHA256':{f'tests/test_{name}.py':hashlib.sha256((root/'tests'/f'test_{name}.py').read_bytes()).hexdigest() for name in names},
'torch':torch.__version__,'transformers':transformers.__version__,'runtime_imports_passed':True,'new_model_calls':0,'new_environment_episodes':0,
'scope':'Targeted new patch/feedback/bridge seams plus inherited timing interfaces; not a repeat of full F0 or old341test suite.'}
(dest/'complete.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value),flush=True)
assert value['success'] and value['tests']>=29 and not(value['errors'] or value['failures'] or value['skipped'])
PY
