#!/usr/bin/env bash
#SBATCH --job-name=rsi-m12d-guarded-final
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
source_root="$1"
run_root="$2"
trial="$3"
umask 077
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$source_root/src:$source_root/scripts"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python - "$run_root" "$trial" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
r=Path(sys.argv[1]).resolve();trial=int(sys.argv[2]);assert trial in (0,1,2)
resource=json.loads((r/'rsi-resource-before-final.json').read_bytes());guard=resource['guarded_final_wrapper']
p=Path(guard['path']);digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==guard['sha256']
seal=r/'rsi-final-seal.json';assert json.loads(seal.read_bytes())['all_three_adaptations_frozen']
root=r/f'rsi-trial-{trial}'
assert not(root/'final-started.json').exists() and not(root/'final-complete.json').exists()
claim=root/'final-one-shot-claim';claim.mkdir(mode=0o700,exist_ok=False)
receipt={'trial':trial,'resolved_run_root':str(r),'Slurm_job_id':os.environ['SLURM_JOB_ID'],
    'seal_SHA256':hashlib.sha256(seal.read_bytes()).hexdigest(),'guard_wrapper_SHA256':digest,
    'persistent_no_retry':True,'claimed_before_final_source_access':True}
with (claim/'claim.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2);f.write('\n')
PY
cd "$source_root"
python scripts/run_m12d_rsi_final.py --source-root "$source_root" --run-root "$run_root" --trial "$trial"
