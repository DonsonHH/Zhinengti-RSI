#!/usr/bin/env bash
#SBATCH --job-name=rsi-m12f-token-profile
#SBATCH --partition=cu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
set -euo pipefail
source_root="$1"
run_root="$2"
output="$3"
module purge
module load python/3.11-pytorch2.6-cu124
export PYTHONPATH="$source_root/src:$source_root/scripts"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$source_root"
/share/home/renly/renly-patrol-rsi/runs/20261026-m1/model-env/bin/python - "$source_root" "$run_root" "$output" <<'PY'
import ast,copy,gzip,hashlib,json,sys
from pathlib import Path
from transformers import AutoTokenizer
from patrol_rsi.timing_skill import TimingSkill
from patrol_rsi.timing_learning import skill_context,skill_prompt,retrieve_cards
from patrol_rsi.rsi_controlled import ControlledImprover
source,root,output=map(Path,sys.argv[1:]);output.mkdir()
model=Path('/share/home/renly/renly-patrol-rsi/models/Qwen3-8B-b968826d')
tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
tree=ast.parse((source/'scripts/timing_skill_runtime.py').read_text())
system=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='system' for v in n.targets))
cards=[]
for row in json.loads((root/'F0/complete.json').read_bytes())['rows']:
 record=json.loads(gzip.decompress(Path(row['record_path']).read_bytes()))
 cards.extend(retrieve_cards(record['causal_experience_cards'],ControlledImprover().protocol.diagnostics))
cards=retrieve_cards(cards,ControlledImprover().protocol.diagnostics)
large={'schema_version':'timing-skill-v1','rules':[{'all':[{'field':'professional_available','op':'is_missing','value':None} for _ in range(6)]} for _ in range(3)],'default_action':'CONTINUE_TOUR'}
TimingSkill.from_dict(large)
history=[{'skill':large,'accepted':False,'cards':cards} for _ in range(2)]
results=[]
for length in (0,2):
 for prior in (True,False):
  context=skill_context(TimingSkill.initial_e1(),history if length else [],cards,prior_only=prior)
  prompt=skill_prompt(context);dropped=0;before=None
  while True:
   rendered=tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':json.dumps(prompt,ensure_ascii=False)}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   count=len(tokenizer(rendered)['input_ids'])
   if before is None:before=count
   if count<=6144:break
   if not prompt['data']['experience_cards']:raise ValueError('Static schema/parent/history exceeds6144 without cards')
   prompt['data']['experience_cards'].pop();dropped+=1
  results.append({'prior_only':prior,'history_length':length,'input_before_trim':before,'input_tokens':count,'dropped_last_cards':dropped,
   'visible_professions':sorted({v['measurement']['kind'] for v in prompt['data']['experience_cards']}),'actual_prompt':prompt})
value={'tokenizer_only':True,'model_generation_calls':0,'neural_inference':False,'new_environment_episodes':0,'new_physical_processes':0,
 'source_root':str(source),'runtime_source_SHA256':hashlib.sha256((source/'scripts/timing_skill_runtime.py').read_bytes()).hexdigest(),'results':results}
(output/'complete.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k!='actual_prompt'} for r in results]),flush=True)
PY
