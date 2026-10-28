import pathlib,json,gzip,hashlib,runpy,torch
from patrol_rsi.m11 import run_episode,frozen_predictor
from patrol_rsi.inspection_model import TaskPolicy
root=pathlib.Path('/share/home/renly/renly-patrol-rsi/runs/20261028-m111');old=root.parent/'20261027-m11'
torch.set_num_threads(1)
functions=runpy.run_path(str(old/'analysis-code-v1/scripts/audit_m11.py'))
selected=json.loads((old/'report-129954/demo-selection.json').read_text())['case_names'];assert len(selected)==4
predictor=frozen_predictor(old/'frozen-estimator.pt',old/'code-4538d789/research/2026-10-27-m11-scheduling/frozen_estimator.json')
verified=[]
for name in selected:
 path=old/'evaluation-129953/cases'/f'{name}.json.gz';record=json.loads(gzip.decompress(path.read_bytes()));spec=record['specification'];model=None
 if spec['method'].startswith('ppo'):
  folder=old/'policies-v2'/f"{spec['estimator']}-seed-{spec['training_seed']}";metadata=json.loads((folder/'training.json').read_text());checkpoint=folder/'selected.pt'
  assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==metadata['checkpoint_sha256']
  model=TaskPolicy();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict']);model.eval()
 repeated,env,_=run_episode(spec,predictor,spec['method'],model)
 functions['equivalent'](repeated,record);functions['independent_schedule'](record,env);functions['independent_events'](record,env)
 verified.append({'case':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'actions':len(record['action_log']),'completed':len(record['inspection_events'])})
 print('M11_REVIEW_CASE_PASSED',name,flush=True)
with (root/'m11-targeted-review.json').open('x') as f:json.dump({'all_four_declared_demo_cases_replayed':True,'actual_selected_policy_read':True,'independent_clock_route_qualification_and_event_checks':True,'cases':verified,'scope':'Adds targeted replay to existing full1136case audit and fresh local source/transport/statistic verification; does not claim a second full1136case run'},f,indent=2)
