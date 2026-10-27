import pathlib,json,gzip,runpy,torch
from patrol_rsi.m11 import run_episode,frozen_predictor
from patrol_rsi.inspection_model import TaskPolicy
root=pathlib.Path('/share/home/renly/renly-patrol-rsi/runs/20261027-m11');torch.set_num_threads(1)
f=runpy.run_path(str(root/'analysis-code-v1/scripts/audit_m11.py'))
predictor=frozen_predictor(root/'frozen-estimator.pt',root/'code-4538d789/research/2026-10-27-m11-scheduling/frozen_estimator.json')
names=['main--128--provisioned--open--healthy--feedback--3000--nominal--cycle--None','main--128--provisioned--open--coldplate_fouling--fixed_commands--3001--history--ppo_guard--0']
for name in names:
 record=json.loads(gzip.decompress((root/'evaluation-129953/cases'/f'{name}.json.gz').read_bytes()));spec=record['specification'];model=None
 if spec['method'].startswith('ppo'):
  pack=torch.load(root/'policies-v2'/f"{spec['estimator']}-seed-{spec['training_seed']}"/'selected.pt',map_location='cpu',weights_only=True);model=TaskPolicy();model.load_state_dict(pack['state_dict']);model.eval()
 repeated,env,_=run_episode(spec,predictor,spec['method'],model)
 f['equivalent'](repeated,record);f['independent_schedule'](record,env);f['independent_events'](record,env)
 print('M11_AUDIT_PREFLIGHT_OK',name,flush=True)
(root/'audit-preflight.json').write_text(json.dumps({'cases':names,'actual_policy_route_clock_and_event_checks_passed':True},indent=2)+'\n')
