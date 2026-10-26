"""Review stored M1 trajectories without feeding these old test cases to M2."""
import argparse
import json
from pathlib import Path
import numpy as np
from patrol_rsi.environment import PatrolEnv
from patrol_rsi.experiment import room
from patrol_rsi.revisit import trajectory_diagnostics

p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();records=[]
for seed in range(3):
    for method in ['learned','route_greedy','local_greedy','random']:
        x=json.loads((a.evidence/f'baseline/seed-{seed}/test-example-{method}.json').read_text())
        c=x['case'];env=PatrolEnv(room(c['layout_seed']),c['field_seed'])
        age=[env.raw_age[env.scenario.targets].copy()];positions=[env.positions.copy()]
        for event in x['events']:
            env.step(np.asarray(event['actions']))
            age.append(env.raw_age[env.scenario.targets].copy());positions.append(env.positions.copy())
        records.append({'policy_seed':seed,'method':method,'case':c,'target_coverage':env.evaluation()['target_coverage'],
                        'peak_age':int(np.max(age)),'overdue_fraction':float((np.array(age[1:])>24).mean()),
                        **{tool:trajectory_diagnostics(env,age,positions,tool) for tool in ['coverage','revisit','oscillation']}})
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps({'purpose':'exploratory_diagnosis_of_preselected_M1_examples_not_new_test_selection','records':records},indent=2))
for r in records:print(r['policy_seed'],r['method'],r['target_coverage'],r['peak_age'],r['overdue_fraction'])
