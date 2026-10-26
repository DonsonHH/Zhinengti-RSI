"""Recompute finite-space and transferred-protocol experiment evidence."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np

from patrol_rsi.environment import PatrolEnv
from patrol_rsi.experiment import room
from patrol_rsi.m2 import DiagnosticPlan, feedback, rank
from patrol_rsi.m3 import METHODS, config_space, contract, source_plan
from patrol_rsi.revisit import PatrolSkill


def check_metrics(metrics, rows):
    for key, value in metrics.items():
        assert np.isclose(value, np.mean([r[key] for r in rows])), key


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();enumerations={};audit=[];calls=0;slots=0;test_records=0
    for seed in range(3):
        root=a.results/f'enumeration/policy-{seed}'
        rows=[json.loads(line) for line in (root/'configurations.jsonl').read_text().splitlines()]
        assert len(rows)==110 and {PatrolSkill(**r['skill']).version_id for r in rows}=={s.version_id for s in config_space()}
        for r in rows:
            check_metrics(r['metrics'],r['cases'])
            assert len(r['cases'])==6 and all(c['split']=='m3_development' for c in r['cases'])
        selected=json.loads((root/'selection-frozen.json').read_text())
        assert selected['selected']==min(rows,key=lambda r:rank(r['metrics']))
        assert not selected['test_used_for_selection']
        enumerations[seed]={PatrolSkill(**r['skill']).version_id:r for r in rows}
    for trial in range(9):
        root=a.results/f'trial-{trial}';s=json.loads((root/'summary.json').read_text())
        selections=json.loads((root/'selection-frozen.json').read_text())
        assert selections['optimizer_selections']==s['retained'] and not s['test_used_for_selection']
        src=json.loads((a.source/f'trial-{trial}/summary.json').read_text());plan=source_plan(src)
        transfer=json.loads((root/'transferred-protocol.json').read_text())
        assert transfer['plan']==asdict(plan) and transfer['task_config_reset']==asdict(PatrolSkill()) and transfer['history_reset']==[]
        assert set(transfer['plan'])=={'tool','instruction'}
        for method in METHODS:
            active=plan if method=='transferred_protocol' else DiagnosticPlan('revisit','Use raw-age diagnostics to propose one testable configuration change.') if method=='fixed_revisit' else DiagnosticPlan()
            parent=PatrolSkill();score=s['initial_development'];history=[]
            for generation in range(6):
                r=json.loads((root/f'{method}-{generation}.json').read_text());slots+=1
                assert r['parent']==asdict(parent) and r['plan']==asdict(active)
                if method!='random_search':
                    calls+=1;u=r['prompt']
                    assert u['prior_task_feedback']==history[-3:] and u['diagnostic_plan']==asdict(active)
                    assert u['current_skill']==asdict(parent)
                    assert all(d['tool']==active.tool and 8000<=d['layout_seed']<8006 for d in u['diagnostic_results'])
                    assert 'JSON_shape_example_only' not in u and 'JSON_shape_example_only' not in r['rendered_prompt']
                    assert 'patch' not in u and 'next_diagnostic' not in u
                    assert all(h['generation']<generation and set(h)<={'generation','patch','metrics','accepted','error'} for h in u['prior_task_feedback'])
                    assert r['input_tokens']<=4096 and r['output_tokens']<=512 and r['raw_response']
                if 'error' not in r:
                    candidate=PatrolSkill(**r['candidate']);check_metrics(r['metrics'],r['cases'])
                    anchor=enumerations[trial//3][candidate.version_id]['metrics']
                    assert all(np.isclose(value,anchor[k]) for k,value in r['metrics'].items())
                    assert r['accepted']==(rank(r['metrics'])<rank(score))
                    if r['accepted']:parent=candidate;score=r['metrics']
                history.append(feedback(r))
            assert asdict(parent)==s['retained'][method]['skill']
            assert asdict(active)==s['retained'][method]['plan']
        for method, group in s['test'].items():
            check_metrics(group['metrics'],group['cases']);assert len(group['cases'])==24
            for r in group['cases']:
                assert r['split']=='m3_test' and 9000<=r['layout_seed']<9012
                ages=np.asarray(r['target_ages']);assert ages.shape==(97,8)
                env=PatrolEnv(room(r['layout_seed'],r['noise']),r['field_seed'])
                initial=env.measured[env.scenario.targets]
                coverage=(initial|(ages[1:]==0).any(0)).mean()
                assert np.isclose(r['target_coverage'],coverage)
                assert np.isclose(r['mean_overdue_fraction'],(ages[1:]>24).mean())
                assert r['peak_unobserved_steps']==ages.max()
                test_records+=1
        audit.append({'trial':trial,'configuration_and_history_reset':True,'only_source_protocol_transferred':True,
                      'fixed_protocol_not_changed_during_adaptation':True,'candidate_metrics_match_enumeration':True,
                      'selection_chain_and_test_metrics_recomputed':True})
    a.output.write_text(json.dumps({'enumerated_configurations_per_policy':110,'enumeration_development_rollouts':1980,
        'candidate_slots':slots,'real_model_calls':calls,'test_method_case_records':test_records,'audits':audit},indent=2))
    print('Verified: 110x3 configurations,',slots,'candidate slots,',calls,'model calls,',test_records,'test records')


if __name__=='__main__':main()
