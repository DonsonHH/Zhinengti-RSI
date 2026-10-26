"""Audit stored M2 evidence without invoking policies or selecting candidates."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np

from patrol_rsi.environment import PatrolEnv
from patrol_rsi.experiment import room
from patrol_rsi.m2 import DiagnosticPlan, feedback, rank
from patrol_rsi.revisit import PatrolSkill


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audits=[];total_model_calls=0;total_candidates=0
    for trial in range(9):
        root=a.results/f'trial-{trial}';s=json.loads((root/'summary.json').read_text())
        contract=json.loads((root/'contract.json').read_text())
        selections=json.loads((root/'selection-frozen.json').read_text())
        assert selections==s['retained']
        for method in contract['methods']:
            parent=PatrolSkill();score=s['initial_development'];plan=DiagnosticPlan();history=[]
            for generation in range(6):
                r=json.loads((root/f'{method}-{generation}.json').read_text());total_candidates+=1
                assert r['parent']==asdict(parent) and r['diagnostic_plan']==asdict(plan)
                if method!='random_search':
                    total_model_calls+=1
                    user=r['user_prompt'];assert user['history']==history[-3:]
                    assert user['active_diagnostic_plan']==asdict(plan)
                    assert all(d['tool']==plan.tool for d in user['diagnostic_tool_results'])
                    assert all(6000<=d['layout_seed']<6006 for d in user['diagnostic_tool_results'])
                    assert r['raw_response'] and r['rendered_prompt']
                    assert r['input_tokens']<=4096 and r['output_tokens']<=512
                if 'error' not in r:
                    candidate=PatrolSkill(**r['candidate'])
                    assert all(c['split']=='development' and 6000<=c['layout_seed']<6006 for c in r['cases'])
                    assert r['accepted']==(rank(r['metrics'])<rank(score))
                    if method=='updated_improver':plan=DiagnosticPlan(**r['next_diagnostic_plan'])
                    if r['accepted']:parent=candidate;score=r['metrics']
                history.append(feedback(r))
            assert asdict(parent)==selections[method]['skill']
            assert asdict(plan)==selections[method]['final_diagnostic_plan']
        cases_checked=0
        for method,t in s['test'].items():
            for row in t['cases']:
                env=PatrolEnv(room(row['layout_seed'],row['noise']),row['field_seed'])
                initial=env.measured[env.scenario.targets]
                ages=np.array(row['target_ages']);ever=initial|(ages[1:]==0).any(0)
                assert np.isclose(row['target_coverage'],ever.mean())
                assert row['never_observed_targets']==int((~ever).sum())
                assert row['max_unobserved_steps']==int(ages[-1].max())
                assert row['peak_unobserved_steps']==int(ages.max())
                assert np.isclose(row['mean_overdue_fraction'],(ages[1:]>24).mean())
                assert row['split']=='m2_test' and 7000<=row['layout_seed']<7024
                cases_checked+=1
        audits.append({'trial':trial,'test_case_method_records_checked':cases_checked,'complete_parent_selection_chain_verified':True,
                       'prompt_history_exactly_prior_feedback':True,'actual_active_diagnostic_tools_verified':True,
                       'test_age_coverage_and_overdue_metrics_recomputed':True})
    a.output.write_text(json.dumps({'total_candidate_slots':total_candidates,'total_real_model_calls':total_model_calls,
                                   'paired_trials':9,'independent_policy_training_seeds':3,
                                   'audits':audits},indent=2));print(total_candidates,total_model_calls,'all selection chains and traces verified')


if __name__=='__main__':main()
