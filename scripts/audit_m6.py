"""M6 choice masks, commitments, common navigation and independent metrics."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np

from patrol_rsi.dispatch import DispatchEnv
from patrol_rsi.m6 import VARIANTS,contract,rank
from m6_evidence_io import read_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();c=contract();audits=[];initial={};replayed={};count=0
    expected_methods={'learned','slack','risk_age_distance','random_targets','untrained_network'}
    for variant in VARIANTS:
        for seed in range(3):
            root=a.results/f'{variant}-seed-{seed}';s=read_json(root/'summary.json')
            curve=[json.loads(x) for x in (root/'learning.jsonl').read_text().splitlines()]
            assert s['seed']==seed and s['variant']==variant and not s['preflight'] and not s['smoke']
            assert json.loads((root/'contract.json').read_text())==c and s['episodes']==1200
            assert s['gradient_updates']==3600 and s['behavior_logp_max_error']<=2e-5
            assert len(curve)==12 and [r['episode'] for r in curve]==list(range(100,1201,100))
            selected=min(curve,key=lambda r:rank(r['validation']))
            assert s['selected_episode']==selected['episode'] and s['best_validation']==selected['validation']
            assert s['fresh_checkpoint_trace_replay_matches'] and not s['test_used_for_selection']
            assert s['initial_parameter_sha256']!=s['selected_parameter_sha256']
            if seed in initial:assert s['initial_parameter_sha256']==initial[seed]
            else:initial[seed]=s['initial_parameter_sha256']
            path=root/'best-policy.pt'
            if path.exists():assert hashlib.sha256(path.read_bytes()).hexdigest()==s['checkpoint_sha256']
            assert set(s['test'])==expected_methods
            for method,group in s['test'].items():
                assert len(group['cases'])==48
                for key,value in group['metrics'].items():assert np.isclose(value,np.mean([r[key] for r in group['cases']]))
                for i,row in enumerate(group['cases']):
                    case={k:row[k] for k in ['split','layout_seed','field_seed','profile','noise']}
                    assert case==c['test'][i] and row['variant']==variant
                    assert len(row['events'])==96 and len(row['positions'])==97 and len(row['target_ages'])==97
                    ages=np.asarray(row['target_ages']);due=np.asarray(row['target_deadlines'])
                    assert np.isclose(row['deadline_violation'],(ages[1:]>due).mean()) and row['peak_age']==ages.max()
                    assert np.isclose(row['weighted_idleness'],np.mean(row['weighted_idleness_series']))
                    assert np.isclose(row['target_prediction_mse'],np.mean(row['prediction_mse_series']))
                    assert row['estimator_calls']==(0 if variant=='no_vae' else 96)
                    coordinated=variant!='independent_targets' if method in ('learned','untrained_network') else True
                    # Trajectory equality includes the chosen targets, not only
                    # positions: different commands can happen to follow a path.
                    cachekey=json.dumps(case,sort_keys=True)+str(coordinated)+json.dumps([e['choices'] for e in row['events']])
                    signature=hashlib.sha256(json.dumps({'positions':row['positions'],'ages':row['target_ages'],
                        'events':[{'actions':e['actions'],'assigned_targets':e['assigned_targets'],'public_reward':e['public_reward']} for e in row['events']]}).encode()).hexdigest()
                    if cachekey not in replayed:
                        wrapper=DispatchEnv(case);env=wrapper.env;idle=[];rewards=[]
                        assert np.array_equal(env.positions,row['positions'][0])
                        for t,event in enumerate(row['events']):
                            state=wrapper.prepare();assert state['free'].tolist()==event['free_executors'] and state['order'].tolist()==event['order']
                            reward,repeated=wrapper.step(np.asarray(event['choices'],dtype=int),state,coordinated)
                            for key,value in repeated.items():
                                if isinstance(value,float):assert np.isclose(value,event[key]),(variant,seed,method,t,key)
                                else:assert value==event[key],(variant,seed,method,t,key)
                            assert np.array_equal(env.positions,row['positions'][t+1])
                            assert np.array_equal(env.raw_age[env.scenario.targets],row['target_ages'][t+1])
                            idle.append(env.evaluation()['normalized_weighted_idleness']);rewards.append(reward)
                        replayed[cachekey]={'signature':signature,'coverage':env.evaluation()['target_coverage'],
                            'weighted':float(np.mean(idle)),'cost':-float(np.mean(rewards)),
                            'movement':env.evaluation()['total_movement_distance'],'assignments':wrapper.assignments,
                            'duplicates':wrapper.duplicate_assignments/max(1,wrapper.assignments)}
                    ref=replayed[cachekey];assert ref['signature']==signature
                    for key,source in [('target_coverage','coverage'),('weighted_idleness','weighted'),('public_cost','cost'),
                                       ('movement_distance','movement'),('target_assignments','assignments'),('duplicate_assignment_fraction','duplicates')]:
                        assert np.isclose(row[key],ref[source]),(variant,seed,method,key)
                    if coordinated:assert row['duplicate_assignment_fraction']==0
                    count+=1
            audits.append({'variant':variant,'seed':seed,'selected_episode':s['selected_episode'],
                'validation_selection_verified':True,'joint_behavior_likelihood_verified_during_all_updates':True,
                'choices_commitment_release_and_common_navigation_replayed':True,'fresh_selected_checkpoint_trace_matches':True})
    # Same common heuristics must not depend on random policy initialization.
    for variant in VARIANTS:
        seeds=[read_json(a.results/f'{variant}-seed-{i}/summary.json') for i in range(3)]
        for method in ('slack','risk_age_distance','random_targets'):
            assert seeds[0]['test'][method]['metrics']==seeds[1]['test'][method]['metrics']==seeds[2]['test'][method]['metrics']
    joint=read_json(a.results/'joint_vae-seed-0/summary.json')
    independent=read_json(a.results/'independent_targets-seed-0/summary.json')
    for method in ('slack','risk_age_distance','random_targets'):
        assert joint['test'][method]['metrics']==independent['test'][method]['metrics']
    result={'from_scratch_training_tasks':9,'paired_random_seeds_per_variant':3,'same_seed_initializations_match':True,
        'test_method_case_records':count,'distinct_case_and_target_choice_sequences_replayed':len(replayed),
        'team_training_steps':9*1200*96,'gradient_updates':9*3600,'audits':audits,
        'common_heuristics_independent_of_policy_weights':True}
    a.output.write_text(json.dumps(result,indent=2));print('M6 AUDIT VERIFIED',len(audits),'tasks;',count,'test records;',len(replayed),'choice sequences')


if __name__=='__main__':main()
