"""Independent M5 validation selection, paired initialization and trace audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from patrol_rsi.deadline import target_deadlines
from patrol_rsi.environment import PatrolEnv
from patrol_rsi.experiment import room
from patrol_rsi.m5 import METRICS,VARIANTS,contract,rank


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();c=contract();audits=[];initial={};geometries={};replayed={};count=0
    for variant in VARIANTS:
        for seed in range(3):
            root=a.results/f'{variant}-seed-{seed}'
            s=json.loads((root/'summary.json').read_text());curves=[json.loads(line) for line in (root/'learning.jsonl').read_text().splitlines()]
            assert not s['smoke'] and s['seed']==seed and s['variant']==variant
            assert json.loads((root/'contract.json').read_text())==c
            assert s['episodes']==2400 and s['steps']==230400 and s['ddqn_updates']==57585
            assert len(curves)==12 and [r['episode'] for r in curves]==list(range(200,2401,200))
            selected=min(curves,key=lambda r:rank(r['validation']))
            assert s['selected_episode']==selected['episode'] and s['best_validation']==selected['validation']
            assert s['fresh_checkpoint_trace_replay_matches'] and s['replay_case_split']=='m5_test'
            assert not s['test_used_for_selection'] and not s['rsi_enabled']
            assert s['initial_parameter_sha256']!=s['selected_parameter_sha256']
            if seed in initial:assert s['initial_parameter_sha256']==initial[seed]
            else:initial[seed]=s['initial_parameter_sha256']
            checkpoint=root/'best-policy.pt'
            if checkpoint.exists():assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==s['checkpoint_sha256']
            expected_methods={'learned','guard12'}
            if variant=='deadline_vae':expected_methods|={'human_workflow','deadline_planner','random','testtime_no_vae','untrained_network','old_m1'}
            assert set(s['test'])==expected_methods
            for method,group in s['test'].items():
                assert len(group['cases'])==48
                for k,v in group['metrics'].items():assert np.isclose(v,np.mean([r[k] for r in group['cases']])),(variant,seed,method,k)
                for index,row in enumerate(group['cases']):
                    case={k:row[k] for k in ['split','layout_seed','field_seed','profile','noise']}
                    assert case==c['test'][index]
                    due=target_deadlines(case['layout_seed'],case['profile']);ages=np.asarray(row['target_ages']);positions=np.asarray(row['positions'])
                    assert ages.shape==(97,8) and positions.shape==(97,4,2) and row['target_deadlines']==due.tolist()
                    assert len(row['actions'])==96 and len(row['prediction_mse_series'])==96
                    actions_array=np.asarray(row['actions']);assert actions_array.shape==(96,4) and np.all((actions_array>=-1)&(actions_array<8))
                    from patrol_rsi.environment import MOVES
                    deltas=np.zeros((96,4,2),dtype=int);moving=actions_array>=0;deltas[moving]=MOVES[actions_array[moving]]
                    assert np.array_equal(positions[1:],positions[:-1]+deltas)
                    assert np.isclose(row['deadline_violation'],(ages[1:]>due).mean())
                    assert np.isclose(row['deadline_violation'],np.mean(row['deadline_violation_series']))
                    assert np.isclose(row['weighted_idleness'],np.mean(row['weighted_idleness_series']))
                    assert np.isclose(row['target_prediction_mse'],np.mean(row['prediction_mse_series']))
                    assert row['peak_age']==int(ages.max())
                    if variant=='no_vae' or method=='testtime_no_vae':assert row['estimator_calls']==0
                    else:assert row['estimator_calls']==96
                    if method not in {'guard12','human_workflow','deadline_planner'}:assert row['intervention_fraction']==0
                    if method=='deadline_planner':assert row['intervention_fraction']==1
                    key=json.dumps(case,sort_keys=True)+json.dumps(row['positions'])
                    age_hash=hashlib.sha256(ages.tobytes()).hexdigest()
                    if key not in replayed or index==0:
                        env=PatrolEnv(room(case['layout_seed'],case['noise']),case['field_seed'])
                        env._neighborhoods=geometries.setdefault(case['layout_seed'],{})
                        assert np.array_equal(env.positions,positions[0]) and np.array_equal(env.raw_age[env.scenario.targets],ages[0])
                        true_idle=[]
                        for t,actions in enumerate(row['actions']):
                            if index==0:
                                q=np.asarray(row['action_scores_first_case_only'][t]);assert q.shape==(4,8)
                                assert np.array_equal(env.safe_actions(q),actions)
                            env.step(np.asarray(actions,dtype=int))
                            assert np.array_equal(env.positions,positions[t+1]) and np.array_equal(env.raw_age[env.scenario.targets],ages[t+1])
                            true_idle.append(env.evaluation()['normalized_weighted_idleness'])
                        replayed[key]={'ages':age_hash,'coverage':env.evaluation()['target_coverage'],
                                       'weighted':float(np.mean(true_idle)),'distance':env.evaluation()['total_movement_distance']}
                    reference=replayed[key]
                    assert reference['ages']==age_hash
                    assert np.isclose(row['target_coverage'],reference['coverage'])
                    assert np.isclose(row['weighted_idleness'],reference['weighted'])
                    assert np.isclose(row['movement_distance'],reference['distance'])
                    count+=1
            if variant=='no_deadline':
                # With identical public observations and no deadline input,
                # changing the evaluation deadline must not alter actions.
                for method in ['learned','guard12']:
                    rows=s['test'][method]['cases']
                    balanced={(r['layout_seed'],r['noise']):r for r in rows if r['profile']=='balanced'}
                    for row in rows:
                        if row['profile']=='tight':assert row['positions']==balanced[(row['layout_seed'],row['noise'])]['positions']
            audits.append({'variant':variant,'seed':seed,'selected_episode':s['selected_episode'],
                'validation_only_checkpoint_selection_verified':True,'parameter_update_and_fresh_checkpoint_replay_verified':True,
                'all_metrics_and_traces_verified':True,'test_methods':len(s['test']),
                'initial_parameter_sha256':s['initial_parameter_sha256'],'selected_checkpoint_sha256':s['checkpoint_sha256']})
    # Pure geometry planner and random-action controls do not depend on the
    # retrained network. Their repeated copies must be exactly identical.
    full=[json.loads((a.results/f'deadline_vae-seed-{seed}/summary.json').read_text()) for seed in range(3)]
    for method in ['deadline_planner','random']:
        assert full[0]['test'][method]['metrics']==full[1]['test'][method]['metrics']==full[2]['test'][method]['metrics']
    result={'from_scratch_training_tasks':12,'independent_random_seeds_per_variant':3,'policy_initializations_paired_across_variants':True,
        'test_method_case_records':count,'unique_geometric_trajectories_replayed':len(replayed),
        'team_training_steps':sum(230400 for _ in audits),'ddqn_updates':sum(57585 for _ in audits),
        'audits':audits,'pure_planner_and_random_controls_repeat_exactly':True}
    a.output.write_text(json.dumps(result,indent=2));print('M5 AUDIT VERIFIED',len(audits),'runs,',count,'test records,',len(replayed),'distinct traces')


if __name__=='__main__':main()
