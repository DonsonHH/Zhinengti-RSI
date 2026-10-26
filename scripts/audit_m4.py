"""Audit executable workflow/protocol selection and rederive test metrics."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np

from patrol_rsi.environment import MOVES, PatrolEnv
from patrol_rsi.experiment import room
from patrol_rsi.m4 import (ImproverProtocol, METRICS, OPTIMIZERS, contract, deadlines,
                           meta_accept, rank, select_memory, snapshot)
from patrol_rsi.workflow import Workflow, human_workflow


def check_mean(metrics,rows):
    for k,v in metrics.items():assert np.isclose(v,np.mean([r[k] for r in rows])),k


def audit_optimizer(directory,cases,protocol):
    selection=json.loads((directory/'selection-frozen.json').read_text())
    assert selection['selection_cases']==cases and selection['protocol']==snapshot(asdict(protocol))
    parent=human_workflow();score=selection['initial_metrics'];history=[];valid=0;calls=0
    for g in range(selection['slots']):
        r=json.loads((directory/f'candidate-{g}.json').read_text())
        assert r['generation']==g and r['parent_id']==parent.version_id and r['parent_workflow']==parent.to_dict()
        assert r['protocol']==snapshot(asdict(protocol))
        if r['origin']=='language_model':
            calls+=1;u=r['prompt']
            assert u['current_workflow']==parent.to_dict()
            assert u['memory']==select_memory(history,protocol.memory)
            assert u['improver_protocol']==snapshot(asdict(protocol))
            assert all(set(d)==set(protocol.diagnostics)|{'layout_seed','profile'} for d in u['tool_results'])
            assert all(d['layout_seed'] in {c['layout_seed'] for c in cases} for d in u['tool_results'])
            assert all(h['generation']<g for h in u['memory'])
            assert r['input_tokens']<=6144 and r['output_tokens']<=768
            assert r['raw_response'] and r['rendered_prompt']
        if 'error' in r:
            assert not r['accepted']
            history.append({'generation':g,'accepted':False,'error':r['error']});continue
        valid+=1;candidate=Workflow.from_dict(r['candidate_workflow'])
        assert candidate.version_id==r['candidate_id']
        check_mean(r['metrics'],r['cases'])
        assert [{k:row[k] for k in ['split','layout_seed','field_seed','profile','noise']} for row in r['cases']]==cases
        assert r['accepted']==(rank(r['metrics'])<rank(score))
        if r['accepted']:parent=candidate;score=r['metrics']
        history.append({'generation':g,'workflow':candidate.to_dict(),'metrics':r['metrics'],'accepted':r['accepted']})
    assert selection['workflow']==parent.to_dict() and selection['workflow_id']==parent.version_id and selection['metrics']==score
    assert calls==selection['model_calls'] and not selection['test_used_for_selection']
    return {'candidate_slots':selection['slots'],'valid':valid,'model_calls':calls}


def audit_meta(directory):
    selected=json.loads((directory/'protocol-frozen.json').read_text());parent=ImproverProtocol()
    total_calls=0;total_slots=0;accepted=0;valid=0;history=[]
    for r in range(len(selected['records'])):
        record=json.loads((directory/f'round-{r}.json').read_text())
        assert record['parent']==snapshot(asdict(parent))
        if record['proposal_origin']=='language_model':
            total_calls+=1;reply=record['model_proposal']
            assert reply['prompt']['previous_meta_feedback']==history
            assert reply['prompt']['current_protocol']==snapshot(asdict(parent))
            assert reply['input_tokens']<=6144 and reply['output_tokens']<=768
        if 'error' in record:
            assert not record['accepted'];history.append(record);continue
        valid+=1;candidate=ImproverProtocol.from_dict(record['candidate'])
        cases=contract()['meta_development'][r]
        before=audit_optimizer(directory/f'round-{r}-incumbent',cases,parent)
        after=audit_optimizer(directory/f'round-{r}-candidate',cases,candidate)
        assert record['incumbent_inner']==json.loads((directory/f'round-{r}-incumbent/selection-frozen.json').read_text())
        assert record['candidate_inner']==json.loads((directory/f'round-{r}-candidate/selection-frozen.json').read_text())
        total_calls+=before['model_calls']+after['model_calls'];total_slots+=before['candidate_slots']+after['candidate_slots']
        assert record['accepted']==meta_accept(record['incumbent_inner'],record['candidate_inner'])
        if record['accepted']:parent=candidate;accepted+=1
        history.append({'round':r,'parent':record['parent'],'candidate':record['candidate'],'accepted':record['accepted'],
                        'incumbent_metrics':record['incumbent_inner']['metrics'],'candidate_metrics':record['candidate_inner']['metrics']})
    assert selected['records']==history and selected['protocol']==snapshot(asdict(parent))
    return parent,{'rounds':len(history),'valid_meta_candidates':valid,'accepted_meta_updates':accepted,
                   'model_calls':total_calls,'inner_workflow_slots':total_slots}


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audits=[];neighborhoods={};replayed={};total_calls=0;slots=0;test_rows=0
    for trial in range(9):
        root=a.results/f'trial-{trial}';s=json.loads((root/'summary.json').read_text())
        assert s['trial']==trial and s['policy_seed']==trial//3 and s['proposal_seed']==contract()['proposal_seeds'][trial%3]
        random_p,random_a=audit_meta(root/'meta-random');model_p,model_a=audit_meta(root/'meta-model')
        assert s['protocols']['validated_random_meta']==snapshot(asdict(random_p))
        assert s['protocols']['validated_model_meta']==snapshot(asdict(model_p))
        frozen=json.loads((root/'all-selections-frozen.json').read_text());assert frozen==s['retained']
        task=[]
        for method in OPTIMIZERS:
            d=audit_optimizer(root/method,contract()['development'],ImproverProtocol.from_dict(s['protocols'][method]))
            assert frozen[method]==json.loads((root/method/'selection-frozen.json').read_text())
            task.append(d);slots+=d['candidate_slots']
        calls=random_a['model_calls']+model_a['model_calls']+sum(d['model_calls'] for d in task)
        assert calls==s['actual_total_model_calls'];total_calls+=calls
        assert not s['smoke'] and not s['test_used_for_selection']
        for method,result in s['test'].items():
            check_mean(result['metrics'],result['cases']);Workflow.from_dict(result['workflow'])
            assert len(result['cases'])==24
            for row in result['cases']:
                case={k:row[k] for k in ['split','layout_seed','field_seed','profile','noise']}
                assert case in contract()['test']
                ages=np.asarray(row['target_ages']);due=deadlines(case)
                assert row['target_deadlines']==due.tolist() and ages.shape==(97,8)
                assert np.isclose(row['mean_deadline_violation'],(ages[1:]>due).mean())
                assert row['peak_age']==ages.max()
                cache_key=json.dumps(case,sort_keys=True)+json.dumps(row['positions'])
                if cache_key not in replayed:
                    env=PatrolEnv(room(case['layout_seed'],case['noise']),case['field_seed'])
                    env._neighborhoods=neighborhoods.setdefault(case['layout_seed'],{})
                    initial=env.raw_age[env.scenario.targets].copy();assert np.array_equal(initial,ages[0])
                    idleness=[];points=np.asarray(row['positions'])
                    assert np.array_equal(points[0],env.positions)
                    for t in range(96):
                        moves=points[t+1]-points[t];actions=[]
                        for delta in moves:
                            actions.append(-1 if not delta.any() else next(i for i,m in enumerate(MOVES) if np.array_equal(m,delta)))
                        env.step(np.asarray(actions,dtype=int))
                        assert np.array_equal(env.raw_age[env.scenario.targets],ages[t+1])
                        idleness.append(env.evaluation()['normalized_weighted_idleness'])
                    replayed[cache_key]=(env.evaluation()['target_coverage'],float(np.mean(idleness)),env.evaluation()['total_movement_distance'],hashlib.sha256(ages.tobytes()).hexdigest())
                coverage,weighted,movement,age_hash=replayed[cache_key]
                assert age_hash==hashlib.sha256(ages.tobytes()).hexdigest()
                assert np.isclose(row['target_coverage'],coverage) and np.isclose(row['weighted_idleness'],weighted)
                assert np.isclose(row['movement_distance'],movement)
                assert 0<=row['intervention_fraction']<=1 and 0<=row['ddqn_command_fraction']<=1
                test_rows+=1
        audits.append({'trial':trial,'meta_random':random_a,'meta_model':model_a,'task_workflow_selections_verified':True,
                       'no_meta_task_workflow_or_memory_transferred':True,'test_metrics_and_geometric_trajectories_replayed':True,
                       'model_calls':calls})
    a.output.write_text(json.dumps({'real_model_calls':total_calls,'task_candidate_slots':slots,'test_method_case_records':test_rows,
                                   'unique_geometric_trajectories_replayed':len(replayed),'audits':audits},indent=2))
    print('AUDIT VERIFIED',total_calls,'real model calls;',slots,'task slots;',test_rows,'test records;',len(replayed),'distinct traces')


if __name__=='__main__':main()
