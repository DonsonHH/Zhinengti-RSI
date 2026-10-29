"""Fresh full action replay, independent route/qualification/gap audit."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import torch

from patrol_rsi.aisles import build_agents
from patrol_rsi.causal_observation import capture_public,validate_public
from patrol_rsi.causal_run import run
from patrol_rsi.m11 import frozen_predictor
from patrol_rsi.m12 import actor_scorer,load_actors
from patrol_rsi.public_stopping import StopConfig,stopping_decision
from diagnose_queue_cost import distance
from audit_m111 import independent_certificates


def read(path):
    data=path.read_bytes();return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)


def audit_routes(record,env):
    agents=build_agents(env.graph,'provisioned');position={a['id']:a['position'] for a in agents}
    free={i:5. for i in position};events={(e['agent'],e['task'],e['time_s']):e for e in record['inspection_events']}
    seen=set();bytask={j:[] for j in range(len(env.tasks))}
    for row in record['assignments']:
        i,j,t=row['agent'],row['task'],row['time_s'];task=env.tasks[j]
        if task.required_role!=agents[i]['role'] or j not in agents[i]['tour'] or t<free[i]-1e-9:
            raise RuntimeError('Qualification, ownership or busy conflict')
        length=distance(env.graph,position[i],task.position_m)
        if abs(length-row['distance_m'])>1e-8:raise RuntimeError('Independent Dijkstra differs')
        route=[tuple(v) for v in row['route']]
        if route[0]!=position[i] or route[-1]!=task.position_m:raise RuntimeError('Path endpoints')
        if abs(sum(env.graph.adjacency[a][b] for a,b in zip(route,route[1:]))-length)>1e-8:raise RuntimeError('Nonminimal/invalid path')
        finish=t+math.ceil((length+task.service_s)/5-1e-10)*5
        event=events[(i,j,finish)]
        if not event['calibrated'] or event['access']!=task.access or finish>1800:raise RuntimeError('Invalid measurement')
        if env.trace.measurement(task,int(finish/5)-1)!=event['measurement']:raise RuntimeError('Recorded measurement differs from actual process')
        seen.add((i,j,finish));bytask[j].append(finish);position[i]=task.position_m;free[i]=finish
    if len(seen)!=len(events):raise RuntimeError('Unmatched inspection')
    for j,times in bytask.items():
        times=[0.]+sorted(times)+[1800.]
        if max(b-a for a,b in zip(times,times[1:]))>env.deadlines[j]+1e-9:raise RuntimeError('Exact gap violated')
    used=[r for r in record['selected_certificates'] if r['task']!=r['baseline_next']]
    if len(used)>1:raise RuntimeError('Global budget exceeded')
    return len(seen)


def main():
    p=argparse.ArgumentParser()
    for name in ('pilot','output','actors','estimator-weights','estimator-manifest'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    predictor=frozen_predictor(args.estimator_weights,args.estimator_manifest)
    actors,_=load_actors(args.actors);score=actor_scorer(actors[0])
    count=events=frames=certificates=0
    for row in read(args.pilot/'pilot-index.json'):
        path=args.pilot/row['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Original pilot fingerprint')
        old=read(path);new,env=run(row['specification'],predictor,score,row['method'],StopConfig(),capture_times=(370.,))
        for key in ('action_log','inspection_events','selected_certificates','assignments','anomaly_intervals','public_snapshots'):
            if old[key]!=new[key]:raise RuntimeError('Fresh replay differs in '+key)
        events+=audit_routes(old,env)
        cert_record=dict(old,initial_certificates=env.initial_certificates,metrics=dict(old['metrics'],
            maximum_exact_gap_ratio=max(r['maximum_gap_including_initial_and_terminal_s']/env.deadlines[r['task']]
                for r in old['exact_task_gaps'])))
        certificates+=independent_certificates(cert_record,env)[1]
        for snap in old['public_snapshots']:validate_public(snap)
        if row['method']=='revised':
            for d in old['stopping_decisions']:
                if d['action']=='TOUR':continue
                # Independent arithmetic and finite suffix maximum law.
                for selected in (d.get('selected'),d.get('future_selected')):
                    if selected:
                        expected=selected['target_benefit']-selected['first_round_collateral_cost']-selected['future_risk_correction']-selected['execution_cost']
                        if abs(expected-selected['net_score'])>1e-10:raise RuntimeError('Value components do not sum')
                if (d['action']=='CHECK')!=(d['selected'] is not None):raise RuntimeError('Action/value disagreement')
                if d['CHECK']['end_s']!=d['WAIT']['end_s'] or d['CHECK']['budget_after']!=0 or d['WAIT']['budget_after']!=1:
                    raise RuntimeError('CHECK/WAIT continuation mismatch')
                frames+=1
        count+=1;print('CAUSAL_AUDIT',count,flush=True)
    result={'fresh_weight_replayed_cases':count,'independent_route_measurements':events,
        'independent_complete_return_certificates':certificates,
        'value_component_and_branch_contract_frames':frames,'exact_revisit_and_global_budget_passed':True,
        'formal_test_accessed':False,'scope':'Engineering pilot replay and independent path/gap/measurement checks, not benefit inference'}
    with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')


if __name__=='__main__':main()
