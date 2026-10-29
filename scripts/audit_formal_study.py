"""One bounded fresh replay per formal case, routes, certificates and sums."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from patrol_rsi.formal_run import execute
from patrol_rsi.formal_study import conditions,load_protocol,method_definitions
from patrol_rsi.formal_study import verify_case_index,verified_selection
from patrol_rsi.m11 import frozen_predictor
from patrol_rsi.m12 import actor_scorer,load_actors
from patrol_rsi.public_stopping import StopConfig
from audit_causal_preflight import audit_routes
from audit_m111 import independent_certificates


def read(path):
    raw=path.read_bytes();return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)


def same(a,b):
    if isinstance(a,dict):return set(a)==set(b) and all(same(v,b[k]) for k,v in a.items())
    if isinstance(a,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return isinstance(b,(int,float)) and np.isclose(a,b,rtol=1e-10,atol=1e-11)
    return a==b


def main():
    p=argparse.ArgumentParser()
    for name in ('folder','protocol','output','actors','estimator-weights','estimator-manifest'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1);folder=args.folder
    frozen=read(folder/'phase-frozen.json');protocol=load_protocol(args.protocol)
    if frozen['phase']!='development':
        name='development' if frozen['phase']=='validation' else 'validation'
        selected=verified_selection(folder.parent/(name+'-selection.json'),protocol,folder.parent,name)
        if frozen['predecessor_selection']!=selected:raise RuntimeError('Final stage selection changed')
    methods=method_definitions(protocol,frozen['phase'],frozen['predecessor_selection'])
    specs=conditions(protocol,frozen['phase'],frozen['chunk'])
    rows=read(folder/'case-index.json');processes=read(folder/'process-index.json')
    if frozen['conditions']!=specs or frozen['methods']!=methods or len(rows)!=len(specs)*len(methods):
        raise RuntimeError('Frozen phase scope mismatch')
    expected=[(k,s) for k in methods for s in specs]
    if [(r['alias'],r['specification']) for r in rows]!=expected:raise RuntimeError('Missing, reordered or extra case')
    predictor=frozen_predictor(args.estimator_weights,args.estimator_manifest)
    actors,_=load_actors(args.actors);score=actor_scorer(actors[0])
    visits=certificates=decisions=0;seen=set()
    for i,row in enumerate(rows):
        path=folder/row['path'];raw=path.read_bytes()
        if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:raise RuntimeError('Original case bytes')
        old=json.loads(gzip.decompress(raw));d=methods[row['alias']]
        fresh,env=execute(row['specification'],predictor,score,d['method'],StopConfig(**d['config']) if d['config'] else StopConfig())
        verify_case_index(row,old,fresh)
        for key in ('action_log','inspection_events','selected_certificates','assignments','anomaly_intervals',
            'stopping_decisions','causal_snapshot_stamps','source_attribution','selected_public_snapshots','public_queue_comparison','initial_certificates'):
            if not same(old[key],fresh[key]):raise RuntimeError('Fresh policy/weight replay differs: '+key)
        if len(old['action_log'])!=359 or len(old['stopping_decisions'])!=359:raise RuntimeError('Partial episode')
        visits+=audit_routes(old,env)
        cert_record=dict(old,metrics=dict(old['metrics'],
            maximum_exact_gap_ratio=max(r['maximum_gap_including_initial_and_terminal_s']/env.deadlines[r['task']]
                for r in old['exact_task_gaps'])))
        certificates+=independent_certificates(cert_record,env)[1]
        # Independent recomputation from stored actual interval/confirmation
        # records, themselves checked against the complete fresh execution.
        missed=sum(not w['detected'] for w in old['anomaly_intervals'])
        weight=sum(w['weight'] for w in old['anomaly_intervals'])
        delay=sum(w['weight']*w['delay_or_censor_s'] for w in old['anomaly_intervals'])
        if missed!=row['metrics']['missed_segments'] or not same(weight,row['event_weight_sum']) or not same(delay,row['weighted_delay_sum']):
            raise RuntimeError('Independent case metric sums differ')
        for item in old['stopping_decisions']:
            if item.get('action') in ('CHECK','WAIT'):
                if item['CHECK']['end_s']!=1800 or item['WAIT']['end_s']!=1800 or item['CHECK']['budget_after']!=0 or item['WAIT']['budget_after']!=1:
                    raise RuntimeError('CHECK WAIT consistency lost')
                for v in (item.get('selected'),item.get('future_selected')):
                    if v:
                        cfg=item['config']
                        expected_score=v['target_benefit']-cfg['collateral']*(v['first_round_collateral_cost']+v['future_risk_correction'])-v['execution_cost']
                        if not same(expected_score,v['net_score']):raise RuntimeError('Value component arithmetic differs')
                decisions+=1
        if row['process_key'] not in seen:
            source=processes[row['process_key']];pfile=folder/source['path']
            if hashlib.sha256(pfile.read_bytes()).hexdigest()!=source['sha256']:raise RuntimeError('Physical archive bytes')
            with np.load(pfile) as data:
                if set(data.files)!=set(env.trace.arrays) or any(not np.array_equal(data[k],env.trace.arrays[k]) for k in data.files):
                    raise RuntimeError('Saved physical arrays differ from fresh process')
            seen.add(row['process_key'])
        if (i+1)%8==0:print('FORMAL_AUDIT',frozen['phase'],frozen['chunk'],i+1,flush=True)
    result={'phase':frozen['phase'],'chunk':frozen['chunk'],'actual_cases_replayed':len(rows),
        'independent_route_measurements':visits,'independent_return_certificates':certificates,
        'consistent_stopping_frames':decisions,'physical_arrays_verified':len(seen),'exact_violations':0,
        'case_index_sha256':hashlib.sha256((folder/'case-index.json').read_bytes()).hexdigest(),
        'global_optional_budget':1,'all_statistics_and_original_cases_checked':True}
    with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print('FORMAL_AUDIT_DONE',len(rows),flush=True)


if __name__=='__main__':main()
