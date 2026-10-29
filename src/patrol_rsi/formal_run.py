"""Bounded phase executor. Immutable policies; no adaptive test access."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from . import causal_run as runner
from .formal_study import conditions,load_protocol,method_definitions
from .m11 import frozen_predictor,seed_digest
from .m12 import load_actors,actor_scorer
from .public_stopping import StopConfig


def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')


def execute(spec,predictor,score,method,config):
    """Observe the unchanged decision call; no extra candidate/rollout."""
    original=runner.decision;selected_snapshots=[];sources=Counter();cache_types=Counter()
    def observe(bundle,used=False,config=StopConfig()):
        value=original(bundle,used,config)
        selected=value.get('selected')
        for kind,item in (('CHECK',selected),('WAIT_best',value.get('future_selected'))):
            if item:
                sources[kind+':'+item['target_source']]+=1
                row=bundle['tasks'][item['task']]
                if item['target_source']=='history_assimilated':
                    raw=row['cvae_raw'];history=row['history_assimilated']
                    cache_types[kind+':'+('equals_pure_prior' if raw['available'] and raw['time_s']==history['time_s']
                        and abs(raw['value']-history['value'])<1e-10 else 'measurement_assimilated_or_unavailable_prior')]+=1
        if selected:selected_snapshots.append(copy.deepcopy(bundle))
        return value
    runner.decision=observe
    try:record,env=runner.run(spec,predictor,score,method,config)
    finally:runner.decision=original
    record['selected_public_snapshots']=selected_snapshots
    record['source_attribution']={'target_source_occurrences':dict(sources),'history_cache_types':dict(cache_types),
        'scope':'Observed actual score provenance, no estimator intervention or causal module attribution.'}
    return record,env


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=('development','validation','test'),required=True)
    p.add_argument('--chunk',type=int,required=True)
    for name in ('output','protocol','actors','estimator-weights','estimator-manifest','gate'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--selection',type=Path)
    args=p.parse_args();torch.set_num_threads(1)
    protocol=load_protocol(args.protocol);gate=read(args.gate)
    if not gate['success'] or gate['skipped'] or gate['tests']<215:raise RuntimeError('Real Torch start gate not passed')
    selection=read(args.selection) if args.selection else None
    methods=method_definitions(protocol,args.phase,selection)
    specs=conditions(protocol,args.phase,args.chunk)
    args.output.mkdir(exist_ok=False);(args.output/'cases').mkdir();(args.output/'processes').mkdir()
    write(args.output/'phase-frozen.json',{'phase':args.phase,'chunk':args.chunk,'conditions':specs,'methods':methods,
        'started_unix':time.time(),'protocol_sha256':sha(args.protocol),
        'gate_sha256':sha(args.gate),'selection_sha256':sha(args.selection) if args.selection else None,
        'predecessor_selection':selection,'job_id':os.environ.get('SLURM_JOB_ID')})
    predictor=frozen_predictor(args.estimator_weights,args.estimator_manifest)
    actors,weights=load_actors(args.actors);score=actor_scorer(actors[0])
    write(args.output/'model-inputs.json',{'actors':weights,'CVAE_sha256':sha(args.estimator_weights),
        'torch':torch.__version__,'new_training':False,'LLM_calls':0})
    index=[];processes={};started=time.perf_counter()
    for alias,definition in methods.items():
        config=StopConfig(**definition['config']) if definition['config'] else StopConfig()
        for spec in specs:
            begin=time.perf_counter();record,env=execute(spec,predictor,score,definition['method'],config)
            wall=time.perf_counter()-begin
            case='--'.join(str(spec[k]) for k in ('phase','seed','scenario','duration_s','fault_start_s','noise_c'))+'--'+alias
            record['formal_identity']={'alias':alias,'execution_method':definition['method'],'config':definition['config']}
            dest=args.output/'cases'/(case+'.json.gz');raw=(json.dumps(record,allow_nan=False)+'\n').encode()
            dest.write_bytes(gzip.compress(raw,mtime=0))
            key='--'.join(str(spec[k]) for k in ('seed','scenario','duration_s','fault_start_s'))
            if key not in processes:
                path=args.output/'processes'/(key+'.npz');np.savez_compressed(path,**env.trace.arrays)
                processes[key]={'path':str(path.relative_to(args.output)),'bytes':path.stat().st_size,
                    'sha256':sha(path),'numeric_digest':env.trace.fingerprint}
            elif processes[key]['numeric_digest']!=env.trace.fingerprint:raise RuntimeError('Paired process drift')
            selected=[d['selected'] for d in record['stopping_decisions'] if d.get('selected')]
            index.append({'case':case,'alias':alias,'definition':definition,'specification':spec,'metrics':record['metrics'],
                'wall_s':wall,'path':str(dest.relative_to(args.output)),'sha256':sha(dest),'bytes':dest.stat().st_size,
                'process_key':key,'event_weight_sum':sum(w['weight'] for w in record['anomaly_intervals']),
                'weighted_delay_sum':sum(w['weight']*w['delay_or_censor_s'] for w in record['anomaly_intervals']),
                'anomaly_intervals':record['anomaly_intervals'],'selected_scores':selected,
                'stopping_CHECK_frames':sum(d.get('action')=='CHECK' for d in record['stopping_decisions']),
                'waiting_positive_current_frames':sum(d.get('action')=='WAIT' and d.get('q_check',0)>0 for d in record['stopping_decisions']),
                'source_attribution':record['source_attribution']})
            if len(index)%8==0:print('FORMAL_PROGRESS',args.phase,args.chunk,len(index),alias,flush=True)
    for seed,model in actors.items():
        if seed_digest(model)!=weights[str(seed)]['parameters_sha256']:raise RuntimeError('Old PPO changed')
    write(args.output/'case-index.json',index);write(args.output/'process-index.json',processes)
    write(args.output/'summary.json',{'phase':args.phase,'chunk':args.chunk,'actual_cases':len(index),
        'expected_cases':len(methods)*len(specs),'physical_configurations':len(processes),
        'seconds':time.perf_counter()-started,'model_weights_unchanged':True,'new_training':False,'LLM_calls':0})
    print('FORMAL_DONE',args.phase,args.chunk,len(index),flush=True)


if __name__=='__main__':main()
