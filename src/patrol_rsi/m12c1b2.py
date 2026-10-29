"""Frozen phase/noise confirmation and information controls for C1-B1."""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .m12 import actor_scorer, load_actors, sha, write
from .m11 import frozen_predictor, seed_digest
from .m12b import specification, TEST
from .m12c1b import aggregate
from .robust_timing import blind_time
from .robust_timing_run import run

PHASES = (150, 335, 520, 705)
SEEDS = (8300, 8301)


def configs():
    return {'tour': {'method':'tour'}, 'cue_immediate': {'method':'cue'},
        'evidence_now': {'method':'evidence'}, 'lookahead': {'method':'lookahead'},
        'lookahead_permuted': {'method':'lookahead','signal_mode':'permuted'},
        'lookahead_masked': {'method':'lookahead','signal_mode':'masked'},
        'lookahead_conservative': {'method':'lookahead','conservative':True},
        **{f'blind_{i}': {'method':'blind','blind_seed':i} for i in (0,1,2)}}


def contract():
    return {'stage':'C1-B2 frozen timing confirmation, information and blind controls',
        'physical_seeds':list(SEEDS), 'fault_start_grid_s':list(PHASES), 'noise_c':[.2,1.],
        'physical_configurations':64, 'observation_conditions':128, 'public_control_groups':configs(),
        'formal_rollouts':1280, 'chunks':8, 'cases_per_chunk':160,
        'blind_trigger_s':{str(i):blind_time(i) for i in (0,1,2)},
        'one_global_optional_check':True, 'new_neural_training':False,'LLM_calls':0,'improver_updated':False,
        'selection':'No hyperparameter selection; all ten frozen controls reported.',
        'independence':'Two physical seeds shared across scenarios/phases; noise/control repetitions dependent.',
        'scope':'Controlled same-layout confirmation, not field or new learning/RSI evidence'}


def specs(chunk=0, smoke=False):
    if smoke:
        return [dict(specification('smoke',a,b,8399),fault_start_s=335) for a,b in (('healthy',900),('coldplate_fouling',180))]
    if type(chunk) is not int or chunk not in range(8):
        raise ValueError('Eight predeclared phase/seed chunks')
    phase,seed=PHASES[chunk//2],SEEDS[chunk%2]
    return [dict(specification('test',a,b,seed,noise),fault_start_s=phase) for a,b in TEST for noise in (.2,1.)]


def name(spec,tag):
    return '--'.join(str(spec[k]) for k in ('phase','scenario','duration_s','seed','fault_start_s','noise_c','estimator'))+'--'+tag


def main():
    p=argparse.ArgumentParser()
    for key in ('output','actors','estimator-weights','estimator-manifest'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--chunk',type=int,choices=range(8),default=0);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();torch.set_num_threads(1);started=time.perf_counter()
    predictor=frozen_predictor(args.estimator_weights,args.estimator_manifest)
    actors,fingerprints=load_actors(args.actors);actor_score=actor_scorer(actors[0])
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    (output/'cases').mkdir();(output/'process-traces').mkdir()
    write(output/'contract.json',contract())
    write(output/'immutable-inputs.json',{'actors':fingerprints,'estimator_sha256':sha(args.estimator_weights),
        'job_id':os.environ.get('SLURM_JOB_ID'),'torch':torch.__version__,'numpy':np.__version__})
    write(output/'controls-frozen-before-test.json',{'configurations':configs(),'conditions':specs(args.chunk,args.smoke),
        'frozen_at_unix':time.time(),'chunk':args.chunk,'smoke':args.smoke})
    rows,processes,groups=[],{},{}
    for tag,config in configs().items():
        group=[]
        for spec in specs(args.chunk,args.smoke):
            record,env=run(spec,predictor,actor_score,config['method'],config=config)
            key=name(spec,tag);path=output/'cases'/(key+'.json.gz')
            path.write_bytes(gzip.compress((json.dumps(record)+'\n').encode(),mtime=0))
            process='--'.join(str(spec[k]) for k in ('scenario','duration_s','seed','fault_start_s'))
            if process not in processes:
                dest=output/'process-traces'/(process+'.npz');np.savez_compressed(dest,**env.trace.arrays)
                processes[process]={'path':str(dest.relative_to(output)),'sha256':sha(dest),
                    'bytes':dest.stat().st_size,'numeric_digest':env.trace.fingerprint}
            elif processes[process]['numeric_digest']!=env.trace.fingerprint:
                raise RuntimeError('Timing or information control changed physical process')
            extra=[r for r in record['workflow_decisions'] if r['task']!=r['baseline_next']]
            positive_wait=[r for r in record['timing_decisions'] if r['decision']=='wait_value' and (r['now_value'] or 0)>0]
            row={'case':key,'tag':tag,'specification':spec,'deployment':record['deployment'],
                'path':str(path.relative_to(output)),'sha256':sha(path),'bytes':path.stat().st_size,
                'process_key':process,'metrics':record['metrics'],'anomaly_intervals':record['anomaly_intervals'],
                'event_weight_sum':sum(r['weight'] for r in record['anomaly_intervals']),
                'weighted_delay_sum':sum(r['weight']*r['delay_or_censor_s'] for r in record['anomaly_intervals']),
                'optional_action':extra[0] if extra else None,
                'positive_now_wait_frames':len(positive_wait),
                'total_wait_value_frames':sum(r['decision']=='wait_value' for r in record['timing_decisions']),
                'last_positive_wait_s':positive_wait[-1]['time_s'] if positive_wait else None}
            rows.append(row);group.append(row)
            if len(rows)%16==0:
                print('B2_CASES',len(rows),tag,flush=True)
        groups[tag]={'case_ids':[r['case'] for r in group],'aggregate':aggregate(group)}
    for seed,model in actors.items():
        if seed_digest(model)!=fingerprints[str(seed)]['parameters_sha256']:
            raise RuntimeError('Frozen old actor changed')
    write(output/'groups.json',groups);write(output/'case-index.json',rows);write(output/'process-index.json',processes)
    write(output/'summary.json',{'smoke':args.smoke,'chunk':args.chunk,'actual_cases':len(rows),'physical_configurations':len(processes),
        'new_neural_training':False,'LLM_calls':0,'seconds':time.perf_counter()-started,
        'old_actors_unchanged':True,'job_id':os.environ.get('SLURM_JOB_ID')})
    print('B2_COMPLETE',args.chunk,len(rows),flush=True)


if __name__=='__main__':
    main()
