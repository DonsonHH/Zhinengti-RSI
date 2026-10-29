"""Frozen-weight Torch regression and ten-case CPU pilot, no training/Qwen."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import platform
import time
import unittest

import numpy as np
import torch

from patrol_rsi.causal_observation import ObservedTask,capture_public,task_signal
from patrol_rsi.causal_run import run
from patrol_rsi.improver_task import PatrolProgram
from patrol_rsi.m11 import frozen_predictor
from patrol_rsi.m12 import actor_scorer,load_actors
from patrol_rsi.public_stopping import StopConfig,public_frontier,decision


def read(path):
    raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)


def write(path,value):
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')


def replay_saved(record,predictor,times):
    env=ObservedTask(record['specification'],PatrolProgram(),predictor)
    captured=[]
    for row in record['action_log']:
        if row['time_s']!=env.clock:raise RuntimeError('Saved action clock drift')
        if env.clock in times:captured.append(capture_public(env))
        env.step({int(k):v for k,v in row['choices'].items()})
    if env.events!=record['inspection_events']:
        raise RuntimeError('Frozen-model causal replay changed qualified measurements')
    if env.selected_certificates!=record['selected_certificates']:
        raise RuntimeError('Causal recorder changed return certificates')
    return captured


def main():
    p=argparse.ArgumentParser()
    for name in ('output','actors','estimator-weights','estimator-manifest','protocol'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1);root=Path(__file__).resolve().parents[1]
    protocol=read(args.protocol)
    write(args.output/'protocol-receipt.json',{'sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        'started_unix':time.time(),'protocol':protocol})
    stream=io.StringIO();started=time.perf_counter()
    tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover(str(root/'tests')))
    (args.output/'full-tests.txt').write_text(stream.getvalue(),encoding='utf-8')
    write(args.output/'full-tests.json',{'tests':tests.testsRun,'skipped':len(tests.skipped),
        'failures':len(tests.failures),'errors':len(tests.errors),'seconds':time.perf_counter()-started,
        'torch':torch.__version__,'python':platform.python_version(),'success':tests.wasSuccessful()})
    if tests.skipped or not tests.wasSuccessful():raise RuntimeError('Actual Torch suite must pass with no skips')
    predictor=frozen_predictor(args.estimator_weights,args.estimator_manifest)
    actors,fingerprints=load_actors(args.actors);score=actor_scorer(actors[0])
    old=root/'research/2026-10-29-m12c1b2-confirm/evidence'
    index={r['case']:r for r in read(old/'report/compact-index.json.gz')}
    files=[next((old/'demo').glob('*--8300--335--0.2--history--lookahead.json.gz'))]
    files.extend(sorted((old/'failure-examples').glob('*history--lookahead.json.gz')))
    regressions=[]
    for path in files:
        case=path.name.removesuffix('.json.gz');row=read(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=index[case]['sha256']:raise RuntimeError('Old case source drift')
        when=next(r['time_s'] for r in row['workflow_decisions'] if r['task']!=r['baseline_next'])
        bundle=replay_saved(row,predictor,{when})[0]
        target=156 if row['specification']['seed']==8300 else 108
        signal,source=task_signal(bundle['tasks'][target],45. if target==156 else 25.)
        frontier=public_frontier(bundle)
        selected=next((r for r in frontier if r['task']==target and r['dispatch_time_s']==when),None)
        if signal is None or signal<=0 or selected is None or selected['target_benefit']<=0:
            raise RuntimeError('Old positive public cue was lost by causal adapter')
        value=decision(bundle)
        regressions.append({'original_case':case,'snapshot_time_s':when,'public_state':bundle,
            'record_sha256':index[case]['sha256'],'target':target,'target_signal':signal,'signal_source':source,
            'current_target_value':selected,'stopping_decision':value,
            'measurement_and_certificate_replay_exact':True})
        print('REGRESSION',case,signal,flush=True)
    write(args.output/'old-public-regressions.json',{'development_only':True,'cases':regressions})
    config=StopConfig(**next(v['config'] for v in protocol['revised_candidates'] if v['id']=='C2'))
    (args.output/'cases').mkdir();records=[]
    for scenario,duration in protocol['pilot']['fixtures']:
        spec=dict(phase='engineering_pilot',seed=protocol['engineering_pilot_seed'],scenario=scenario,
            duration_s=duration,fault_start_s=335,noise_c=.2,estimator='history',control='feedback',resources='provisioned')
        for method in protocol['main_methods']:
            started=time.perf_counter();record,env=run(spec,predictor,score,method,config,capture_times=(370.,))
            wall=time.perf_counter()-started
            case=scenario+'--'+method;raw=(json.dumps(record)+'\n').encode()
            dest=args.output/'cases'/(case+'.json.gz');dest.write_bytes(gzip.compress(raw,mtime=0))
            records.append({'case':case,'method':method,'specification':spec,'metrics':record['metrics'],
                'wall_s':wall,'trace_fingerprint':record['trace_fingerprint'],'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
                'path':str(dest.relative_to(args.output)),'actions':len(record['action_log']),
                'decisions':len(record['stopping_decisions']),'completed_inspections':len(record['inspection_events'])})
            print('PILOT',case,round(wall,3),'s',flush=True)
    if len(records)!=10 or any(r['actions']!=359 or r['decisions']!=359 for r in records):
        raise RuntimeError('Pilot was incomplete')
    for scenario,duration in protocol['pilot']['fixtures']:
        if len({r['trace_fingerprint'] for r in records if r['specification']['scenario']==scenario})!=1:
            raise RuntimeError('Methods changed paired physical process')
    write(args.output/'pilot-index.json',records)
    write(args.output/'model-inputs.json',{'actors':fingerprints,'estimator_sha256':hashlib.sha256(args.estimator_weights.read_bytes()).hexdigest(),
        'torch':torch.__version__,'numpy':np.__version__,'new_training':False,'LLM_calls':0})
    write(args.output/'pilot-gate.json',{'complete_cases':len(records),'max_wall_s':max(r['wall_s'] for r in records),
        'cost_within_frozen_cap':all(r['wall_s']<=protocol['resource_caps']['pilot_max_elapsed_s_per_case'] for r in records),
        'all_hard_constraints_passed':all(r['metrics']['tasks_violating_exact_gap']==0 and r['metrics']['selected_detours']<=1 for r in records),
        'formal_test_executed':False,'scope':'Engineering pilot only; not used to choose coefficients/candidates or claim benefit.'})
    print('PREFLIGHT_FINISHED',len(records),flush=True)


if __name__=='__main__':main()
