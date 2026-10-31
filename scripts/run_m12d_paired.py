"""Formal phase-scoped five-arm execution with no final-before-seal path."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from patrol_rsi.active_inspection import ProcessTrace
from patrol_rsi.aligned_patrol import arm_programs,run_episode
from patrol_rsi.facility import FacilityConfig
from patrol_rsi.improver_task import ImproverTask,PhasedTrace
from patrol_rsi.limited_prediction import need,read,sha,write
from patrol_rsi.m11 import frozen_predictor
from patrol_rsi.m12 import load_actors,actor_scorer
from patrol_rsi.m9 import arrays_digest
from patrol_rsi.paired_campaign import protocols,manifest,require_final_seal,PAIRED_SHA,exclude_history,verify_identity
from audit_m11 import independent_schedule,independent_events
from audit_m111 import independent_certificates


def write_gz(path,value):
    with path.open('xb') as f:f.write(gzip.compress(json.dumps(value,allow_nan=False).encode(),mtime=0))


class StoredTrace(ProcessTrace):
    def __init__(self,folder):
        info=read(folder/'complete.json')
        need(sha(folder/'trace.npz')==info['trace_file_SHA256'],'Immutable generated process bytes')
        self.metadata=info['metadata'];self.config=FacilityConfig(**self.metadata['configuration'])
        self.seed=info['seed'];self.scenario=info['scenario'];self.control='feedback';self.fingerprint=info['fingerprint']
        with np.load(folder/'trace.npz') as f:
            self.arrays={key:f[key] for key in info['array_names']}
            packet_arrays={key:f['packet__'+key] for key in info['packet_names']}
        need(arrays_digest(self.arrays)==self.fingerprint,'Physical fingerprint identity')
        self.packets=[{key:value[i].copy() if isinstance(value[i],np.ndarray) else value[i].item() for key,value in packet_arrays.items()} for i in range(360)]


class InputAuditedTask(ImproverTask):
    def packet(self):
        packet=super().packet()
        if not hasattr(self,'public_packet_hashes'):self.public_packet_hashes=[]
        self.public_packet_hashes.append(arrays_digest(packet))
        return packet


def save_trace(trace,folder):
    folder.mkdir(exist_ok=False)
    keys=tuple(trace.packets[0]);packet={key:np.asarray([v[key] for v in trace.packets]) for key in keys}
    np.savez_compressed(folder/'trace.npz',**trace.arrays,**{'packet__'+k:v for k,v in packet.items()})
    write(folder/'complete.json',{'metadata':trace.metadata,'seed':trace.seed,'scenario':trace.scenario,'fingerprint':trace.fingerprint,
        'trace_file_SHA256':sha(folder/'trace.npz'),'array_names':list(trace.arrays),'packet_names':list(keys)})


def execute(spec,program,trace,predictor,scorer):
    holder=[]
    def factory(s,p,model):
        with patch('patrol_rsi.improver_task.phased_trace',return_value=trace):env=InputAuditedTask(s,p,model)
        holder.append(env);return env
    before=arrays_digest(trace.arrays)
    record=run_episode(spec,program,predictor,scorer,environment_factory=factory);env=holder[-1]
    record.update(initial_certificates=env.initial_certificates,scalar_trace=env.scalar_trace,
        workflow_contract={'task_weights':env.asset_weights.tolist(),'deadlines_s':env.deadlines.tolist()},
        team={'count':len(env.agents),'per_agent':[{k:a[k] for k in ('id','role','zone','travel_m','service_s','idle_s','completed')} for a in env.agents]})
    record['metrics']['maximum_exact_gap_ratio']=max(v['maximum_gap_including_initial_and_terminal_s']/env.deadlines[v['task']] for v in record['exact_task_gaps'])
    record['metrics']['maximum_exact_gap_s']=max(v['maximum_gap_including_initial_and_terminal_s'] for v in record['exact_task_gaps'])
    need(len(env.public_packet_hashes)==360,'All360 noisy public telemetry packets recorded')
    record['public_telemetry_sequence_SHA256']=__import__('hashlib').sha256(''.join(env.public_packet_hashes).encode()).hexdigest()
    need(arrays_digest(trace.arrays)==before,'Actions cannot alter frozen physical process')
    return record,env


def audit_record(record,env):
    independent_schedule(record,env);independent_events(record,env);plans,total=independent_certificates(record,env)
    need(record['team']['count']==18 and record['metrics']['selected_detours']<=1,'Common staff/global budget')
    by_task={j:[] for j in range(len(env.tasks))}
    for e in record['inspection_events']:by_task[e['task']].append(e['time_s'])
    count=0
    for row in record['workflow_decisions']:
        if row['task']!=row['baseline_next']:
            need(not row['one_extra_already_used'],'No second optional action')
            need(row['detours_used_before']<2,'Original owned-round quota')
            previous=[t for t in by_task[row['task']] if t<=row['time_s']]
            need(not previous or row['time_s']-max(previous)>=record['workflow']['cooldown_s'],'Original extra-check cooldown')
            count+=1
    need(count==record['metrics']['selected_detours'],'Independent one-CHECK count')
    return {'passed':True,'independent_path_qualification_measurement_checks':len(record['inspection_events']),
        'independent_full_return_certificates':plans,'selected_plan_count':total,'initial_terminal_revisit_verified':True,
        'physical_process_unchanged':True}


def usage_details(record):
    rows=record['workflow_decisions'];ppo=[v for v in rows if v['tool']=='ppo']
    return dict(record['usage'],PPO_single_legal_candidate_dispatches=sum(v['legal_candidates']==1 for v in ppo),
        actual_PPO_multichoice_takeover_rate=sum(v['legal_candidates']>1 for v in ppo)/len(rows) if rows else 0.,
        rule_selected_dispatches=sum(v['rule_index'] is not None and v['tool']!='common_budget_return_tour' for v in rows),
        forced_return_Tour_dispatches=sum(v['tool']=='common_budget_return_tour' for v in rows),
        ordinary_return_Tour_dispatches=sum(v['tool']=='return_tour' for v in rows))


def main():
    parser=argparse.ArgumentParser()
    for name in ('paired','rsi','manifest','review','gate','run-root','actors','estimator-weights','estimator-manifest'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--phase',choices=('smoke','development','validation','final_test'),required=True)
    parser.add_argument('--seed-index',type=int,default=0);args=parser.parse_args();os.umask(0o077);torch.set_num_threads(1)
    p,r=protocols(args.paired,args.rsi);need(read(args.manifest)==manifest(p),'Frozen complete planned process manifest')
    approval=read(args.review);gate=read(args.gate)
    need(approval['approved'] and approval['paired_SHA256']==PAIRED_SHA,'Independent source/phase execution review')
    identity=read(args.run_root/'execution-identity.json')
    identity_SHA=sha(args.run_root/'execution-identity.json')
    verify_identity(identity,{'paired':args.paired,'rsi':args.rsi,'manifest':args.manifest,'review':args.review,'gate':args.gate,
        'history':args.run_root/'historical-identities.json','actor0':args.actors/'history-seed-0/selected.pt',
        'actor1':args.actors/'history-seed-1/selected.pt','actor2':args.actors/'history-seed-2/selected.pt',
        'CVAE':args.estimator_weights,'CVAE_manifest':args.estimator_manifest})
    need(gate['success'] and gate['tests']>=286 and not(gate['errors'] or gate['failures'] or gate['skipped']),'Actual regression before full episode')
    need(sha(args.estimator_weights)==p['frozen_models']['temperature_CVAE_SHA256'],'Frozen temperature CVAE')
    need(sha(args.actors/'history-seed-0/selected.pt')==p['frozen_models']['actor']['sha256'],'Frozen actor')
    if args.phase=='final_test':require_final_seal(args.run_root/'validation-seal.json',args.manifest)
    if args.phase=='validation':
        previous=read(args.run_root/'development-complete.json')
        need(previous['complete'] and previous['execution_identity_SHA256']==identity_SHA,'Same-source development fully audited before validation')
    if args.phase=='smoke':
        selected=[{'id':'engineering-30990','physical_key':'30990--coldplate_fouling--180--335','seed':30990,
            'specification':{'phase':'smoke','seed':30990,'scenario':'coldplate_fouling','duration_s':180,'fault_start_s':335,'noise_c':.2,
                'estimator':'history','control':'feedback','resources':'provisioned','variant':'open'}}];seed=30990
    else:
        seed=p['partition'][args.phase][args.seed_index]
        selected=[v for v in read(args.manifest)['rows'] if v['phase']==args.phase and v['seed']==seed]
        need(len(selected)==18,'Exactly18conditions/9processes eachseed')
        smoke=read(args.run_root/'smoke/seed-30990/complete.json')
        need(smoke['complete'] and smoke['execution_identity_SHA256']==identity_SHA,'Same-source full executor smoke before formal phases')
    dest=args.run_root/args.phase/f'seed-{seed}';dest.mkdir(parents=True,exist_ok=False);(dest/'sources').mkdir();(dest/'cases').mkdir()
    write(dest/'started.json',{'phase':args.phase,'seed':seed,'protocol_SHA256':PAIRED_SHA,'manifest_SHA256':sha(args.manifest),
        'execution_identity_SHA256':identity_SHA,
        'review_SHA256':sha(args.review),'gate_SHA256':sha(args.gate),'job_id':os.environ.get('SLURM_JOB_ID'),'started_unix':time.time()})
    predictor=frozen_predictor(args.estimator_weights,args.estimator_manifest);actors,info=load_actors(args.actors);scorer=actor_scorer(actors[0])
    rows=[];processes={};started=time.perf_counter()
    historical=read(args.run_root/'historical-identities.json')
    for item in selected:
        spec=item['specification'];key=item['physical_key'];folder=dest/'sources'/key
        if key not in processes:
            trace=PhasedTrace(spec['seed'],spec['scenario'],spec['duration_s'],spec['fault_start_s']);save_trace(trace,folder)
            exclude_history(spec['seed'],trace.fingerprint,historical)
            processes[key]=trace.fingerprint
            del trace
        trace=StoredTrace(folder)
        for arm,program in arm_programs().items():
            case=item['id']+'--'+arm;attempt=dest/'cases'/(case+'--attempt.json')
            write(attempt,{'case':case,'charged_formal_episode':args.phase!='smoke','started_unix':time.time(),'protocol_SHA256':PAIRED_SHA})
            record,env=execute(spec,program,trace,predictor,scorer);audit=audit_record(record,env)
            if args.phase=='smoke' and arm=='Tour':
                rejected=[]
                for label in ('qualification','extra_quota','return_certificate','initial_terminal_gap'):
                    broken=copy.deepcopy(record)
                    if label=='qualification':broken['inspection_events'][0]['role']='wrong_profession'
                    elif label=='extra_quota':broken['metrics']['selected_detours']=2
                    elif label=='return_certificate':broken['selected_certificates'][0]['certificate']['fallback_completion_s'][0]+=5
                    else:broken['exact_task_gaps'][0]['maximum_gap_including_initial_and_terminal_s']+=600
                    try:audit_record(broken,env)
                    except (ValueError,RuntimeError,AssertionError):rejected.append(label)
                need(len(rejected)==4,'All negative legality/quota/return/head-tail audits must reject')
                write(dest/'negative-executor-tests.json',{'all_four_rejected':True,'cases':rejected,'source':'actual full smoke record;no formal data'})
            replay,_=execute(spec,program,trace,predictor,scorer)
            for name in ('action_log','inspection_events','selected_certificates','anomaly_intervals','usage','public_telemetry_sequence_SHA256'):
                need(record[name]==replay[name],'Every actual episode frozen-weight replay: '+name)
            audit['independent_full_episode_replay_passed']=True
            path=dest/'cases'/(case+'.json.gz');write_gz(path,record)
            rows.append({'case':case,'arm':arm,'seed':seed,'physical_key':key,'noise_c':spec['noise_c'],'phase':args.phase,
                'specification':spec,'process_fingerprint':trace.fingerprint,'source_trace_file_SHA256':sha(folder/'trace.npz'),
                'record_path':str(path),'record_SHA256':sha(path),'metrics':record['metrics'],'by_role':record['by_role'],
                'usage':usage_details(record),'action_log_SHA256':__import__('hashlib').sha256(json.dumps(record['action_log'],sort_keys=True).encode()).hexdigest(),
                'public_telemetry_sequence_SHA256':record['public_telemetry_sequence_SHA256'],
                'inspection_events_SHA256':__import__('hashlib').sha256(json.dumps(record['inspection_events'],sort_keys=True).encode()).hexdigest(),
                'audit':audit})
            write(dest/'cases'/(case+'--complete.json'),rows[-1]);print('M12D_EPISODE_AUDITED',args.phase,seed,case,flush=True)
        need(arrays_digest(trace.arrays)==processes[key],'All five arms same unchanged process')
    need(sha(args.run_root/'execution-identity.json')==identity_SHA,'Execution identity cannot refresh mid-run')
    write(dest/'complete.json',{'complete':True,'phase':args.phase,'seed':seed,'conditions':len(selected),'formal_episodes':len(rows) if args.phase!='smoke' else 0,
        'engineering_episodes':len(rows) if args.phase=='smoke' else 0,'audit_replays':len(rows),'processes':processes,'rows':rows,
        'models':info,'CVAE_SHA256':sha(args.estimator_weights),'protocol_SHA256':PAIRED_SHA,'wall_s':time.perf_counter()-started,
        'execution_identity_SHA256':identity_SHA,
        'final_access':args.phase=='final_test','job_id':os.environ.get('SLURM_JOB_ID')})


if __name__=='__main__':main()
