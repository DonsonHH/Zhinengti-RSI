"""Bounded real four-group RSI adaptation/validation; final remains separate."""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from patrol_rsi.aligned_patrol import arm_programs
from patrol_rsi.facility_workflow import public_outcome
from patrol_rsi.improver_protocol import Improver,combine_diagnostics,validation_summary,TOOLS,MEMORY
from patrol_rsi.improver_task import PatrolProgram,PhasedTrace
from patrol_rsi.limited_prediction import read,write,sha,need
from patrol_rsi.m11 import frozen_predictor
from patrol_rsi.m12 import load_actors,actor_scorer,ModelCaller,verify_open_model
from patrol_rsi.m12b import random_program,parse
from patrol_rsi.m9 import arrays_digest
from patrol_rsi.paired_campaign import protocols,exclude_history,verify_identity
from patrol_rsi.rsi_campaign import cases,prefix_program,macro,retain_on_validation,GROUPS,FatalCampaignBoundary
from patrol_rsi.rsi_controlled import (ControlledImprover,INSTRUCTION_TEMPLATES,
    BudgetedEvaluationStore,run_slots,paired_protocol_update,transfer_reset)
from run_m12d_paired import StoredTrace,save_trace,execute,audit_record,write_gz,InputAuditedTask,usage_details

MODEL_PATH=Path('/share/home/renly/renly-patrol-rsi/models/Qwen3-8B-b968826d')
ACTORS=Path('/share/home/renly/renly-patrol-rsi/runs/20261027-m11/policies-v2')
CVAE=Path('/share/home/renly/renly-patrol-rsi/runs/20261028-m111/frozen-estimator.pt')
PREDICTOR=SCORER=None
DEADLINE=None


def within_runtime():
    if DEADLINE is not None and time.monotonic()>=DEADLINE:raise FatalCampaignBoundary('Frozen allocation runtime stop;no retry or extra proposal')


def plain(v):
    if hasattr(v,'to_dict'):return v.to_dict()
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [plain(x) for x in v]
    return v


def init_worker(source):
    global PREDICTOR,SCORER
    torch.set_num_threads(1);s=Path(source)
    PREDICTOR=frozen_predictor(CVAE,s/'research/2026-10-28-m111-coverage/frozen_estimator.json')
    actors,_=load_actors(ACTORS);SCORER=actor_scorer(actors[0])


def work(item):
    spec,pg,trace_path,path=item;path=Path(path);trace=StoredTrace(Path(trace_path));program=PatrolProgram.from_dict(pg)
    previous=InputAuditedTask._update
    def tracked(env,completed):
        previous(env,completed)
        if round(env.clock)%30==0 and env.clock<1800:
            if not hasattr(env,'rsi_tool_observations'):env.rsi_tool_observations=[]
            difference=np.abs(env.current_history-env.state.observer.predicted_temperature)
            env.rsi_tool_observations.append({'time_s':env.clock,'mean_abs_difference_degC':float(difference.mean()),'peak_abs_difference_degC':float(difference.max())})
    write(path.with_suffix('.attempt.json'),{'specification':spec,'program':pg,'attempt_charged':True,'start_unix':time.time()})
    with patch.object(InputAuditedTask,'_update',tracked):record,env=execute(spec,program,trace,PREDICTOR,SCORER)
    audit=audit_record(record,env)
    record['thermal_tool_observations']=env.rsi_tool_observations
    record['public_selection']=public_outcome(record,env.asset_weights)
    write_gz(path,record)
    weight=sum(v['weight'] for v in record['anomaly_intervals'])
    return {'specification':spec,'workflow':pg,'version_id':program.version_id,'process_fingerprint':trace.fingerprint,
        'metrics':record['metrics'],'usage':usage_details(record),'by_role':record['by_role'],'public_selection':record['public_selection'],
        'public_telemetry_sequence_SHA256':record['public_telemetry_sequence_SHA256'],
        'event_weight_sum':weight,'weighted_delay_sum':sum(v['weight']*v['delay_or_censor_s'] for v in record['anomaly_intervals']),
        'record_path':str(path),'record_SHA256':sha(path),'audit':audit}


class Store:
    def __init__(self,root,source,pool,history,*,resume=False):
        self.root=root;self.source=source;self.pool=pool;self.history=history;self.index={};self.processes={};self.sequence=0
        (root/'cases').mkdir(exist_ok=resume);(root/'sources').mkdir(exist_ok=resume);(root/'evaluations').mkdir(exist_ok=resume)
        if resume:
            previous=read(root/'adapt-validation-complete.json')
            self.index={v['record_path']:v for v in previous['case_index']};self.processes=previous['processes']
            self.sequence=10000
        self.budgets={k:BudgetedEvaluationStore(self.callback,cap) for k,cap in
            (('meta_round0',180),('meta_round1',180),('fresh_task_adapt',1152),('fresh_task_validation',216),('fresh_task_final',432))}
    def cache(self,phase):return self.budgets[phase]
    def ensure_trace(self,spec):
        key=f"{spec['seed']}--{spec['scenario']}--{spec['duration_s']}--{spec['fault_start_s']}"
        dest=self.root/'sources'/key
        if key not in self.processes:
            trace=PhasedTrace(spec['seed'],spec['scenario'],spec['duration_s'],spec['fault_start_s'])
            exclude_history(spec['seed'],trace.fingerprint,self.history)
            need(trace.fingerprint not in self.processes.values(),'No RSI cross-phase/seed physical reuse')
            save_trace(trace,dest);self.processes[key]=trace.fingerprint
        return dest
    def callback(self,case_specs,program):
        within_runtime()
        phase=case_specs[0]['phase'];need(all(v['phase']==phase for v in case_specs),'Single allowed task phase per store')
        if phase=='fresh_task_final':
            seal=read(self.root.parent/'rsi-final-seal.json')
            for artifact in seal['artifacts']:need(sha(Path(artifact['path']))==artifact['sha256'],'All3 mappings/protocol/source/model sealed before final')
        identity=f'{self.sequence:04d}';self.sequence+=1
        write(self.root/'evaluations'/(identity+'--start.json'),{'phase':phase,'program':program.to_dict(),'episode_attempts_charged':len(case_specs)})
        jobs=[]
        for spec in case_specs:
            signature=json.dumps({'specification':spec,'workflow':program.to_dict()},sort_keys=True)
            case=hashlib.sha256(signature.encode()).hexdigest()[:24];path=self.root/'cases'/(case+'.json.gz')
            jobs.append((spec,program.to_dict(),str(self.ensure_trace(spec)),str(path)))
        rows=self.pool.map(work,jobs,chunksize=1)
        for row in rows:self.index[row['record_path']]=row
        # Only diagnostics enter the next prompt; labels remain on evaluator side.
        records=[json.loads(gzip.decompress(Path(v['record_path']).read_bytes())) for v in rows]
        result={'validator':validation_summary(rows),'public':combine_diagnostics(records),'case_ids':[v['record_path'] for v in rows]}
        write(self.root/'evaluations'/(identity+'--complete.json'),{'phase':phase,'program':program.to_dict(),'result':result})
        return result
    def rows(self,result):return [self.index[p] for p in result['case_ids']]
    def ledger(self):
        return {k:{'requests':v.requests,'cache_hits':v.cache_hits,'attempted_episodes':v.attempted_episode_evaluations,'successful_episodes':v.actual_episode_evaluations,'cap':v.episode_cap} for k,v in self.budgets.items()}


class Caller(ModelCaller):
    def __init__(self,model,root):
        super().__init__(model,'cuda');self.root=root;self.receipts=[];self.versions=tuple(v._version for v in self.model.parameters());(root/'calls').mkdir()
    def call(self,user,seed):
        within_runtime()
        if len(self.receipts)>=26:raise FatalCampaignBoundary('Fixed26modelcall cap;campaign stop')
        number=len(self.receipts);base=self.root/'calls'/f'call-{number:02d}'
        system='Return exactly one JSON object using the response_keys requested. Follow the exact response schema. No source code or explanations outside JSON.'
        rendered=self.tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':json.dumps(user)}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=self.tokenizer(rendered,return_tensors='pt').to('cuda');n=int(inputs['input_ids'].numel())
        if n>6144:raise FatalCampaignBoundary('6144 inputtoken cap;campaign stop')
        attempt={'number':number,'seed':seed,'input_tokens':n,'prompt':user,'rendered_prompt':rendered,'thinking':False,'output_cap':768,'attempted':True}
        write(base.with_suffix('.start.json'),attempt);self.receipts.append(attempt);torch.manual_seed(seed);start=time.perf_counter()
        with torch.inference_mode():ids=self.model.generate(**inputs,max_new_tokens=768,do_sample=True,temperature=.7,top_p=.8,top_k=20)
        value={'input_tokens':n,'output_tokens':int(ids.shape[1]-n),'generation_s':time.perf_counter()-start,
            'raw_response':self.tokenizer.decode(ids[0,n:],skip_special_tokens=True)}
        if tuple(v._version for v in self.model.parameters())!=self.versions:raise FatalCampaignBoundary('Frozen Qwen weight boundary;campaign stop')
        self.receipts[-1].update(value);write(base.with_suffix('.result.json'),value)
        return value
    def workflow(self,prompt,seed):
        user=dict(prompt,output_envelope={'diagnosis':'short string','workflow':'PatrolProgram with exactly workflow_keys above','expected_effect':'short string'})
        user['common_execution_contract']='All task groups share AT MOST ONE optional CHECK across the whole team per1800s episode, then compulsoryTour. Originaltwo-per-owned-round/cooldown/qualification/fullreturn remainadditionalhardmasks. No extra atmissiontail ifnextTouritemcannotfinish. ScoringPPOafterglobalbudgetis shadowonly.'
        generated=self.call(user,seed)
        try:
            pg,envelope=parse(generated['raw_response'],'workflow')
            return {'workflow':pg.to_dict(),**{k:generated[k] for k in ('input_tokens','output_tokens')}}
        except (ValueError,TypeError,KeyError):return {'workflow':None,**{k:generated[k] for k in ('input_tokens','output_tokens')}}
    def meta(self,incumbent,reports,seed):
        user={'task':'Propose one controlled improver protocol, not a task program. Only public meta feedback is supplied.',
            'common_execution_contract':'Alltaskprograms use common atmostONEglobaloptionalCHECK thenTour/1800s;qualification/cooldown/complete-return/tail fixed. No protocol canchangequota or atomictools.',
            'current_protocol':incumbent.to_dict(),'prior_feedback':reports,'protocol_schema':incumbent.to_dict(),
            'allowed_diagnostics':list(TOOLS),'allowed_memory':list(MEMORY),'instruction_templates':list(INSTRUCTION_TEMPLATES),
            'allowed_generators':['qwen_once','random_program'],'allowed_dedup':['exact_version','canonical_rule_order'],
            'allowed_acceptance':['strict_improvement','keep_parent'],'allowed_failure_fallback':['keep_parent','safe_tour'],
            'response_keys':['diagnosis','protocol','expected_effect'],
            'output_envelope':{'diagnosis':'short string','protocol':'ControlledImprover object with exact schema','expected_effect':'short string'}}
        generated=self.call(user,seed)
        try:
            envelope=json.loads(generated['raw_response'][generated['raw_response'].index('{'):generated['raw_response'].rindex('}')+1])
            need(set(envelope)=={'diagnosis','protocol','expected_effect'} and all(isinstance(envelope[k],str) and 1<=len(envelope[k])<=500 for k in ('diagnosis','expected_effect')),'Closed protocol response envelope')
            return ControlledImprover.from_dict(envelope['protocol']),True
        except (ValueError,TypeError,KeyError):return incumbent,False


def main():
    global DEADLINE
    p=argparse.ArgumentParser()
    for n in ('source-root','run-root','review'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--trial',type=int,required=True);a=p.parse_args();os.umask(0o077);torch.set_num_threads(2)
    need(a.trial in (0,1,2),'Exactly3generationrepeats')
    source=a.source_root;prior=source/'research/2026-10-30-m12c1b4b2-alignment';paired,rsi=protocols(prior/'MultiAgent-Patrol-Paired-v1.json',prior/'RSI-Controlled-Meta-v1.json')
    review=read(a.review);need(review['approved'] and review['RSI_SHA256']==sha(prior/'RSI-Controlled-Meta-v1.json'),'Independent bounded RSI launch approval')
    identity=read(a.run_root/'rsi-execution-identity.json');identity_SHA=sha(a.run_root/'rsi-execution-identity.json')
    verify_identity(identity,{'paired':prior/'MultiAgent-Patrol-Paired-v1.json','rsi':prior/'RSI-Controlled-Meta-v1.json',
        'review':a.review,'history':a.run_root/'historical-identities.json',
        'actor0':ACTORS/'history-seed-0/selected.pt','actor1':ACTORS/'history-seed-1/selected.pt',
        'actor2':ACTORS/'history-seed-2/selected.pt','CVAE':CVAE,
        'CVAE_manifest':source/'research/2026-10-28-m111-coverage/frozen_estimator.json'})
    indexed={v['path']:v['sha256'] for v in identity['artifacts']}
    for path in (a.run_root/'qwen-preflight/preflight.json',a.run_root/'rsi-cost-profile.json',MODEL_PATH/'verified_manifest.json'):
        need(indexed.get(str(path))==sha(path),'Actual successful preflight,modelrevision and costprofile bound beforeRSI')
    gate=read(Path(identity['bound_inputs']['gate']))
    need(gate['success'] and gate['tests']>=293 and not(gate['skipped'] or gate['failures'] or gate['errors']),'Reviewed293Torch gate before formal RSI')
    root=a.run_root/f'rsi-trial-{a.trial}';root.mkdir(exist_ok=False)
    DEADLINE=time.monotonic()+3420
    write(root/'started.json',{'trial':a.trial,'job_id':os.environ.get('SLURM_JOB_ID'),'review_SHA256':sha(a.review),'real_model_call_cap':26,'start_unix':time.time()})
    history=read(a.run_root/'historical-identities.json');models=verify_open_model(MODEL_PATH)
    need(models['manifest_sha256']==read(a.run_root/'qwen-preflight/preflight.json')['model']['manifest_sha256'],'Same one-preflight verified Qwen weights')
    with mp.get_context('spawn').Pool(2,initializer=init_worker,initargs=(str(source),)) as pool:
        store=Store(root,source,pool,history);caller=Caller(MODEL_PATH,root);generation=rsi['generation_repeats'][a.trial]
        generators={'qwen_once':caller.workflow,'random_program':lambda prompt,seed:{'workflow':random_program(seed).to_dict()}}
        adapt_specs=cases('fresh_task_adapt',paired,rsi);fixed=ControlledImprover()
        direct=run_slots(fixed,store.cache('fresh_task_adapt'),generators,adapt_specs,generation*10000+8000,13)
        need(direct['status']=='completed_bounded_slots','Fixed search boundary success')
        random=run_slots(ControlledImprover(generator='random_program'),store.cache('fresh_task_adapt'),generators,adapt_specs,generation*10000+9500,13)
        need(random['status']=='completed_bounded_slots','Random search boundary success')
        incumbent=ControlledImprover();rounds=[];feedback=[]
        for number in range(2):
            proposal,valid=caller.meta(incumbent,feedback,generation*10000+3000+number)
            incumbent,receipt=paired_protocol_update(incumbent,proposal,store.cache(f'meta_round{number}'),generators,
                cases(f'meta_round{number}',paired,rsi),generation*10000+number*10)
            need(receipt['status']!='stopped_boundary_failure','Any meta boundary failure stops campaign')
            receipt['protocol_proposal_valid']=valid;rounds.append(plain(receipt))
            feedback.append({'round':number,'protocol_update_accepted':receipt['accepted'],'proposal_valid':valid})
        reset=transfer_reset(incumbent);need(reset['program']==PatrolProgram() and not reset['history'] and not reset['candidate_archive'],'Only protocol transfer')
        meta=run_slots(reset['protocol'],store.cache('fresh_task_adapt'),generators,adapt_specs,generation*10000+8000,3)
        need(meta['status']=='completed_bounded_slots','Transfer search boundary success')
        programs={'Tour':PatrolProgram(),'human_fixed':arm_programs()['human_professional'],'random_search':random['program'],
            'fixed3':prefix_program(direct,3),'fixed13':direct['program'],'controlled_meta':meta['program']}
        adaptation={name:store.cache('fresh_task_adapt').evaluate(adapt_specs,pg) for name,pg in programs.items()}
        validation_specs=cases('fresh_task_validation',paired,rsi)
        validation={name:store.cache('fresh_task_validation').evaluate(validation_specs,pg) for name,pg in programs.items()}
        statistics={name:macro(store.rows(value)) for name,value in validation.items()}
        mappings={}
        for name,pg in programs.items():
            gate=retain_on_validation(statistics[name],statistics['Tour']) if name!='Tour' else {'retain':True}
            mappings[name]={'nominal_program':pg.to_dict(),'validation_gate':gate,'actual_program':(pg if gate['retain'] else PatrolProgram()).to_dict()}
        need(sha(a.run_root/'rsi-execution-identity.json')==identity_SHA,'No midrun RSI source/model binding refresh')
        write(root/'validation-mapping-seal.json',{'models':models,'programs':mappings,'validation':statistics,
            'execution_identity_SHA256':identity_SHA,
            'model_calls':len(caller.receipts),'RSI_protocol_SHA256':sha(prior/'RSI-Controlled-Meta-v1.json'),'final_sources_opened':False,'sealed_unix':time.time()})
        write(root/'adapt-validation-complete.json',{'complete':True,'trial':a.trial,'generation_seed':generation,
            'execution_identity_SHA256':identity_SHA,
            'direct':plain(direct),'random':plain(random),'meta_rounds':rounds,'transfer':plain(meta),'programs':mappings,
            'adaptation':adaptation,'validation':statistics,'model_call_receipts':caller.receipts,'costs':store.ledger(),
            'case_index':list(store.index.values()),'processes':store.processes,'final_source_access':False,
            'allcalls_input_tokens':sum(v['input_tokens'] for v in caller.receipts),'allcalls_output_tokens':sum(v.get('output_tokens',0) for v in caller.receipts),
            'scope':'Real bounded adaptation/validation. Final is a separate CPU stage after all3mappings/source seals.'})
        print('M12D_RSI_ADAPT_VALIDATION_COMPLETE',a.trial,len(caller.receipts),sum(v.attempted_episode_evaluations for v in store.budgets.values()),flush=True)


if __name__=='__main__':main()
