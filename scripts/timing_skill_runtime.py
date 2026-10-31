"""Source-bound actual evaluation and open-model calls for M12-F campaigns."""
from __future__ import annotations
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

import torch

from patrol_rsi.limited_prediction import need,read,write,sha
from patrol_rsi.m11 import frozen_predictor
from patrol_rsi.m12 import load_actors,actor_scorer,verify_open_model
from patrol_rsi.m4 import ModelCaller
from patrol_rsi.timing_skill import TimingSkill
from patrol_rsi.timing_learning import aggregate,retrieve_cards,reject_hidden_prompt
from run_m12d_paired import StoredTrace,save_trace,write_gz
from run_m12e_timing import IndependentPhaseTrace
from timing_skill_execution import run_policy,policy_id

ACTORS=Path('/share/home/renly/renly-patrol-rsi/runs/20261027-m11/policies-v2')
CVAE=Path('/share/home/renly/renly-patrol-rsi/runs/20261028-m111/frozen-estimator.pt')
MODEL=Path('/share/home/renly/renly-patrol-rsi/models/Qwen3-8B-b968826d')
PREDICTOR=SCORER=None


def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def budget_reservation(root,kind,amount,cap,purpose):
    """Append-only reservation receipts plus a locked counter, including failures."""
    import fcntl
    area=Path(root)/'cost';area.mkdir(exist_ok=True)
    with (area/f'{kind}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        counter=area/f'{kind}-counter.json'
        old=read(counter) if counter.exists() else {'reserved':20 if kind=='episodes' else 0,'F0_charged':20 if kind=='episodes' else 0}
        need(old['reserved']+amount<=cap,'Frozen total '+kind+' cap')
        receipt={'first':old['reserved'],'amount':amount,'purpose':purpose,'job_id':os.environ['SLURM_JOB_ID'],'unix':time.time()}
        with (area/f'{kind}-reservations.jsonl').open('a') as f:f.write(json.dumps(receipt,allow_nan=False)+'\n')
        old['reserved']+=amount;temporary=area/f'{kind}-counter-{os.getpid()}.tmp'
        write(temporary,old);temporary.replace(counter)
        fcntl.flock(lock,fcntl.LOCK_UN)
        return receipt


def fail_stop(root,error):
    try:write(Path(root)/'F-STOP.json',{'error_type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),'unix':time.time(),'job_id':os.environ.get('SLURM_JOB_ID')})
    except FileExistsError:pass


def check_runtime(root,deadline=None):
    need(not(Path(root)/'F-STOP.json').exists(),'Prior F source/runtime/safety stop; no retry')
    need(deadline is None or time.monotonic()<deadline,'Frozen allocation soft runtime cap')


def check_storage(root):
    total=0
    for path in Path(root).rglob('*'):
        try:
            if path.is_file():total+=path.stat().st_size
        except FileNotFoundError:pass  # Another declared atomic counter replacement.
    need(total<=6000000000,'Frozen6GB new-storage cap')
    return total


def init_worker(source):
    global PREDICTOR,SCORER
    torch.set_num_threads(1)
    need(not torch.cuda.is_initialized(),'CPU workers must not inherit CUDA')
    s=Path(source);PREDICTOR=frozen_predictor(CVAE,s/'research/2026-10-28-m111-coverage/frozen_estimator.json')
    actors,_=load_actors(ACTORS);SCORER=actor_scorer(actors[0])


def execute_work(item):
    case,policy,folder,path,root,capture,deadline=item;path=Path(path)
    try:
        check_runtime(root,deadline)
        write(path.with_suffix('.attempt.json'),{'specification':case['specification'],'policy_spec':policy,'charged_actual_episode_attempt':True,'unix':time.time()})
        trace=StoredTrace(Path(folder));need(trace.fingerprint==case['process_fingerprint'],'Exact physical source in worker')
        tape=path.with_suffix('.probe.jsonl.gz') if capture else None
        with gzip.open(tape,'xt',encoding='utf-8') if tape is not None else __import__('contextlib').nullcontext(None) as stream:
            def observe(env,candidates):
                if stream is not None:
                    for row in candidates:stream.write(json.dumps(row['features'],allow_nan=False)+'\n')
            record,env,audit=run_policy(case['specification'],policy,trace,PREDICTOR,SCORER,capture_tape=observe if capture else None)
        write_gz(path,record)
        eligibility=[{k:v[k] for k in ('task','onset_s','end_s','weight','samples')} for v in record['anomaly_intervals']]
        value={'specification':case['specification'],'process_fingerprint':trace.fingerprint,'event_eligibility_SHA256':fingerprint(eligibility),
            'metrics':record['metrics'],'usage':record['usage'],'audit':audit,'action_SHA256':record['action_log_SHA256'],
            'record_path':str(path),'record_SHA256':sha(path),'probe_path':str(tape) if tape is not None else None,
            'probe_SHA256':sha(tape) if tape is not None else None}
        write(path.with_suffix('.completed.json'),value)
        return value
    except BaseException as error:
        fail_stop(root,error);raise


class EvaluationStore:
    def __init__(self,root,source,pool,area,cases,cap,context,*,deadline=None):
        self.root,self.source,self.pool=Path(root),Path(source),pool
        self.area=Path(area);self.area.mkdir(parents=True,exist_ok=False)
        self.cases=copy.deepcopy(cases);self.cap=cap;self.context=copy.deepcopy(context);self.deadline=deadline
        self.cache={};self.results={};self.requests=[];self.attempted=0;self.card_cache={}
        need(len({fingerprint(v['specification']) for v in cases})==len(cases),'Unique frozen evaluation cohort')

    def evaluate(self,policy,*,capture=False):
        policy=copy.deepcopy(policy)
        check_runtime(self.root,self.deadline)
        check_storage(self.root)
        key=fingerprint({'cases':self.cases,'policy':policy,'context':self.context})
        self.requests.append({'key':key,'cache_hit':key in self.cache,'policy':policy})
        if key in self.cache:
            if capture:need(all(v['probe_path'] is not None for v in self.cache[key]['rows']),'Initial capture must precede cache-only reference requests')
            return copy.deepcopy(self.cache[key])
        need(self.attempted+len(self.cases)<=self.cap,'Actual partition environment cap')
        reservation=budget_reservation(self.root,'episodes',len(self.cases),6000,str(self.area)+'/'+key)
        self.attempted+=len(self.cases)
        dest=self.area/f'eval-{len(self.cache):03d}-{policy_id(policy)[:12]}';dest.mkdir()
        write(dest/'started.json',{'key':key,'policy':policy,'reservation':reservation,'context':self.context,'cases':self.cases})
        items=[(case,policy,case['trace_folder'],str(dest/f'case-{i:02d}.json.gz'),str(self.root),capture,self.deadline) for i,case in enumerate(self.cases)]
        rows=list(self.pool.map(execute_work,items,chunksize=1))
        result={'key':key,'policy':policy,'statistics':aggregate(rows,[v['specification'] for v in self.cases]),'rows':rows}
        write(dest/'complete.json',result);self.cache[key]=result;self.results[key]=result
        print('M12F_EVALUATED',str(self.area),len(self.cache),self.attempted,policy_id(policy),flush=True)
        return copy.deepcopy(result)

    def cards(self,result,diagnostics):
        key=(result['key'],tuple(diagnostics))
        if key not in self.card_cache:
            candidates=[]
            for row in result['rows']:
                path=Path(row['record_path']);need(sha(path)==row['record_SHA256'],'Original actual feedback record unchanged')
                record=json.loads(gzip.decompress(path.read_bytes()))
                candidates.extend(retrieve_cards(record['causal_experience_cards'],diagnostics))
            self.card_cache[key]=retrieve_cards(candidates,diagnostics)
        return copy.deepcopy(self.card_cache[key])

    def behavior(self,candidate,initial):
        skill=TimingSkill.from_dict(candidate['policy']['skill']);reference=TimingSkill.initial_e1();different=0;tested=0
        for row in initial['rows']:
            path=Path(row['probe_path']);need(sha(path)==row['probe_SHA256'],'Frozen same-input public probe')
            with gzip.open(path,'rt',encoding='utf-8') as f:
                for line in f:
                    public=json.loads(line);different+=skill.action(public)!=reference.action(public);tested+=1
        a={fingerprint(v['specification']):v['action_SHA256'] for v in initial['rows']}
        changed=sum(v['action_SHA256']!=a[fingerprint(v['specification'])] for v in candidate['rows'])
        return {'same_input_rows':tested,'decision_differences':different,'actual_trajectory_differences':changed,'behavior_changed':different>0 and changed>0}

    def finish(self):
        value={'attempted_episode_reservations':self.attempted,'cap':self.cap,'logical_requests':len(self.requests),
            'cache_hits':sum(v['cache_hit'] for v in self.requests),'requests':self.requests,'results':list(self.results.values())}
        write(self.area/'store-complete.json',value)
        return value


def future_source_folder(root,case,history):
    """One staging owner at a time; numerical identity is global across partitions."""
    import fcntl
    spec=case['specification'];directory=root/'future-sources'/case['partition'];directory.mkdir(parents=True,exist_ok=True)
    name='--'.join(str(spec[k]) for k in ('seed','scenario','duration_s','fault_start_s'));folder=directory/name
    with (root/'future-source.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not folder.exists():
            write(directory/(name+'--generation-attempt.json'),{'specification':spec,'planned_manifest_case':case['case_id'],'actual_generation_attempt':True})
            trace=IndependentPhaseTrace(spec)
            need(trace.fingerprint not in history['numeric_fingerprints'],'No historical physical process reuse')
            save_trace(trace,folder)
        info=read(folder/'complete.json');fp=info['fingerprint']
        need(fp not in history['numeric_fingerprints'],'Reused file must also exclude historical processes')
        index=root/'future-numeric-process-index.json';values=read(index) if index.exists() else {}
        identity={'partition':case['partition'],'physical_key':name}
        need(fp not in values or values[fp]==identity,'No cross-partition or cross-key numerical process reuse')
        if fp not in values:
            values[fp]=identity;temporary=root/f'future-index-{os.getpid()}.tmp';write(temporary,values);temporary.replace(index)
        fcntl.flock(lock,fcntl.LOCK_UN)
    return folder


def source_cases(root,cases,*,known=False,history=None):
    """Generate only this explicitly opened cohort, once per physical process."""
    root=Path(root);result=[]
    for original in cases:
        case=copy.deepcopy(original);spec=case['specification'];check_runtime(root)
        if known:
            need(spec['seed'] in (33000,33001) and spec['phase']=='development','Known E1 development only')
            folder=Path(case['trace_folder']);need(sha(folder/'complete.json')==case['source_meta_SHA256'],'Known source metadata authority')
        else:
            need(spec['seed'] not in history['seed_groups'],'New cohort seed not historical')
            folder=future_source_folder(root,case,history)
        info=read(folder/'complete.json')
        need(info['seed']==spec['seed'] and info['scenario']==spec['scenario'] and info['metadata']['configuration']['fault_start_s']==spec['fault_start_s'] and info['metadata']['configuration']['fault_end_s']==spec['fault_start_s']+spec['duration_s'],'Actual declared physical source')
        need(sha(folder/'trace.npz')==info['trace_file_SHA256'],'Immutable physical file identity')
        if known:need(info['fingerprint']==case['process_fingerprint'],'Known physical identity')
        case.update(trace_folder=str(folder),process_fingerprint=info['fingerprint'],actual_source_meta_SHA256=sha(folder/'complete.json'))
        result.append(case)
    keys={}
    for case in result:
        spec=case['specification'];key=tuple(spec[k] for k in ('seed','scenario','duration_s','fault_start_s'));fp=case['process_fingerprint']
        need(fp not in keys or keys[fp]==key,'Distinct declared processes must have distinct numerical identities');keys[fp]=key
    return result


class ActualCaller(ModelCaller):
    def __init__(self,root,area,call_cap,*,deadline):
        self.root,self.area,self.cap,self.deadline=Path(root),Path(area),call_cap,deadline
        self.area.mkdir(parents=True,exist_ok=False);self.receipts=[]
        need(torch.cuda.is_available() and torch.cuda.device_count()==1,'One allocated CUDA device for actual model calls')
        device=torch.cuda.get_device_properties(0)
        need('V100S' in device.name and device.total_memory>=32000000000,'Frozen V100S32GB resource class')
        write(self.area/'model-startup.json',{'GPU':device.name,'total_GPU_bytes':device.total_memory,'torch':torch.__version__,
            'job_id':os.environ['SLURM_JOB_ID'],'model_directory':str(MODEL),'unix':time.time()})
        need(sha(MODEL/'verified_manifest.json')=='a1a327b7aee47f77c9348796bcb2d459d8792a8bc6644339b713668528af4361','Exact frozen Qwen manifest')
        self.model_manifest=verify_open_model(MODEL)
        super().__init__(MODEL,'cuda')
        self.versions=tuple(v._version for v in self.model.parameters())
        write(self.area/'model-loaded.json',{'frozen_manifest_SHA256':sha(MODEL/'verified_manifest.json'),
            'parameter_tensors':len(self.versions),'model_verified':self.model_manifest,'unix':time.time()})

    def call(self,user,seed,*,meta=False):
        check_runtime(self.root,self.deadline);need(len(self.receipts)<self.cap,'Fixed per-allocation model call cap')
        user=copy.deepcopy(user);reject_hidden_prompt(user);dropped=0
        system='Return exactly one JSON object with the requested response_keys. Follow the bounded typed schema. No text outside JSON. Use only supplied public evidence.'
        cards=user['completed_causal_experience' if meta else 'data']['experience_cards']
        while True:
            rendered=self.tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':json.dumps(user,ensure_ascii=False)}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs=self.tokenizer(rendered,return_tensors='pt');count=int(inputs['input_ids'].numel())
            if count<=6144:break
            need(bool(cards),'Fixed schema/parent/history exceeds input cap; no generation/retry')
            cards.pop();dropped+=1
        receipt=budget_reservation(self.root,'model_calls',1,75,str(self.area)+'/'+str(len(self.receipts)))
        path=self.area/f'call-{len(self.receipts):02d}'
        start={'seed':seed,'prompt':user,'rendered_prompt':rendered,'input_tokens':count,'dropped_last_cards':dropped,
            'reservation':receipt,'output_cap':1024,'unknown_output_charge_upper_bound':1024,'thinking':False}
        write(path.with_suffix('.start.json'),start);self.receipts.append(start)
        def timeout(signum,frame):raise TimeoutError('Frozen90s model call timeout; no retry')
        previous=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,90)
        begin=time.perf_counter();generated=None;answer=None
        try:
            torch.manual_seed(seed)
            with torch.inference_mode():
                generated=self.model.generate(**inputs.to('cuda'),max_new_tokens=1024,do_sample=True,temperature=.7,top_p=.8,top_k=20)
            answer={'raw_response':self.tokenizer.decode(generated[0,count:],skip_special_tokens=True),
                'input_tokens':count,'output_tokens':int(generated.shape[1]-count),'generation_s':time.perf_counter()-begin,
                'dropped_last_cards':dropped,'prompt_SHA256':fingerprint(user)}
            need(tuple(v._version for v in self.model.parameters())==self.versions,'Frozen Qwen parameters')
            write(path.with_suffix('.result.json'),answer);self.receipts[-1].update(answer)
            return answer
        except BaseException as error:
            failed={'error_type':type(error).__name__,'message':str(error),'input_tokens':count,
                'generation_s':time.perf_counter()-begin,'output_tokens':int(generated.shape[1]-count) if generated is not None else None,
                'output_tokens_charge_upper_bound':int(generated.shape[1]-count) if generated is not None else 1024,
                'known_response':answer,'no_retry':True}
            write(path.with_suffix('.failed.json'),failed);self.receipts[-1].update(failed)
            fail_stop(self.root,error);raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
