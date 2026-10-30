"""Small allowlisted wrapper around existing Improver/PatrolProgram interfaces."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

from .improver_protocol import Improver,program_prompt,validation_rank
from .improver_task import PatrolProgram

INSTRUCTION_TEMPLATES=(Improver().instruction,
    'Compare public successes with failures and propose one distinct typed workflow; preserve the protected tour.',
    'Analyze public measurement age, professional evidence and duplicate proposals before proposing one bounded workflow.')


def require(value,message):
    if not value:raise ValueError(message)


@dataclass(frozen=True)
class ControlledImprover:
    protocol: Improver=Improver()
    generator: str='qwen_once'
    dedup: str='exact_version'
    acceptance: str='strict_improvement'
    failure_fallback: str='keep_parent'

    def __post_init__(self):
        require(isinstance(self.protocol,Improver),'Reuse existing protocol')
        require(self.protocol.instruction in INSTRUCTION_TEMPLATES,'Controlled transfer accepts only public instruction templates')
        require(self.generator in ('qwen_once','random_program'),'Fixed candidate generators')
        require(self.dedup in ('exact_version','canonical_rule_order'),'Fixed bounded dedup rules')
        require(self.acceptance in ('strict_improvement','keep_parent'),'No relaxation of fixed validator')
        require(self.failure_fallback in ('keep_parent','safe_tour'),'Fixed failure fallback')

    def to_dict(self):
        return {'protocol':self.protocol.to_dict(),'generator':self.generator,'dedup':self.dedup,
            'acceptance':self.acceptance,'failure_fallback':self.failure_fallback}

    @classmethod
    def from_dict(cls,value):
        require(isinstance(value,dict) and set(value)=={'protocol','generator','dedup','acceptance','failure_fallback'},'Closed controlled protocol schema')
        return cls(Improver.from_dict(value['protocol']),value['generator'],value['dedup'],value['acceptance'],value['failure_fallback'])

    @property
    def version_id(self):return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True).encode()).hexdigest()[:16]


def program_key(program,mode):
    # Ordered rules are semantically significant. Canonical means serialization
    # and repeated identical conditions only; never reorder or merge thresholds.
    require(mode in ('exact_version','canonical_rule_order'),'Closed dedup mode')
    if mode=='exact_version':return program.version_id
    value=program.to_dict();rules=[];seen=set()
    for rule in value['rules']:
        key=rule['signal'],rule['threshold'],rule['action']
        if key not in seen:rules.append(rule);seen.add(key)
    value['rules']=rules
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:16]


PUBLIC_FIELDS={
    'coverage':('coverage','violating_tasks','weighted_age_ratio','first_alert_tasks'),
    'efficiency':('travel_m','service_s','extra_checks','estimator_calls','dispatches','different_from_actor'),
    'thermal_disagreement':('meaning','mean_abs_difference_degC','peak_abs_difference_degC')}
PROF_FIELDS=('checks','distinct_checked_tasks','distinct_measured_alarm_tasks','new_measured_alerts','persistent_rechecks',
    'recovery_confirmations','measurement_field','unit','mean_measured_value','largest_public_completion_gap_s')
FORBIDDEN={'onset_s','end_s','seed','scenario','fault_start_s','event_id','anomaly_intervals','test_results','final_test',
    'missed_segments','weighted_censored_delay_s','oracle','truth','future_value'}


def reject_hidden(value):
    if isinstance(value,dict):
        require(not(set(value)&FORBIDDEN),'Hidden evaluator field in public prompt')
        for v in value.values():reject_hidden(v)
    elif isinstance(value,(list,tuple)):
        for v in value:reject_hidden(v)


def public_prompt(program,views,history,protocol):
    reject_hidden(views);clean={}
    def project(row,fields):
        require(isinstance(row,dict),'Typed diagnostic object')
        result={}
        for key in fields:
            if key not in row:continue
            value=row[key]
            if key in ('meaning','measurement_field','unit'):
                require(isinstance(value,str) and len(value)<=500,'Bounded public text field')
            else:require(value is None or type(value) in (int,float) and math.isfinite(value),'Scalar finite public diagnostic')
            result[key]=value
        return result
    for tool in protocol.protocol.diagnostics:
        if tool=='novelty':continue
        row=views.get(tool,{})
        if tool=='professional_evidence':
            clean[tool]={role:project(values,PROF_FIELDS)
                for role,values in row.items() if role in ('thermal','electrical','cooling','control')}
        else:clean[tool]=project(row,PUBLIC_FIELDS[tool])
    memory=[{k:v[k] for k in ('candidate_id','workflow','accepted','reason') if k in v} for v in history]
    for row in memory:
        if row.get('reason')=='duplicate_consumes_slot':row['reason']='duplicate'
    reject_hidden(memory)
    return program_prompt(program,clean,memory,protocol.protocol)


def partitions_disjoint(partitions):
    """Caller must supply both numerical fingerprints and generation seed groups."""
    seen_seeds={};seen_processes={}
    for phase,rows in partitions.items():
        require(rows,'No empty task partition')
        for row in rows:
            require(set(row)=={'seed_group','process_fingerprint'},'Explicit physical identity,not phase-label-only split')
            for key,seen in (('seed_group',seen_seeds),('process_fingerprint',seen_processes)):
                identity=row[key];require(identity not in seen or seen[identity]==phase,'Cross-partition physical dependence')
                seen[identity]=phase
    return True


def transfer_reset(protocol):
    require(isinstance(protocol,ControlledImprover),'Only a protocol transfers')
    return {'protocol':ControlledImprover.from_dict(protocol.to_dict()),'program':PatrolProgram(),'history':[],
        'candidate_archive':{},'task_outcomes':[]}


class BudgetedEvaluationStore:
    """Shared immutable cache inside ONE task partition; caps actual episodes."""
    def __init__(self,evaluate,episode_cap):
        require(type(episode_cap) is int and episode_cap>0,'Positive fixed episode budget')
        self.evaluate_uncached=evaluate;self.episode_cap=episode_cap;self.cache={}
        self.requests=0;self.cache_hits=0;self.actual_episode_evaluations=0;self.attempted_episode_evaluations=0

    def evaluate(self,cases,program):
        self.requests+=1
        key=json.dumps({'cases':cases,'program':program.to_dict()},sort_keys=True)
        if key in self.cache:
            self.cache_hits+=1
            return json.loads(self.cache[key])
        require(self.attempted_episode_evaluations+len(cases)<=self.episode_cap,'Fixed environment evaluation cap reached')
        self.attempted_episode_evaluations+=len(cases)
        value=self.evaluate_uncached(cases,program)
        self.actual_episode_evaluations+=len(cases)
        self.cache[key]=json.dumps(value,sort_keys=True,allow_nan=False)
        return json.loads(self.cache[key])


def run_slots(protocol,store,generators,cases,seed,slots,*,initial=None):
    """Dependency-injected trial; store owns frozen validation and safety checks."""
    require(0<=slots<=13 and protocol.generator in generators,'Finite implemented generator budget')
    program=PatrolProgram();parent=store.evaluate(cases,program)
    if initial is not None:
        require(initial['version_id']==program.version_id and initial['case_ids']==parent['case_ids'],'Initial cache belongs to same default task')
    seen={program_key(program,protocol.dedup)};history=[];receipts=[];costs={'candidate_attempts':0,'model_calls':0,'input_tokens':0,'output_tokens':0,'environment_evaluation_requests':1}
    for slot in range(slots):
        costs['candidate_attempts']+=1;entry={'slot':slot,'accepted':False,'reason':None,'phase':'prompt'}
        entry['parent_id']=program.version_id;candidate=None
        try:
            prompt=public_prompt(program,parent['public'],history,protocol);entry['phase']='generation'
            if protocol.generator=='qwen_once':costs['model_calls']+=1
            value=generators[protocol.generator](prompt,seed+slot)
            for key in ('input_tokens','output_tokens'):costs[key]+=int(value.get(key,0))
            entry['phase']='parse';candidate=PatrolProgram.from_dict(value['workflow']);entry['candidate_id']=candidate.version_id
            key=program_key(candidate,protocol.dedup)
            if key in seen:entry['reason']='duplicate_consumes_slot'
            else:
                seen.add(key);entry['phase']='evaluation';costs['environment_evaluation_requests']+=1
                child=store.evaluate(cases,candidate)
                require(child['validator']['violating_tasks']==0 and child['validator']['coverage']==1.,'Fixed safety/coverage validator failed')
                accepted=protocol.acceptance=='strict_improvement' and validation_rank(child['validator'])<validation_rank(parent['validator'])
                entry.update(accepted=accepted,reason='validated_improvement' if accepted else 'keep_parent')
                if accepted:program,parent=candidate,child
        except Exception as error:
            entry.update(reason='failure_consumes_slot',error=type(error).__name__+': '+str(error),failure_phase=entry['phase'])
            if entry['phase']=='generation':entry['token_usage_for_failed_call']='unknown_unless_provider_receipt_available;not zero'
            # Fail closed on hidden inputs or evaluator failures. Generation/
            # parse failures consume their slot and may use only fixed fallback.
            if entry['phase'] in ('prompt','evaluation'):
                receipts.append(entry)
                return {'status':'stopped_boundary_failure','program':program,'history':history,'receipts':receipts,'costs':costs}
            if protocol.failure_fallback=='safe_tour':
                program=PatrolProgram();costs['environment_evaluation_requests']+=1;parent=store.evaluate(cases,program)
        history.append({'candidate_id':entry.get('candidate_id'),'workflow':candidate.to_dict() if candidate else None,
            'accepted':entry['accepted'],'reason':entry['reason']});receipts.append(entry)
    return {'status':'completed_bounded_slots','program':program,'history':history,'receipts':receipts,'costs':costs,
        'protocol':protocol.to_dict(),'version_id':program.version_id,'case_ids':parent['case_ids']}


def paired_protocol_update(incumbent,candidate,store,generators,cases,seed,slots=2):
    """Same default task, cases and seeds; only retained protocol leaves pair."""
    require(slots==2,'Fixed two inner proposals per paired meta arm')
    baseline=store.evaluate(cases,PatrolProgram())
    initial={'version_id':PatrolProgram().version_id,'case_ids':baseline['case_ids']}
    old=run_slots(incumbent,store,generators,cases,seed,slots,initial=initial)
    if old['status']=='stopped_boundary_failure':
        return incumbent,{'accepted':False,'status':'stopped_boundary_failure','incumbent':old,'candidate':None,
            'matched_generation_seed':seed,'initial_program':PatrolProgram().to_dict(),'history_start_lengths':[0],
            'retained_protocol':incumbent.to_dict(),'transfer_state':None,'candidate_arm_started':False}
    new=run_slots(candidate,store,generators,cases,seed,slots,initial=initial)
    valid=old['status']==new['status']=='completed_bounded_slots'
    accepted=False
    if valid and incumbent.version_id!=candidate.version_id:
        previous=store.evaluate(cases,old['program']);proposed=store.evaluate(cases,new['program'])
        accepted=validation_rank(proposed['validator'])<validation_rank(previous['validator'])
    retained=candidate if accepted else incumbent
    return retained,{'accepted':accepted,'status':'paired_engineering_or_development_update' if valid else 'stopped_boundary_failure',
        'incumbent':old,'candidate':new,'matched_generation_seed':seed,'initial_program':PatrolProgram().to_dict(),
        'history_start_lengths':[0,0],'retained_protocol':retained.to_dict(),'transfer_state':transfer_reset(retained)}
