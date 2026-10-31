"""Typed timing proposals and causal-public experience for a fixed improver."""
import json
import hashlib
import math
import re
import numpy as np
from collections import defaultdict

from .timing_skill import TimingSkill,KINDS,FIELDS,BOOL_FIELDS,NUMERIC_DOMAINS,MARGIN_COEFFICIENTS
from .rsi_controlled import ControlledImprover,INSTRUCTION_TEMPLATES
from .improver_protocol import TOOLS,MEMORY
from .timing_skill_features import FEATURE_MEANINGS
from .meta_schema import strict_json


def exact_policy_result(results,policy):
    """Use the cache's exact JSON identity, including numeric representation."""
    encode=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    wanted=encode(policy);matches=[v for v in results if encode(v['policy'])==wanted]
    if len(matches)!=1:raise ValueError('Exactly one immutable evaluation of the exact serialized policy required')
    return matches[0]


def aggregate(rows,expected_specs):
    """Noise -> physical process -> seed -> macro; healthy wait is undefined."""
    if not rows or not expected_specs:raise ValueError('Nonempty frozen cohort and completed evaluation required')
    key=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
    actual=[key(v['specification']) for v in rows];expected=[key(v) for v in expected_specs]
    if len(set(actual))!=len(actual) or len(set(expected))!=len(expected) or set(actual)!=set(expected):
        raise ValueError('Exactly the frozen unique condition cohort required')
    processes=defaultdict(list)
    for row in rows:
        metrics=row['metrics']
        if any(type(metrics[k]) is not int or metrics[k]<0 for k in ('anomaly_segments','missed_segments','selected_detours')):
            raise ValueError('Nonnegative integer event/check counts required')
        if metrics['missed_segments']>metrics['anomaly_segments'] or metrics['selected_detours']>1:
            raise ValueError('Physical event count and single-check bounds')
        for field in ('travel_m','consequence_weighted_censored_delay_s'):
            value=metrics[field]
            if value is None and field!='travel_m':continue
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                raise ValueError('Finite nonnegative selection metrics')
        s=row['specification'];key=tuple(s[k] for k in ('seed','scenario','duration_s','fault_start_s'))
        processes[key].append(row)
    seeds=defaultdict(list)
    mean=lambda values:sum(values)/len(values) if values else None
    identities=[];seen_fingerprints={}
    for physical,group in processes.items():
        if len(group)!=2 or {v['specification']['noise_c'] for v in group}!={.2,1.}:
            raise ValueError('Exactly the two preregistered paired noises per physical process')
        if len({v['process_fingerprint'] for v in group})!=1 or len({v['event_eligibility_SHA256'] for v in group})!=1:
            raise ValueError('Paired noise must share physical process and event eligibility')
        fp=group[0]['process_fingerprint']
        if fp in seen_fingerprints and seen_fingerprints[fp]!=physical:
            raise ValueError('Different process labels cannot duplicate the same numerical process')
        seen_fingerprints[fp]=physical
        values=[v['metrics'] for v in group];waits=[v['consequence_weighted_censored_delay_s'] for v in values]
        if len({v['anomaly_segments'] for v in values})!=1 or any((w is None)!=(values[0]['anomaly_segments']==0) for w in waits):
            raise ValueError('Wait eligibility must match identical physical events in both noises')
        identities.append([list(physical),group[0]['process_fingerprint'],group[0]['event_eligibility_SHA256']])
        seeds[physical[0]].append({'miss':mean([v['missed_segments'] for v in values]),
            'wait':mean([v for v in waits if v is not None])})
    result=[]
    for seed,values in sorted(seeds.items()):
        result.append({'seed':seed,'macro_missed_count':mean([v['miss'] for v in values]),
            'macro_wait_s':mean([v['wait'] for v in values if v['wait'] is not None])})
    return {'seeds':result,'macro_missed_count':mean([v['macro_missed_count'] for v in result]),
        'macro_wait_s':mean([v['macro_wait_s'] for v in result if v['macro_wait_s'] is not None]),
        'mean_CHECK':mean([r['metrics']['selected_detours'] for r in rows]),
        'mean_walk_m':mean([r['metrics']['travel_m'] for r in rows]),'conditions':len(rows),'physical_processes':len(processes),
        'cohort_SHA256':hashlib.sha256(json.dumps(sorted(actual)).encode()).hexdigest(),
        'process_event_identities':sorted(identities)}


TOL=1e-9


def selection_key(value):
    return (value['macro_missed_count'],value['macro_wait_s'],value['mean_CHECK'],value['mean_walk_m'])


def per_seed_no_more_misses(candidate,baseline):
    if candidate['cohort_SHA256']!=baseline['cohort_SHA256'] or candidate['process_event_identities']!=baseline['process_event_identities']:
        raise ValueError('Same frozen cohort and physical event eligibility across policies')
    b={v['seed']:v['macro_missed_count'] for v in baseline['seeds']}
    if {v['seed'] for v in candidate['seeds']}!=set(b):raise ValueError('Paired identical seed groups required')
    return all(v['macro_missed_count']<=b[v['seed']]+TOL for v in candidate['seeds'])


def nonregression(candidate,baseline):
    if not per_seed_no_more_misses(candidate,baseline):return False
    cm,bm=candidate['macro_missed_count'],baseline['macro_missed_count']
    if cm<bm-TOL:return True
    if abs(cm-bm)>TOL:return False
    c,b=candidate['macro_wait_s'],baseline['macro_wait_s']
    return c is None and b is None or c is not None and b is not None and c<=b+TOL


def strict_gain(candidate,baseline):
    if not per_seed_no_more_misses(candidate,baseline):return False
    cm,bm=candidate['macro_missed_count'],baseline['macro_missed_count']
    if cm<bm-TOL:return True
    if abs(cm-bm)>TOL:return False
    c,b=candidate['macro_wait_s'],baseline['macro_wait_s']
    if c is not None and b is not None and b>TOL and (b-c)/b>=.05-TOL:return True
    same_wait=c is None and b is None or c is not None and b is not None and abs(c-b)<=TOL
    checks=baseline['mean_CHECK']
    return bool(same_wait and checks>TOL and (checks-candidate['mean_CHECK'])/checks>=.05-TOL
        and candidate['mean_walk_m']<=baseline['mean_walk_m']+TOL)


def strictly_better_key(candidate,baseline):
    if (candidate['macro_wait_s'] is None)!=(baseline['macro_wait_s'] is None):return False
    for c,b in zip(selection_key(candidate),selection_key(baseline)):
        if c is None and b is None:continue
        if abs(c-b)>TOL:return c<b
    return False


def sample_skill(seed):
    rng=np.random.default_rng(seed);fields=sorted(FIELDS|{'evidence_margin'});rules=[]
    for _ in range(int(rng.integers(0,4))):
        atoms=[]
        for _ in range(int(rng.integers(1,7))):
            field=fields[int(rng.integers(len(fields)))];value=None
            if field=='evidence_margin':op='positive';values=MARGIN_COEFFICIENTS
            elif field in BOOL_FIELDS:op=('eq','is_missing')[int(rng.integers(2))];values=(False,True)
            elif field=='kind':op=('eq','is_missing')[int(rng.integers(2))];values=KINDS
            else:
                op=('ge','gt','le','lt','is_missing')[int(rng.integers(5))]
                values=NUMERIC_DOMAINS[field]+(('kind_threshold',) if field=='professional_signal' else ())
            if op!='is_missing':value=values[int(rng.integers(len(values)))]
            atoms.append({'field':field,'op':op,'value':value})
        rules.append({'all':atoms})
    return TimingSkill.from_dict({'schema_version':'timing-skill-v1','rules':rules,'default_action':'CONTINUE_TOUR'})


def project_card(card):
    """Only an actual later qualified reading may label a prior public choice."""
    decision=card['decision'];measurement=card['measurement'];features=decision['features']
    TimingSkill.initial_e1().decide(features)
    kind=measurement['kind']
    field,unit={'thermal':('temperature_c','degC'),'electrical':('contact_temperature_c','degC'),
        'cooling':('pod_flow_kg_s','kg/s'),'control':('controller_sensor_error_c','degC')}[kind]
    if features['kind']!=kind or measurement['field']!=field or measurement['unit']!=unit:
        raise ValueError('Same-profession actual measurement and public decision required')
    if measurement['qualified_role']!=kind or measurement['calibrated'] is not True:
        raise ValueError('Actual qualified calibrated measurement only')
    start,end=decision['time_s'],measurement['time_s']
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in (start,end,measurement['value'])) or end<start:
        raise ValueError('Ordered finite causal observation times')
    if abs(card['elapsed_to_actual_measurement_s']-(end-start))>1e-9:
        raise ValueError('Actual elapsed feedback time')
    if not re.fullmatch(r'(thermal|electrical|cooling|control):[0-9]+',decision['task_public_id']):
        raise ValueError('Only public professional task identity')
    if type(measurement['observed_alarm']) is not bool or measurement['transition'] not in (
        'new_measured_alert','persistent_recheck','recovery_confirmation','normal_check'):
        raise ValueError('Observed measurement transition only')
    fields=('time_s','task_public_id','agent','features','skill_action','executed_action',
        'team_action','candidate_finish_s','Tour_target_finish_s','matched_rule_index','check_intent')
    selected={k:decision[k] for k in fields if k in decision}
    observed={k:measurement[k] for k in ('time_s','kind','value','field','unit','qualified_role','calibrated','observed_alarm','transition')}
    return json.loads(json.dumps({'decision':selected,'measurement':observed,'elapsed_to_actual_measurement_s':end-start},allow_nan=False))


def retrieve_cards(cards,diagnostics):
    """At most an observed-alert and normal contrast per profession, no truth rank."""
    projected=[project_card(card) for card in cards]
    def order(card):
        f=card['decision']['features'];values=[]
        for tool in diagnostics:
            if tool=='professional_evidence':values.append(-int(card['measurement']['observed_alarm']))
            elif tool=='coverage':values.append(-float(f.get('inspection_age_ratio') or 0))
            elif tool=='efficiency':values.append(-float(f.get('shift_s') or 0))
            elif tool=='thermal_disagreement':values.append(-float(f.get('thermal_disagreement') or 0))
            elif tool=='novelty':values.append(-sum(v is None for v in f.values()))
        # These tie-breakers are publicly known, not episode performance.
        return tuple(values)+(-float(f.get('direct_age_s') or 0),card['decision']['time_s'],card['decision']['task_public_id'])
    primary=[];contrasts=[]
    for kind in KINDS:
        chosen=[]
        for alarm in (True,False):
            pool=[v for v in projected if v['measurement']['kind']==kind and v['measurement']['observed_alarm'] is alarm]
            if pool:chosen.append(min(pool,key=order))
        if chosen:primary.append(chosen[0]);contrasts.extend(chosen[1:])
    return primary+contrasts


def parse_skill_reply(text):
    value=strict_json(text)
    if not isinstance(value,dict) or set(value)!={'diagnosis','skill','expected_effect'}:
        raise ValueError('Exactly diagnosis, skill and expected_effect')
    if any(type(value[k]) is not str or not 1<=len(value[k])<=1500 for k in ('diagnosis','expected_effect')):
        raise ValueError('Bounded nonempty explanation strings')
    return TimingSkill.from_dict(value['skill'])


def parse_meta_reply(text):
    value=strict_json(text)
    if not isinstance(value,dict) or set(value)!={'diagnosis','protocol','expected_effect'}:
        raise ValueError('Exactly diagnosis, protocol and expected_effect')
    if any(type(value[k]) is not str or not 1<=len(value[k])<=1500 for k in ('diagnosis','expected_effect')):
        raise ValueError('Bounded nonempty explanation strings')
    result=ControlledImprover.from_dict(value['protocol'])
    if result.acceptance!='strict_improvement' or result.failure_fallback!='keep_parent':
        raise ValueError('Immutable strict acceptance and parent failure contract')
    return result


FORBIDDEN_PROMPT_KEYS={'seed','scenario','fault_start_s','fault_end_s','onset_s','end_s','oracle','truth','anomaly_intervals',
    'missed_segments','macro_missed_count','macro_wait_s','weighted_delay_sum','test_results','final_test','process_fingerprint','record_path','metrics'}


def reject_hidden_prompt(value):
    if isinstance(value,dict):
        if set(value)&FORBIDDEN_PROMPT_KEYS:raise ValueError('Hidden evaluator or process identity in prompt')
        for item in value.values():reject_hidden_prompt(item)
    elif isinstance(value,(tuple,list)):
        for item in value:reject_hidden_prompt(item)


def skill_prompt(context):
    reject_hidden_prompt(context)
    domains={field:{'op':['ge','gt','le','lt','is_missing'],'thresholds':list(values),
        'missing':'is_missing requires null; numeric comparison with missing is false'} for field,values in NUMERIC_DOMAINS.items()}
    domains['professional_signal']['thresholds'].append('kind_threshold')
    value={'task':'Propose ONE executable TimingSkill to improve inspection timing using only supplied causal public experience. A distinct skill is useful only if its actual CHECK/CONTINUE_TOUR behavior changes.',
        'common_contract':'Known static layout,18 qualified owned executors,1800s; at most ONE additional check for the whole team. CONTINUE_TOUR continues mandatory work and preserves quota. External original legality,60s cooldown,fullreturn and exactrevisit are immutable. Do not invent observations or read future faults.',
        'skill_schema':{'keys':['schema_version','rules','default_action'],'schema_version':'timing-skill-v1','default_action':'CONTINUE_TOUR',
            'rules':'0..3 OR rules; each exactly {all:[1..6 AND predicates]}; CHECK iff any complete rule true; otherwise CONTINUE_TOUR',
            'predicate_keys':['field','op','value'],'boolean_fields':sorted(BOOL_FIELDS),'boolean_ops':{'eq':'literal true/false','is_missing':None},
            'kind':{'eq':list(KINDS),'is_missing':None},'numeric_fields':domains,
            'evidence_margin':{'op':'positive','value':list(MARGIN_COEFFICIENTS),'meaning':'professional_signal*importance*lead_s > value*shift_s; missing terms=false'},
            'kind_threshold':'.5 for control, .05 otherwise; profession must be known'},
        'public_feature_meanings':dict(FEATURE_MEANINGS,already_alarm='Already measured public alarm, not hidden truth. Rechecking it is not a new physical-event discovery.',
            professional_available='An actual own-profession cue exists; false makes professional_signal null.',ever_checked='At least one qualified check completed.',
            quota_available='True for candidates before the single additional opportunity is consumed.',estimate_available='Temperature-CVAE tool output exists; timestamp can still be unknown.',
            importance='Public normalized consequence importance, not a fault probability.',direct_available='Actual direct channel exists, including remote readings where explicitly allowed.'),
        'data':context,'output_envelope':{'diagnosis':'short string,1..1500 chars','skill':'exact TimingSkill object','expected_effect':'short string,1..1500 chars'},
        'response_keys':['diagnosis','skill','expected_effect'],
        'format_example':{'diagnosis':'The initial reference is supplied in data.','skill':TimingSkill.initial_e1().to_dict(),'expected_effect':'This is the initial reference only, not an improved answer.'}}
    reject_hidden_prompt(value)
    return value


def meta_prompt(incumbent,context):
    reject_hidden_prompt(context)
    return {'task':'Propose ONE ControlledImprover protocol for generating TimingSkill objects on later reset tasks. Do not propose a task skill. Full nested protocol is mandatory.',
        'current_protocol':incumbent.to_dict(),'completed_causal_experience':context,
        'schema':{'outer_keys':['protocol','generator','dedup','acceptance','failure_fallback'],
            'inner_protocol_keys':['diagnostics','memory','instruction'],'diagnostics':'Choose1..3 distinct tools, order matters',
            'diagnostic_choices':list(TOOLS),'memory_choices':list(MEMORY),'instruction_choices':list(INSTRUCTION_TEMPLATES),
            'generator':['qwen_once','random_program'],'dedup':['exact_version','canonical_rule_order'],
            'acceptance':['strict_improvement'],'failure_fallback':['keep_parent']},
        'response_keys':['diagnosis','protocol','expected_effect'],
        'output_envelope':{'diagnosis':'short string','protocol':incumbent.to_dict(),'expected_effect':'short string'},
        'reset_contract':'Only this validated protocol transfers. Newtask always starts initialE1 and empty history/archive. No model,skilllanguage,qualification,quota,legality,evaluator,budget or data change.'}


def skill_context(initial,history,baseline_cards,*,protocol=None,prior_only=False):
    neutral=[]
    for row in history[-2:]:
        if row.get('skill') is not None:
            neutral.append({'skill':TimingSkill.from_dict(row['skill']).to_dict()})
        else:neutral.append({'schema_error':'previous output was not a valid TimingSkill'})
    protocol=protocol or ControlledImprover()
    value={'initial_skill':initial.to_dict(),'proposal_history':neutral,
        'diagnostics':list(protocol.protocol.diagnostics),'instruction':protocol.protocol.instruction,
        'experience_cards':[],'acceptance_feedback':[]}
    if prior_only:return value
    indexed=list(enumerate(history))
    if protocol.protocol.memory=='none':memory=[]
    elif protocol.protocol.memory=='recent':memory=indexed[-2:]
    else:
        accepted=[(i,v) for i,v in indexed if v.get('accepted')]
        rejected=[(i,v) for i,v in indexed if not v.get('accepted')]
        memory=accepted[-1:]+rejected[-1:]
    cards=list(baseline_cards)+[card for _,row in memory for card in row.get('cards',[])]
    value['experience_cards']=retrieve_cards(cards,protocol.protocol.diagnostics)
    value['acceptance_feedback']=[{'candidate_id':TimingSkill.from_dict(v['skill']).version_id,
        'previous_proposal_offset':len(history)-i,'accepted':bool(v['accepted'])}
        for i,v in memory if v.get('skill') is not None and 'accepted' in v]
    return value
