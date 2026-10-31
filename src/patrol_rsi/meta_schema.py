"""Strict full/compact meta proposals over the unchanged ControlledImprover DSL."""
from __future__ import annotations
import json
import math
from .improver_protocol import TOOLS,MEMORY
from .rsi_controlled import ControlledImprover,INSTRUCTION_TEMPLATES,transfer_reset

ENUMS={'generator':('qwen_once','random_program'),'dedup':('exact_version','canonical_rule_order'),
    'acceptance':('strict_improvement','keep_parent'),'failure_fallback':('keep_parent','safe_tour')}
OUTER={'diagnosis','protocol','expected_effect'}
FORMAT_CHALLENGES=({'coverage':1.0,'violating_tasks':0,'extra_checks':1,'duplicate_proposals':2},
    {'coverage':1.0,'violating_tasks':0,'extra_checks':0,'duplicate_proposals':4},
    {'coverage':1.0,'violating_tasks':0,'extra_checks':1,'duplicate_proposals':0})


def require(value,message):
    if not value:raise ValueError(message)


def strict_json(raw):
    require(isinstance(raw,str),'Raw output must be text')
    def closed(pairs):
        result={}
        for key,value in pairs:
            require(key not in result,'Duplicate JSON object key');result[key]=value
        return result
    def finite(v):
        result=float(v);require(math.isfinite(result),'Nonfinite JSON number');return result
    return json.loads(raw,object_pairs_hook=closed,parse_float=finite,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('Nonfinite JSON constant')))


def format_parents():
    result=[ControlledImprover()]
    for diagnostics,memory,template in ((['coverage','novelty'],'none',1),(['thermal_disagreement','professional_evidence'],'best_and_failures',2)):
        value=ControlledImprover().to_dict();value['protocol'].update(diagnostics=diagnostics,memory=memory,instruction=INSTRUCTION_TEMPLATES[template])
        result.append(ControlledImprover.from_dict(value))
    return result


def check_envelope(value):
    require(isinstance(value,dict) and set(value)==OUTER,'Exactly diagnosis/protocol/expected_effect')
    require(all(isinstance(value[k],str) and 1<=len(value[k])<=500 for k in ('diagnosis','expected_effect')),'Bounded diagnosis/effect text')


def compile_compact(value):
    """Explicit NEW interface: model chooses leaves, compiler supplies nesting only."""
    require(isinstance(value,dict) and set(value)=={'diagnostic_ids','memory_id','instruction_id'}|set(ENUMS),'Closed compact leaf schema')
    ids=value['diagnostic_ids']
    require(isinstance(ids,list) and 1<=len(ids)<=3 and all(type(v) is int and 0<=v<len(TOOLS) for v in ids) and len(set(ids))==len(ids),'One to three distinct diagnostic IDs')
    for key,options in (('memory_id',MEMORY),('instruction_id',INSTRUCTION_TEMPLATES)):
        require(type(value[key]) is int and 0<=value[key]<len(options),'Compact integer index outside fixed enum')
    for key,options in ENUMS.items():require(type(value[key]) is str and value[key] in options,'Compact enum outside unchanged DSL')
    result={'protocol':{'diagnostics':[TOOLS[v] for v in ids],'memory':MEMORY[value['memory_id']],
        'instruction':INSTRUCTION_TEMPLATES[value['instruction_id']]},**{k:value[k] for k in ENUMS}}
    return ControlledImprover.from_dict(result)


def parse_proposal(raw,method):
    value=strict_json(raw);check_envelope(value)
    require(method in ('full_nested','compact_leaves'),'Frozen E0 interface only')
    proposal=ControlledImprover.from_dict(value['protocol']) if method=='full_nested' else compile_compact(value['protocol'])
    return proposal,value


def full_schema():
    return {'type':'object','additionalProperties':False,'required':sorted(OUTER),'properties':{
        'diagnosis':{'type':'string','minLength':1,'maxLength':500},'expected_effect':{'type':'string','minLength':1,'maxLength':500},
        'protocol':{'type':'object','additionalProperties':False,'required':['protocol',*ENUMS],
            'properties':{'protocol':{'type':'object','additionalProperties':False,'required':['diagnostics','memory','instruction'],
                'properties':{'diagnostics':{'type':'array','minItems':1,'maxItems':3,'uniqueItems':True,'items':{'enum':list(TOOLS)}},
                    'memory':{'enum':list(MEMORY)},'instruction':{'enum':list(INSTRUCTION_TEMPLATES)}}},
                **{key:{'enum':list(values)} for key,values in ENUMS.items()}}}}}


def compact_schema():
    value=full_schema();value['properties']['protocol']={'type':'object','additionalProperties':False,
        'required':['diagnostic_ids','memory_id','instruction_id',*ENUMS],
        'properties':{'diagnostic_ids':{'type':'array','minItems':1,'maxItems':3,'uniqueItems':True,'items':{'type':'integer','minimum':0,'maximum':4}},
            'memory_id':{'type':'integer','minimum':0,'maximum':2},'instruction_id':{'type':'integer','minimum':0,'maximum':2},
            **{key:{'enum':list(values)} for key,values in ENUMS.items()}}}
    return value


def prompt(method,parent,challenge):
    require(type(challenge) is int and 0<=challenge<3,'Three frozen synthetic format challenges')
    public=FORMAT_CHALLENGES[challenge]
    example=ControlledImprover.from_dict(parent.to_dict())
    leaf={'diagnostic_ids':[0,1],'memory_id':2,'instruction_id':1,**{k:example.to_dict()[k] for k in ENUMS}}
    nested=compile_compact(leaf).to_dict()
    return {'purpose':'FORMAT PREFLIGHT ONLY. Synthetic public diagnostics, not task evidence or a self-improvement result.',
        'task':'Propose one valid protocol different from the parent. Choose1..3 DISTINCT diagnostics; preserve all tool, safety, evaluator and budget boundaries.',
        'interface':method,'parent_protocol':parent.to_dict(),'synthetic_public_diagnostics':public,
        'allowed_diagnostic_order':list(TOOLS),'allowed_memory_order':list(MEMORY),'allowed_instruction_templates_order':list(INSTRUCTION_TEMPLATES),
        'response_schema':full_schema() if method=='full_nested' else compact_schema(),
        'complete_valid_example':{'diagnosis':'Compare professional evidence and failures.','protocol':nested if method=='full_nested' else leaf,
            'expected_effect':'A legal alternative for later independent evaluation; no performance claim.'},
        'important':'Full interface has TWO protocol keys at different nesting levels. Compact interface is NEW deterministic structural assembly, never a repair of invalid full outputs. No markdown fences, extra keys, task IDs, seeds, event times or private labels.'}


def changed_fields(parent,child):
    before=parent.to_dict();after=child.to_dict();keys=[]
    for key in ('diagnostics','memory','instruction'):
        if before['protocol'][key]!=after['protocol'][key]:keys.append('protocol.'+key)
    keys.extend(k for k in ENUMS if before[k]!=after[k])
    return keys


def reset_receipt(protocol):
    reset=transfer_reset(protocol)
    require(reset['program'].rules==() and reset['program'].fallback=='tour' and not reset['history'] and not reset['candidate_archive'] and not reset['task_outcomes'],'Only protocol transfers to empty task state')
    return {'only_protocol_transferred':True,'program':reset['program'].to_dict(),'history_length':0,'candidate_archive_length':0,'task_outcomes_length':0}
