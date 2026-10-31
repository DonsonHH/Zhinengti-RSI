"""Bounded editable timing predicates; physical action legality stays external."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math

BOOL_FIELDS={'professional_available','direct_available','direct_reading_normal','direct_is_qualified','estimate_available','already_alarm','ever_checked','quota_available'}
NUMERIC_DOMAINS={
    'professional_signal':(0,.05,.1,.25,.5,1),
    'inspection_age_ratio':(0,.25,.5,.75,1),
    'direct_age_s':(0,30,60,120,300,600),
    'direct_age_ratio':(0,.25,.5,.75,1),
    'estimate_age_s':(0,15,30,60,120),
    'thermal_disagreement':(0,.025,.05,.1,.25,.5,1),
    'prior_disagreement':(0,.025,.05,.1,.25,.5,1),
    'pod_thermal_signal':(0,.05,.1,.25,.5,1),
    'pod_flow_deficit':(0,.05,.1,.25,.5,1),
    'pod_service_gap':(0,.05,.1,.25,.5,1),
    'lead_s':(0,15,30,60,120,300),
    'shift_s':(0,5,15,30,60,120),
    'completion_cost_s':(0,15,30,60,120,300),
    'importance':(.25,.5,.75,1)}
FIELDS=set(NUMERIC_DOMAINS)|BOOL_FIELDS|{'kind'}
KINDS=('thermal','electrical','cooling','control')
MARGIN_COEFFICIENTS=(0,.01,.05,.1,.25,1)


def require(value,message):
    if not value:raise ValueError(message)


def validate_atom(atom):
    require(isinstance(atom,dict) and set(atom)=={'field','op','value'},'Exactly field/op/value per predicate')
    field,op,value=(atom[k] for k in ('field','op','value'))
    require(type(field) is str and type(op) is str,'Literal field/operator names')
    require(field in FIELDS|{'evidence_margin'},'Only registered causal public fields')
    if field=='evidence_margin':
        require(op=='positive' and type(value) in (int,float) and value in MARGIN_COEFFICIENTS,'Bounded public evidence/shift tradeoff')
    elif op=='is_missing':require(value is None,'Missingness predicate has null value')
    elif field in BOOL_FIELDS:require(op=='eq' and type(value) is bool,'Boolean predicate requires literal true/false')
    elif field=='kind':require(op=='eq' and value in KINDS,'Known profession only')
    else:
        require(op in ('ge','gt','le','lt'),'Registered numeric comparison')
        reference=field=='professional_signal' and value=='kind_threshold'
        require(reference or type(value) in (int,float) and value in NUMERIC_DOMAINS[field] and math.isfinite(value),'Threshold outside predeclared domain')
    return field,op,value


@dataclass(frozen=True)
class TimingSkill:
    rules:tuple[tuple[tuple[str,str,object],...],...]

    def __post_init__(self):
        require(isinstance(self.rules,tuple) and len(self.rules)<=3,'Bounded immutable skill rules')
        for rule in self.rules:
            require(isinstance(rule,tuple) and 1<=len(rule)<=6,'One to six immutable AND predicates')
            for atom in rule:
                require(isinstance(atom,tuple) and len(atom)==3,'Immutable predicate triple')
                validate_atom(dict(zip(('field','op','value'),atom)))

    @classmethod
    def from_dict(cls,value):
        require(isinstance(value,dict) and set(value)=={'schema_version','rules','default_action'},'Closed TimingSkill schema')
        require(value['schema_version']=='timing-skill-v1' and value['default_action']=='CONTINUE_TOUR','Fixed skill version and safe default')
        require(isinstance(value['rules'],list) and len(value['rules'])<=3,'At most three OR rules')
        rules=[]
        for row in value['rules']:
            require(isinstance(row,dict) and set(row)=={'all'} and isinstance(row['all'],list) and 1<=len(row['all'])<=6,'One to six AND predicates, depth two')
            rules.append(tuple(validate_atom(v) for v in row['all']))
        return cls(tuple(rules))

    @classmethod
    def initial_e1(cls):
        return cls.from_dict({'schema_version':'timing-skill-v1','rules':[{'all':[
            {'field':'professional_available','op':'eq','value':True},
            {'field':'professional_signal','op':'ge','value':'kind_threshold'},
            {'field':'already_alarm','op':'eq','value':False},
            {'field':'lead_s','op':'gt','value':0},
            {'field':'evidence_margin','op':'positive','value':.05}]}],'default_action':'CONTINUE_TOUR'})

    def to_dict(self):
        return {'schema_version':'timing-skill-v1','rules':[{'all':[{'field':f,'op':o,'value':v} for f,o,v in rule]} for rule in self.rules],'default_action':'CONTINUE_TOUR'}

    @property
    def version_id(self):return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True).encode()).hexdigest()[:16]

    def decide(self,public):
        require(isinstance(public,dict) and set(public)<=FIELDS,'Only typed public timing features, no seed/event/Oracle metadata')
        for field,value in public.items():
            if value is None:continue
            if field in BOOL_FIELDS:require(type(value) is bool,'Boolean public feature')
            elif field=='kind':require(value in KINDS,'Known public profession')
            else:
                try:finite=type(value) in (int,float) and math.isfinite(value) and value>=0
                except OverflowError:finite=False
                require(finite,'Finite nonnegative public scalar or explicit missingness')
        if public.get('direct_available') is False:
            require(all(public.get(k) is None for k in ('direct_age_s','direct_age_ratio','direct_reading_normal','direct_is_qualified')),'Unavailable direct reading has no age, normality or qualification')
        if public.get('direct_reading_normal') is not None:
            require(public.get('direct_available') is True,'Known direct normality requires an available reading')
        if public.get('professional_available') is False:
            require(public.get('professional_signal') is None,'Unavailable professional signal is missing, not zero')
        if public.get('estimate_available') is False:
            require(public.get('estimate_age_s') is None,'Unavailable estimate has no timestamp age')
        def test(atom):
            field,op,threshold=atom
            if field=='evidence_margin':
                values=[public.get(k) for k in ('professional_signal','importance','lead_s','shift_s')]
                return all(v is not None for v in values) and values[0]*values[1]*values[2]>threshold*values[3]
            value=public.get(field)
            if op=='is_missing':return value is None
            if value is None:return False
            if threshold=='kind_threshold':
                if public.get('kind') not in KINDS:return False
                threshold=.5 if public['kind']=='control' else .05
            if op=='eq':return value==threshold
            if op=='ge':return value>=threshold
            if op=='gt':return value>threshold
            if op=='le':return value<=threshold
            return value<threshold
        for index,rule in enumerate(self.rules):
            if all(test(v) for v in rule):return {'action':'CHECK','matched_rule':index}
        return {'action':'CONTINUE_TOUR','matched_rule':None}

    def action(self,public):return self.decide(public)['action']


def information_baseline():
    value=TimingSkill.initial_e1().to_dict()
    for predicate in ({'field':'direct_age_s','op':'ge','value':120},{'field':'direct_age_s','op':'is_missing','value':None}):
        value['rules'].append({'all':[{'field':'already_alarm','op':'eq','value':False},predicate,
            {'field':'inspection_age_ratio','op':'ge','value':.5},{'field':'lead_s','op':'ge','value':60},{'field':'shift_s','op':'le','value':30}]})
    return TimingSkill.from_dict(value)
