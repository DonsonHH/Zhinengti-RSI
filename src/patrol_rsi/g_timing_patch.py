"""Atomic bounded edits to an immutable TimingSkill, using stable parent IDs."""
import copy
import hashlib
import json

from .timing_skill import TimingSkill,validate_atom,NUMERIC_DOMAINS
from .meta_schema import strict_json


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def patch_view(parent):
    value=parent.to_dict();rules={};predicates={}
    for i,rule in enumerate(value['rules']):
        rid=f'r{i}';rules[rid]={'hash':digest(rule),'rule':copy.deepcopy(rule)}
        for j,atom in enumerate(rule['all']):predicates[f'{rid}.p{j}']={'hash':digest(atom),'predicate':copy.deepcopy(atom)}
    return {'parent_id':digest(value),'rules':rules,'predicates':predicates}


def compile_patch(parent,patch):
    if not isinstance(patch,dict) or set(patch)!={'schema_version','parent_id','edits'}:raise ValueError('Closed patch schema')
    view=patch_view(parent)
    if patch['schema_version']!='timing-patch-v1' or patch['parent_id']!=view['parent_id']:raise ValueError('Exact current parent content hash')
    if not isinstance(patch['edits'],list) or not 1<=len(patch['edits'])<=2:raise ValueError('One or two atomic edits')
    rules={f'r{i}':{f'r{i}.p{j}':copy.deepcopy(atom) for j,atom in enumerate(rule['all'])} for i,rule in enumerate(parent.to_dict()['rules'])}
    schemas={'replace_threshold':{'op','target','expected_hash','value'},'delete_predicate':{'op','target','expected_hash'},
        'add_predicate':{'op','target','expected_hash','predicate'},'delete_rule':{'op','target','expected_hash'},'add_rule':{'op','rule'}}
    for index,edit in enumerate(patch['edits']):
        if not isinstance(edit,dict) or type(edit.get('op')) is not str or edit['op'] not in schemas or set(edit)!=schemas[edit['op']]:raise ValueError('Closed registered edit operation')
        op=edit['op']
        if op=='add_rule':
            check=TimingSkill.from_dict({'schema_version':'timing-skill-v1','rules':[edit['rule']],'default_action':'CONTINUE_TOUR'})
            rules[f'new{index}']={f'new{index}.p{j}':a for j,a in enumerate(check.to_dict()['rules'][0]['all'])};continue
        target=edit['target'];table=view['predicates'] if op in ('replace_threshold','delete_predicate') else view['rules']
        if type(target) is not str or target not in table or edit['expected_hash']!=table[target]['hash']:raise ValueError('Original node precondition')
        rid=target.split('.')[0]
        if rid not in rules:raise ValueError('Target rule was deleted')
        if op in ('replace_threshold','delete_predicate'):
            if target not in rules[rid]:raise ValueError('Target predicate was deleted')
            if op=='delete_predicate':del rules[rid][target]
            else:
                atom=rules[rid][target]
                if atom['field'] not in set(NUMERIC_DOMAINS)|{'evidence_margin'} or atom['op']=='is_missing':raise ValueError('Threshold replacement only for numeric predicates')
                atom['value']=copy.deepcopy(edit['value']);validate_atom(atom)
        elif op=='delete_rule':del rules[rid]
        else:validate_atom(edit['predicate']);rules[rid][f'added{index}']=copy.deepcopy(edit['predicate'])
    return TimingSkill.from_dict({'schema_version':'timing-skill-v1','rules':[{'all':list(v.values())} for v in rules.values()],'default_action':'CONTINUE_TOUR'})


def parse_patch_reply(text,parent):
    value=strict_json(text)
    if not isinstance(value,dict) or set(value)!={'diagnosis','patch','expected_effect'} or any(type(value[k]) is not str or not 1<=len(value[k])<=500 for k in ('diagnosis','expected_effect')):raise ValueError('Closed patch response envelope')
    return compile_patch(parent,value['patch']),value['patch']
