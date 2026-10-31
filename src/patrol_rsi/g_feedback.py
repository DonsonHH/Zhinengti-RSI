"""Public decision-difference cards; later readings never label earlier truth."""
import copy
import re

from .timing_learning import project_card,reject_hidden_prompt
from .timing_skill import KINDS
from .g_timing_patch import digest,compile_patch


def difference_card(parent,child,raw,shadow,run_id,*,patch=None):
    if not re.fullmatch('[a-f0-9]{64}',run_id):raise ValueError('Opaque trajectory identity')
    if set(shadow)!={'parent_team_choice','child_team_choice'}:raise ValueError('Closed same-state team shadow')
    for choice in shadow.values():
        if choice is None:continue
        if set(choice)!={'agent','task_public_id'} or type(choice['agent']) is not int or not 0<=choice['agent']<18 or not re.fullmatch(r'(thermal|electrical|cooling|control):[0-9]+',choice['task_public_id']):raise ValueError('Public team choice only')
    projected=project_card(raw);public=projected['decision']['features']
    if projected['measurement']['time_s']<=projected['decision']['time_s']:raise ValueError('Follow-up must be strictly later than the recorded decision')
    if patch is not None and digest(compile_patch(parent,patch).to_dict())!=digest(child.to_dict()):raise ValueError('Feedback edit must compile to actual child')
    a,b=parent.action(public),child.action(public)
    if raw['decision']['skill_action']!=b:raise ValueError('Child admission must match actual current-input record')
    action=raw['decision']['executed_action']
    value={'trajectory_id':run_id,'decision':projected['decision'],'parent_skill_id':digest(parent.to_dict()),'child_skill_id':digest(child.to_dict()),'applied_patch':copy.deepcopy(patch),
        'same_input_parent_admission':a,'same_input_child_admission':b,'parent_team_choice_is_shadow':True,
        'same_state_team_choices':copy.deepcopy(shadow),'admission_differs':a!=b,'team_choice_differs':shadow['parent_team_choice']!=shadow['child_team_choice'],
        'public_effect':'selected_optional' if action=='CHECK_THIS' else 'admitted_not_selected' if b=='CHECK' else 'declined',
        'actual_follow_up':projected['measurement'],'elapsed_to_actual_measurement_s':projected['elapsed_to_actual_measurement_s'],
        'counterfactual_measurement':'unobserved','earlier_temperature_truth':'unobserved'}
    reject_hidden_prompt(value)
    return value


def retrieve_difference_cards(cards):
    """Four-profession coverage followed by at most four causal contrast examples."""
    unique={}
    for card in cards:
        reject_hidden_prompt(card)
        key=(card['trajectory_id'],card['decision']['task_public_id'],card['decision']['time_s'],card['actual_follow_up']['time_s'])
        if key in unique and unique[key]!=card:raise ValueError('Conflicting public card identity')
        unique[key]=card
    rows=list(unique.values())
    tie=lambda c:(c['decision']['time_s'],c['actual_follow_up']['time_s'],c['decision']['task_public_id'],c['trajectory_id'])
    selected=[]
    for kind in KINDS:
        pool=[v for v in rows if v['actual_follow_up']['kind']==kind]
        if pool:selected.append(min(pool,key=tie))
    pools=([
        v for v in rows if v['admission_differs'] or v['team_choice_differs']],
        [v for v in rows if v['public_effect']=='selected_optional' and v['actual_follow_up']['observed_alarm']],
        [v for v in rows if not v['actual_follow_up']['observed_alarm']],
        [v for v in rows if v['public_effect']=='admitted_not_selected'])
    for pool in pools:
        chosen=next((v for v in sorted(pool,key=tie) if v not in selected),None)
        if chosen is not None:selected.append(chosen)
    return copy.deepcopy(selected[:8])
