"""Public current-action ranking and a separate archived-schedule evaluator."""
from __future__ import annotations

import numpy as np

from .information_audit import PublicSignals,score_candidates
from .limited_prediction import need
from .public_stopping import StopConfig

RANKERS=('professional_cue','public_net','public_net_no_history','time_cost_only')


def verify_geometry_subset(archived,rebuilt):
    need(set(archived)==set(rebuilt),'Closed legal geometry fields')
    for key in archived:
        value=rebuilt[key]
        if value.ndim==2 and value.shape[1]<archived[key].shape[1]:
            value=np.pad(value,((0,0),(0,archived[key].shape[1]-value.shape[1])),constant_values=-1 if key=='other_tasks' else 0)
        need(np.array_equal(archived[key],value),'Originalreturngeometry independentlyrecomputed frompublicschedule')


def public_ranks(bundle,geometry):
    """Inputs have no labels, event windows, seed, injected scenario or oracle."""
    public=PublicSignals(bundle);count=len(geometry['task'])
    if not count:return {key:np.zeros(0) for key in RANKERS}
    need(np.all(geometry['dispatch_time_s']==bundle['time_s']),'Only current dispatchable optional checks')
    ordinary=score_candidates(public,geometry,StopConfig())
    dropped=score_candidates(public,geometry,StopConfig(),drop_history=True)
    cue=ordinary['target_benefit']-ordinary['execution_cost']
    flags=(ordinary['target_signal']>=.05)&~public.alarm[geometry['task']]&(cue>0)
    cost=geometry['finish_s']-geometry['dispatch_time_s']
    time=public.weight[geometry['task']]*geometry['target_lead_s']/(cost+1)
    return {'professional_cue':np.where(flags,cue,-np.inf),
        'public_net':np.where(ordinary['admissible'],ordinary['net_score'],-np.inf),
        'public_net_no_history':np.where(dropped['admissible'],dropped['net_score'],-np.inf),
        'time_cost_only':np.where(time>0,time,-np.inf)}


def best_index(scores,geometry):
    eligible=np.flatnonzero(np.isfinite(scores)&(scores>0))
    return int(min(eligible,key=lambda i:(-scores[i],geometry['finish_s'][i],geometry['agent'][i],geometry['task'][i]))) if len(eligible) else None


def branch_schedule(tour_events,agent,task,dispatch,finish,shift):
    """Independent event-list construction: one extra, then prescribed Tour."""
    schedule=[(float(v['time_s'])+(shift if v['agent']==agent and v['time_s']>dispatch else 0),int(v['task'])) for v in tour_events]
    schedule.append((float(finish),int(task)))
    return sorted((t,j) for t,j in schedule if t<=1800)


def evaluate_schedule(schedule,intervals,B_reference):
    first=[];wait=[];weights=[]
    for event in intervals:
        found=[t for t,j in schedule if j==event['task'] and event['onset_s']<=t<=event['end_s'] and B_reference[int(t//5)-1,j]]
        when=min(found) if found else None
        first.append(when);wait.append((when if when is not None else event['end_s'])-event['onset_s']);weights.append(event['weight'])
    weight=float(sum(weights))
    return {'first_confirmation_s':first,'missed':sum(t is None for t in first),
        'weighted_wait_s':float(np.dot(wait,weights)/weight) if weight else None,
        'event_weight_sum':weight}


def delta_outcome(base,branch,shift):
    change=branch['missed']-base['missed']
    wait=branch['weighted_wait_s']-base['weighted_wait_s'] if base['weighted_wait_s'] is not None else None
    gained=sum(a is None and b is not None for a,b in zip(base['first_confirmation_s'],branch['first_confirmation_s']))
    lost=sum(a is not None and b is None for a,b in zip(base['first_confirmation_s'],branch['first_confirmation_s']))
    gain=-(wait if wait is not None else 0)-.05*shift
    return {'missed_delta':change,'weighted_wait_delta_s':wait,'gained_confirmations':gained,'lost_confirmations':lost,
        'shift_cost_s':.05*shift,'net_delay_gain_s':gain,'net_beneficial':lost==0 and change<=0 and (change<0 or gain>1e-9),
        'net_harmful':lost>0 or change>0 or (change==0 and gain< -1e-9)}
