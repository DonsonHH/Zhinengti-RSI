"""One-budget stopping over a deterministic public tour forecast to 1800s.

CHECK consumes the action and follows TOUR. WAIT keeps it and applies this
same stop/continue recursion at the later dispatch. No future readings are
invented; the real controller rebuilds the model from new causal observations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .causal_observation import task_signal, validate_public
from .inspection_timing import return_slack


@dataclass(frozen=True)
class StopConfig:
    collateral: float = 1.
    future_risk: float = .25
    unknown_risk: float = .05
    tour_soon_s: int = 60
    threshold: float = 0.

    def __post_init__(self):
        if self.collateral not in (0.,.5,1.) or self.future_risk not in (0.,.25) or self.unknown_risk!=.05:
            raise ValueError('Frozen small coefficient domain')
        if self.tour_soon_s not in (0,60) or self.threshold not in (0.,.01):
            raise ValueError('Frozen gate domain')


def _exposure(at,shift,period,end=1800.):
    visits=[]
    while at<=end:
        exposure=max(0.,min(at+shift,end)-at)
        visits.append((at,exposure));at+=period
    return visits


def public_frontier(bundle,config=StopConfig()):
    """All positive-risk legal optional CHECKs under TOUR until mission end.

    Each score uses the same absolute 1800s horizon and fixed observation
    forecast. It is a declared surrogate, not expected true anomaly utility.
    """
    validate_public(bundle)
    root=bundle['time_s'];tasks=bundle['tasks'];result=[]
    pod_signals={}
    for row in tasks:
        sig=task_signal(row,0.)[0]
        if sig is not None:pod_signals.setdefault(row['pod'],[]).append((row['task'],sig))
    for agent in bundle['agents']:
        tour=agent['tour'];cursor=agent['cursor'];edge=agent['costs_s'];costs=agent['first_costs_s']
        if agent['period_s']<=0:raise ValueError('Positive cycle required')
        last={j:max(0.,tasks[j]['last_check_s']) for j in tour}
        limits={j:tasks[j]['deadline_s'] for j in tour}
        t=agent['busy_finish_s'] if agent['busy_target'] is not None else root
        completed=agent['baseline_completed'];detours=agent['detours_used']
        if agent['busy_target'] is not None:
            j=agent['busy_target']
            if j!=tour[cursor]:raise ValueError('Forecast is only used before optional action')
            last[j]=t;costs=edge[str(j)];cursor=(cursor+1)%len(tour);completed+=1
            if completed==len(tour):completed=detours=0
        possible=[j for j in tour if (task_signal(tasks[j],1800-root)[0] or 0)>0 and not tasks[j]['already_alarm']]
        if not possible:continue
        while t<=1795:
            base=tour[cursor];base_finish=t+costs[str(base)]
            if base_finish>1800:break
            order=tour[cursor:]+tour[:cursor];arrivals={};at=t;previous=None
            for j in order:
                at+=costs[str(j)] if previous is None else edge[str(previous)][str(j)]
                arrivals[j]=at;previous=j
            base_slack=return_slack(t,costs,last,tour,cursor,base,limits,edge)
            if base_slack is None:raise ValueError('Public TOUR forecast infeasible')
            for j in possible:
                row=tasks[j];finish=t+costs[str(j)]
                if j==base or detours>=2 or finish>1800 or t-last[j]<60 and (last[j]>0 or row['last_check_s']>=0):continue
                slack=return_slack(t,costs,last,tour,cursor,j,limits,edge)
                if slack is None:continue
                signal,source=task_signal(row,finish-root)
                if signal is None or signal<=0:continue
                lead=max(0.,min(arrivals[j],1800.)-finish)
                target_benefit=row['weight_scaled']*signal*lead/300
                shift=costs[str(j)]+edge[str(j)][str(base)]-costs[str(base)]
                shift=max(0.,shift)
                first_cost=future_cost=exposure=0.;unknown=[];cost_rows=[]
                for k in order:
                    if k==j:continue
                    other=tasks[k];sig,src=task_signal(other,min(arrivals[k]+shift,1800)-root)
                    risk=sig if sig is not None else config.unknown_risk
                    if sig is None:unknown.append(k)
                    # Public same-Pod support is a bounded risk correction,
                    # not a hidden scenario classifier or failure probability.
                    linked=max((v for idx,v in pod_signals.get(other['pod'],[]) if idx!=k),default=0.)
                    corrected=risk+.05*min(linked,1.)
                    visits=_exposure(arrivals[k],shift,agent['period_s'])
                    initial=visits[0][1] if visits else 0.
                    first=other['weight_scaled']*corrected*initial/300
                    future=other['weight_scaled']*corrected*sum(e*math.exp(-(a-root)/120) for a,e in visits[1:])/300
                    first_cost+=first;future_cost+=future;exposure+=sum(e for _,e in visits)
                    cost_rows.append({'task':k,'delay_s':shift,'full_delay_exposure_s':sum(e for _,e in visits),
                        'risk_signal':sig,'missing':sig is None,'risk_source':src,'topology_correction':.05*min(linked,1.),
                        'first_round_cost':first,'future_risk_cost':future})
                execution=.05*shift/300
                risk_correction=config.future_risk*future_cost
                net=target_benefit-config.collateral*(first_cost+risk_correction)-execution
                veto=bool(config.tour_soon_s and arrivals[j]-t<=config.tour_soon_s and net<=config.threshold)
                result.append({'agent':agent['agent'],'task':j,'dispatch_time_s':float(t),'finish_s':finish,
                    'tour_target_finish_s':arrivals[j],'target_lead_s':lead,'target_signal':signal,'target_source':source,
                    'target_benefit':target_benefit,'first_round_collateral_cost':first_cost,
                    'full_delay_exposure_s':exposure,'future_risk_correction':risk_correction,'execution_cost':execution,
                    'net_score':net,'tour_slack_veto':veto,'unknown_collateral_tasks':unknown,
                    'return_slack_s':slack,'task_costs':cost_rows})
            last[base]=base_finish;costs=edge[str(base)];cursor=(cursor+1)%len(tour);completed+=1
            if completed==len(tour):completed=detours=0
            t=base_finish
    return result


def stopping_decision(candidates,*,now,used,threshold=0.):
    if type(used) is not bool:raise ValueError('Boolean budget required')
    common={'end_s':1800.,'global_extra_budget':1,'after_budget_exhaustion':'protected_tour',
        'public_forecast':'current_readings_held_or_bounded_trend; no future observation assimilation'}
    check=dict(common,budget_after=0,continuation='protected_tour_until_1800')
    wait=dict(common,budget_after=1,continuation='same_stopping_rule_at_later_legal_dispatch')
    if used:return {'action':'TOUR','q_check':None,'q_wait':0.,'selected':None,'CHECK':check,'WAIT':wait}
    admissible=[r for r in candidates if not r.get('tour_slack_veto',False) and r['net_score']>threshold]
    current=[r for r in admissible if r['dispatch_time_s']==now]
    future=[r for r in admissible if r['dispatch_time_s']>now]
    key=lambda r:(r['net_score'],-r['dispatch_time_s'],-r['agent'],-r['task'])
    best=max(current,key=key,default=None);later=max(future,key=key,default=None)
    qcheck=best['net_score'] if best else 0.;qwait=later['net_score'] if later else 0.
    choose=best is not None and qcheck>=qwait-1e-12
    return {'action':'CHECK' if choose else 'WAIT','q_check':qcheck,'q_wait':qwait,
        'selected':best if choose else None,'future_selected':later,'CHECK':check,'WAIT':wait,
        'value_definition':'V(t,b=1)=max(0,max legal CHECK at t,V(next TOUR dispatch,b=1)); CHECK goes to b=0',
        'candidate_count':len(candidates)}


def decision(bundle,used=False,config=StopConfig()):
    rows=[] if used else public_frontier(bundle,config)
    value=stopping_decision(rows,now=bundle['time_s'],used=used,threshold=config.threshold)
    value['config']=asdict(config)
    return value
