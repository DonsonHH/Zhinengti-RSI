"""Explicit preserve/use decision over unchanged legal checks and protected Tour."""
from __future__ import annotations
import numpy as np

from .improver_task import choose_program,typed_signal

METHODS=('Tour','human_professional','unconditional_PPO','evidence_gate_PPO','evidence_gate_urgency')
SIGNALS={'thermal':('thermal_excess',.05),'electrical':('contact_heat',.05),
    'cooling':('flow_deficit',.05),'control':('verified_sensor_bias',.5)}
PUBLIC_FIELDS={'agent','task','baseline_next','kind','signal_name','signal','threshold','available','measurement_age_s',
    'signal_source','observed_value','observed_unit','estimate_age_s','weight_scaled','age_ratio','eta_ratio','candidate_finish_s','tour_target_finish_s','lead_s','shift_s','already_alarm'}


def gate_candidate(row):
    if set(row)!=PUBLIC_FIELDS:raise ValueError('Gate accepts only closed public candidate fields')
    return bool(row['task']!=row['baseline_next'] and row['available'] and np.isfinite(row['signal']) and
        row['signal']>=row['threshold'] and not row['already_alarm'] and row['lead_s']>0 and
        row['signal']*row['weight_scaled']*row['lead_s']>.05*row['shift_s'])


def count_later_roots(times,cutoff):
    """Distinct decision roots; plain int survives JSON even for NumPy inputs."""
    return len({float(t) for t in times if t>cutoff})


def evidence_fields(env,i,ids,features):
    now=float(env.clock);plan=env.plans[i];cursor=int(env.protected_cursor[i]);base=int(plan.tour[cursor])
    schedule=plan.check(now,env.agents[i]['position'],env.last_inspection,cursor)
    if not schedule['ok']:raise ValueError('Protected baseline must remain feasible')
    arrivals=dict(zip(schedule['fallback_order'],schedule['fallback_completion_s']));obs=env.state.observer;rows=[]
    recent={v['task']:v['time_s'] for v in env.events}
    for k,raw in enumerate(ids):
        j=int(raw);task=env.tasks[j];name,threshold=SIGNALS[task.kind]
        signal,_=typed_signal(env,ids[k:k+1],features[k:k+1],name);value=float(signal[0])
        estimate_age=None
        if task.kind=='thermal':
            stamp=float(obs.temperature_stamp[task.index]);available=stamp>=0 or env.estimator_calls>0;observed=float(env.current_history[task.index])
            source='history_assimilated_temperature';unit='degC'
            if env.estimator_calls:estimate_age=now-(now//env.workflow.refresh_s)*env.workflow.refresh_s
        elif task.kind=='cooling':stamp=float(obs.flow_stamp[task.index]);available=stamp>=0;observed=float(obs.last_flow[task.index]);source='observed_flow';unit='kg/s'
        elif task.kind=='electrical':stamp=float(recent.get(j,-1));available=stamp>=0;observed=float(obs.last_contact[task.index]);source='qualified_contact_temperature';unit='degC'
        else:
            stamp=float(recent.get(j,-1));available=stamp>=0;source='qualified_verified_sensor_bias';unit='degC'
            last=next((v for v in reversed(env.events) if v['task']==j),None);observed=float(last['measurement']['controller_sensor_error_c']) if last else 0.
        finish=now+plan.travel_service(env.agents[i]['position'],j)
        shift=0. if j==base else finish+plan.cost[j,base]-(now+plan.travel_service(env.agents[i]['position'],base))
        rows.append({'agent':int(i),'task':j,'baseline_next':base,'kind':task.kind,'signal_name':name,
            'signal':value if np.isfinite(value) else 0.,'threshold':threshold,'available':bool(available and np.isfinite(value)),
            'signal_source':source,'observed_value':observed if available and np.isfinite(observed) else None,'observed_unit':unit,'estimate_age_s':estimate_age,
            'measurement_age_s':now-stamp if stamp>=0 else None,'weight_scaled':float(features[k,6]),
            'age_ratio':float(features[k,0]),'eta_ratio':float(features[k,1]),'candidate_finish_s':float(finish),
            'tour_target_finish_s':float(arrivals[j]),'lead_s':max(0.,min(float(arrivals[j]),1800.)-finish),
            'shift_s':float(shift),'already_alarm':bool(env.measured_alert.get(j,False))})
    return rows


def gated_choices(groups,method,used):
    if method not in ('evidence_gate_PPO','evidence_gate_urgency'):raise ValueError('Fixed gated ranking method')
    choices={};logs=[]
    for group in groups:
        i=group['agent'];ids=group['ids'];base=group['baseline_next']
        if not len(ids):choices[i]=None;continue
        eligible=[k for k,row in enumerate(group['public']) if gate_candidate(row)] if not used else []
        if eligible:
            scores=group['scores'] if method=='evidence_gate_PPO' else group['features'][:,0]+group['features'][:,1]
            k=max(eligible,key=lambda k:(float(scores[k]),-k));selected=int(ids[k]);tool='ppo' if method=='evidence_gate_PPO' else 'urgency'
            ranked=len(eligible)
        else:selected=base;tool='return_tour';ranked=0
        choices[i]=selected
        logs.append({'time_s':group['clock'],'agent':i,'task':selected,'baseline_next':base,
            'actor_task':int(ids[int(np.argmax(group['scores']))]),'tool':tool,'rule_index':0 if eligible else None,
            'legal_candidates':ranked if tool=='ppo' else len(ids),'hard_legal_candidates':len(ids),'ranking_candidates':ranked,
            'detours_used_before':group['detours_used'],'baseline_completed_before':group['baseline_completed'],
            'quota_decision':'USE_CHECK' if eligible else 'PRESERVE_CONTINUE_TOUR' if not used else 'BUDGET_SPENT_TOUR'})
    return choices,logs


class TimingSelector:
    """Adapter sees public state only; offline oracle never calls into this object."""
    def __init__(self,method,*,capture=None,forced=None):
        if method not in METHODS:raise ValueError('Five frozen timing methods')
        self.method,self.capture,self.forced=method,capture,forced
    def __call__(self,env,scorer):
        groups=[];cache={};reserved=env.reserved();used=any(v['task']!=v['baseline_next'] for v in env.selected_certificates)
        def scored(f):
            key=(f.shape,f.tobytes())
            if key not in cache:cache[key]=np.asarray(scorer(f))
            return cache[key]
        for raw in np.roll(np.arange(len(env.agents)),-(env.tick%len(env.agents))):
            i=int(raw)
            if env.agents[i]['task'] is not None:continue
            ids,f,_=env.options(i,reserved);scores=scored(f) if len(ids) else np.zeros(0)
            if scores.shape!=(len(ids),) or not np.isfinite(scores).all():raise ValueError('Finite frozen PPO scores')
            base=int(env.plans[i].tour[env.protected_cursor[i]])
            groups.append({'agent':i,'ids':ids,'features':f,'scores':scores,'public':evidence_fields(env,i,ids,f) if len(ids) else [],
                'baseline_next':base,'clock':float(env.clock),'detours_used':int(env.detours_used[i]),'baseline_completed':int(env.baseline_completed[i])})
        if self.method.startswith('evidence_gate'):choices,logs=gated_choices(groups,self.method,used)
        else:choices,logs=choose_program(env,scored)
        if self.forced is not None and env.clock==self.forced['dispatch_time_s']:
            if used:raise ValueError('Forced diagnostic cannot spend already consumed quota')
            i=self.forced['agent'];j=self.forced['task'];g=next(v for v in groups if v['agent']==i)
            if j not in g['ids'] or j==g['baseline_next']:raise ValueError('Offline witness must be a currently legal extra check')
            choices[i]=j
            row=next(v for v in logs if v['agent']==i);row.update(task=j,tool='offline_witness',rule_index=None)
        hypothetical={}
        if self.capture is not None:
            for method in ('evidence_gate_PPO','evidence_gate_urgency'):
                proposal,receipt=gated_choices(groups,method,False)
                first=next((v for v in receipt if v['task']!=v['baseline_next']),None)
                hypothetical[method]=dict(first) if first else None
            self.capture(env,groups,hypothetical)
        first_extra=next((v for v in logs if v['task']!=v['baseline_next']),None)
        legal_extra=sum(v['task']!=v['baseline_next'] for g in groups for v in g['public'])
        selected_public=next((v for g in groups for v in g['public'] if first_extra and v['agent']==first_extra['agent'] and v['task']==first_extra['task']),None)
        frame={'time_s':float(env.clock),'quota_used_before':used,'legal_extra_candidates':legal_extra,
            'gate_eligible_candidates':sum(gate_candidate(v) for g in groups for v in g['public']),
            'decision':'CHECK' if first_extra and not used else 'TOUR_BUDGET_SPENT' if used else 'WAIT_CONTINUE_TOUR' if legal_extra else 'TOUR_NO_OPTION',
            'selected_extra':{k:first_extra[k] for k in ('agent','task','baseline_next','tool')} if first_extra and not used else None,
            'selected_extra_public_evidence':selected_public if first_extra and not used else None,
            'actor_scoring_calls':len(cache)}
        if not hasattr(env,'e1_decision_frames'):env.e1_decision_frames=[];env.e1_first_scores=[]
        env.e1_decision_frames.append(frame)
        if env.tick==0:
            env.e1_first_scores=[{'agent':g['agent'],'baseline_next':g['baseline_next'],'ids':g['ids'].tolist(),'scores':g['scores'].tolist(),
                'features':g['features'].tolist(),'public':g['public']} for g in groups]
        return choices,logs
