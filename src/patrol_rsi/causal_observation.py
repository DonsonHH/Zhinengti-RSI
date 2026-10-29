"""Causal public adapter: provenance, missingness and schedule, never truth."""
from __future__ import annotations

import math
from .inspection_timing import public_packet
from .improver_task import ImproverTask


def reading(value=None, when=None, *, source, unit, now, cutoff=None):
    available=bool(value is not None and when is not None and when>=0)
    return {'available':available,'value':float(value) if available else None,
        'time_s':float(when) if available else None,'age_s':float(now-when) if available else None,
        'context_cutoff_s':float(cutoff if cutoff is not None else when) if available else None,
        'source':source,'unit':unit,'reason':None if available else 'not_observed_or_not_produced'}


class ObservedTask(ImproverTask):
    """Same task equations/actions, with pure prior output timestamps recorded."""
    def __init__(self,spec,program,predictor=None):
        self.prior_record=None
        def tracked(context):
            values=predictor(context)
            self.prior_record={'time_s':float(self.clock),'context_cutoff_s':float(self.clock),
                'values':[float(v) for v in values]}
            return values
        super().__init__(spec,program,tracked if predictor is not None else None)


CHANNELS=('qualified','temperature_measurement','flow','contact','sensor_error',
    'nominal','cvae_raw','history_assimilated')


def capture_public(env):
    """The sole environment adapter. Only public observation/schedule reads.

    CVAE current_history is explicitly assimilated, not called a pure decoder
    sample. The optional pure-output recorder is installed in ObservedTask.
    """
    packet=public_packet(env);now=float(env.clock);obs=env.state.observer
    recent={}
    for event in env.events:
        recent[event['task']]=event
    raw=getattr(env,'prior_record',None)
    tasks=[]
    for j,(old,task) in enumerate(zip(packet['tasks'],env.tasks)):
        pod=task.index//8 if task.kind=='thermal' else task.index
        rack=task.index if task.kind=='thermal' else pod*8
        event=recent.get(j)
        key={'thermal':'temperature_c','cooling':'pod_flow_kg_s',
            'electrical':'contact_temperature_c','control':'controller_sensor_error_c'}[task.kind]
        unit='kg/s' if task.kind=='cooling' else 'degC'
        empty=lambda source,u='degC':reading(source=source,unit=u,now=now)
        channels={k:empty(k,'kg/s' if k=='flow' else 'degC') for k in CHANNELS}
        channels['qualified']=reading(event['measurement'][key] if event else None,event['time_s'] if event else None,
            source='qualified_reference',unit=unit,now=now)
        channels['temperature_measurement']=reading(obs.last_temperature[rack],obs.temperature_stamp[rack],
            source='remote_or_qualified_temperature',unit='degC',now=now)
        flow_stamp=float(obs.flow_stamp[pod])
        remote_pods=set(int(v) for v in env.current_packet['flow_pod_indices'])
        channels['flow']=reading(obs.last_flow[pod],flow_stamp,
            source='remote_flow' if pod in remote_pods and flow_stamp==now else 'qualified_flow',unit='kg/s',now=now)
        electric=next((v for t,v in recent.items() if env.tasks[t].kind=='electrical' and env.tasks[t].index==pod),None)
        control=next((v for t,v in recent.items() if env.tasks[t].kind=='control' and env.tasks[t].index==pod),None)
        if electric:
            channels['contact']=reading(electric['measurement']['contact_temperature_c'],electric['time_s'],
                source='qualified_contact',unit='degC',now=now)
        if control:
            channels['sensor_error']=reading(control['measurement']['controller_sensor_error_c'],control['time_s'],
                source='qualified_sensor_error',unit='degC',now=now)
        channels['nominal']=reading(old['nominal_temperature_c'],now,source='causal_nominal_model',unit='degC',now=now)
        if env.mode=='history' and env.estimator_calls:
            predict_time=math.floor(now/env.workflow.refresh_s)*env.workflow.refresh_s
            assimilated=max(predict_time,float(obs.temperature_stamp[rack]))
            channels['history_assimilated']=reading(old['temperature_c'],assimilated,
                source='prior_mean_plus_completed_measurements',unit='degC',now=now)
        if raw:
            channels['cvae_raw']=reading(raw['values'][rack],raw['time_s'],source='frozen_cvae_prior_mean',
                unit='degC',now=now,cutoff=raw['context_cutoff_s'])
        tasks.append(dict(old,**channels,task=j,pod=pod,required_role=task.required_role,
            access=task.access,qualified_available=event is not None))
    telemetry={k:[float(v) for v in env.current_packet[k]] for k in (
        'public_requested_it_kw','public_delivered_it_kw','pod_pump_command','secondary_supply_c')}
    telemetry['time_s']=float(env.current_packet['time_s'])
    result={'schema':'causal-public-v1','time_s':now,'tasks':tasks,'agents':packet['agents'],
        'telemetry':telemetry,'topology':{'relation':'same_public_pod_power_and_cdu_branch',
            'task_pods':[r['pod'] for r in tasks]},'estimator':{'mode':env.mode,
            'calls':int(env.estimator_calls),'pure_prior_recorded':raw is not None}}
    return validate_public(result)


def validate_public(bundle):
    if set(bundle)!={'schema','time_s','tasks','agents','telemetry','topology','estimator'} or bundle['schema']!='causal-public-v1':
        raise ValueError('Only the explicit causal public schema is accepted')
    now=bundle['time_s']
    if not math.isfinite(now) or now%5 or not 5<=now<=1800:
        raise ValueError('Invalid observation time')
    if bundle['telemetry']['time_s']>now:
        raise ValueError('Future telemetry')
    if set(bundle['telemetry'])!={'time_s','public_requested_it_kw','public_delivered_it_kw','pod_pump_command','secondary_supply_c'}:
        raise ValueError('Unexpected telemetry field')
    if set(bundle['topology'])!={'relation','task_pods'} or set(bundle['estimator'])!={'mode','calls','pure_prior_recorded'}:
        raise ValueError('Unexpected topology/estimator metadata')
    for agent in bundle['agents']:
        if set(agent)!={'agent','tour','cursor','busy_target','busy_finish_s','detours_used','baseline_completed',
                'first_costs_s','costs_s','period_s'}:
            raise ValueError('Unexpected schedule field')
    fields=set(CHANNELS)|{'task','pod','required_role','access','qualified_available',
        'kind','service_s','deadline_s','weight_scaled','last_check_s','already_alarm',
        'temperature_c','last_temperature_c','temperature_age_s','temperature_stamp_s','velocity_c_s',
        'nominal_temperature_c','contact_c','flow_kg_s','sensor_verified'}
    for j,row in enumerate(bundle['tasks']):
        if set(row)!=fields or row['task']!=j or row['last_check_s']>now:
            raise ValueError('Unexpected task/evaluator field or future check')
        for k in CHANNELS:
            r=row[k]
            if set(r)!={'available','value','time_s','age_s','context_cutoff_s','source','unit','reason'}:
                raise ValueError('Unexpected measurement field')
            if r['available']:
                if not all(isinstance(r[v],(int,float)) and math.isfinite(r[v]) for v in ('value','time_s','context_cutoff_s','age_s')):
                    raise ValueError('Available reading must be finite')
                if not 0<=r['context_cutoff_s']<=r['time_s']<=now or abs(r['age_s']-(now-r['time_s']))>1e-8:
                    raise ValueError('Future or inconsistent channel timestamp')
            elif any(r[v] is not None for v in ('value','time_s','context_cutoff_s','age_s')):
                raise ValueError('Unavailable is not zero risk')
    return bundle


def task_signal(row,delta_s=0.):
    """Return (signal or None, source); missing data never silently becomes 0."""
    if row['kind']=='cooling':
        r=row['flow'];return (max(1-r['value']/33.6,0.),r['source']) if r['available'] else (None,'flow_unavailable')
    if row['kind']=='electrical':
        r=row['contact'];return (max(r['value']-60,0.)/30,r['source']) if r['available'] else (None,'contact_unavailable')
    if row['kind']=='control':
        r=row['sensor_error'];return (float(abs(r['value'])>1),r['source']) if r['available'] else (None,'sensor_check_unavailable')
    values=[(row[k]['value'],k) for k in ('history_assimilated','nominal') if row[k]['available']]
    r=row['temperature_measurement']
    if r['available']:
        linear=max(35.,min(110.,r['value']+row['velocity_c_s']*min(120.,r['age_s']+delta_s)))
        values.append((linear,'measured_bounded_trend'))
    if not values:return None,'temperature_unavailable'
    value,source=max(values)
    return max(value-65.,0.)/20,source
