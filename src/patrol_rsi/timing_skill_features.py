"""Causal, timestamped features from existing public channels; no trace/labels."""
from __future__ import annotations
import math
import numpy as np

from .causal_observation import validate_public
from .patrol_timing import PUBLIC_FIELDS


def from_legacy_row(row,*,clock=None):
    """Partial public projection for exact E1 predicate equivalence checks."""
    if set(row)!=PUBLIC_FIELDS:raise ValueError('Exact archived E1 public row required')
    available=bool(row['available']);direct=row['measurement_age_s'] is not None
    return {'kind':row['kind'],'professional_available':available,'professional_signal':float(row['signal']) if available else None,
        'already_alarm':bool(row['already_alarm']),'direct_available':direct,'direct_age_s':row['measurement_age_s'],
        'estimate_available':row['estimate_age_s'] is not None,'estimate_age_s':row['estimate_age_s'],
        'importance':float(row['weight_scaled']),'inspection_age_ratio':float(row['age_ratio']),
        'lead_s':float(row['lead_s']),'shift_s':float(row['shift_s']),
        'completion_cost_s':row['candidate_finish_s']-clock if clock is not None else None,'quota_available':True}


class PublicTimingView:
    """Validated public bundle shared by all candidates at one decision root."""
    def __init__(self,bundle,*,legacy_refresh_s=None):
        self.bundle=validate_public(bundle);self.now=float(bundle['time_s']);self.tasks=bundle['tasks'];self.calls=bundle['estimator']['calls']
        if legacy_refresh_s not in (None,30):raise ValueError('Only explicitly verified legacy30second reconstruction')
        self.legacy_refresh_s=legacy_refresh_s
        self.thermal_by_pod={}
        for row in self.tasks:
            if row['kind']=='thermal':self.thermal_by_pod.setdefault(row['pod'],[]).append(row)
        self.racks=sum(len(v) for v in self.thermal_by_pod.values())
        for field in ('public_requested_it_kw','public_delivered_it_kw'):
            if len(bundle['telemetry'][field])!=self.racks:raise ValueError('Existing per-rack public power channel required')

    def features(self,geometry):
        j=int(geometry['task']);row=self.tasks[j];kind=row['kind'];pod=row['pod'];now=self.now
        if kind=='thermal':
            available=row['temperature_stamp_s']>=0 or self.calls>0;signal=max(row['temperature_c']-65,0)/20
            direct=row['temperature_measurement']
            q=row['qualified'];qualified=direct['available'] and q['available'] and direct['time_s']==q['time_s']
        elif kind=='electrical':available=row['qualified_available'];signal=max(row['contact_c']-60,0)/30;direct=row['contact'];qualified=direct['available']
        elif kind=='cooling':
            available=row['flow']['available'];signal=max(1-row['flow_kg_s']/33.6,0);direct=row['flow'];q=row['qualified']
            qualified=direct['available'] and q['available'] and direct['time_s']==q['time_s']
        else:available=row['qualified_available'];signal=float(row['sensor_verified']);direct=row['sensor_error'];qualified=direct['available']
        direct_available=bool(direct['available']);normal=None
        if direct_available:
            v=direct['value'];normal=v<=65 if kind=='thermal' else v<=60 if kind=='electrical' else v>=33.6 if kind=='cooling' else abs(v)<=1
        age=(now-max(0.,row['last_check_s']))/row['deadline_s']
        temperature_rows=[row] if kind=='thermal' else self.thermal_by_pod[pod]
        known=[r for r in temperature_rows if r['temperature_stamp_s']>=0 or self.calls>0]
        disagreement=max(abs(r['temperature_c']-r['nominal_temperature_c'])/20 for r in known) if known else None
        pure=[r for r in temperature_rows if r['cvae_raw']['available']]
        prior_disagreement=max(abs(r['cvae_raw']['value']-r['nominal_temperature_c'])/20 for r in pure) if pure else None
        estimate_age=now-max(r['cvae_raw']['time_s'] for r in pure) if pure else now-math.floor(now/30)*30 if self.calls>0 and self.legacy_refresh_s==30 else None
        pod_known=[r for r in self.thermal_by_pod[pod] if r['temperature_stamp_s']>=0 or self.calls>0]
        pod_signal=max(max(r['temperature_c']-65,0)/20 for r in pod_known) if pod_known else None
        flow=row['flow'];flow_signal=max(1-flow['value']/33.6,0) if flow['available'] else None
        telemetry=self.bundle['telemetry'];start=pod*8
        requested=sum(telemetry['public_requested_it_kw'][start:start+8]);delivered=sum(telemetry['public_delivered_it_kw'][start:start+8])
        gap=max(requested-delivered,0)/max(requested,1.)
        return {'kind':kind,'professional_available':bool(available),'professional_signal':float(signal) if available else None,
            'already_alarm':bool(row['already_alarm']),'ever_checked':row['last_check_s']>=0,'quota_available':True,
            'direct_available':direct_available,'direct_reading_normal':bool(normal) if normal is not None else None,
            'direct_is_qualified':bool(qualified) if direct_available else None,'direct_age_s':float(direct['age_s']) if direct_available else None,
            'direct_age_ratio':min(float(direct['age_s'])/row['deadline_s'],6.) if direct_available else None,
            'estimate_available':self.calls>0,'estimate_age_s':estimate_age,
            'thermal_disagreement':float(disagreement) if disagreement is not None else None,
            'prior_disagreement':float(prior_disagreement) if prior_disagreement is not None else None,
            'pod_thermal_signal':float(pod_signal) if pod_signal is not None else None,'pod_flow_deficit':float(flow_signal) if flow_signal is not None else None,
            'pod_service_gap':float(gap),'inspection_age_ratio':float(np.float32(min(age,6))),
            'importance':float(np.float32(row['weight_scaled'])),'lead_s':float(geometry['target_lead_s']),
            'shift_s':float(geometry['shift_s']),'completion_cost_s':float(geometry['finish_s']-now)}


FEATURE_MEANINGS={
    'professional_signal':'Existing own-profession cue: thermal fused excess/qualified contact/observed flow/qualified verified bias; None when unavailable, not a probability.',
    'direct_reading_normal':'Available direct channel is on its reference-normal side. Not hidden truth or a guarantee of normality; remote versus qualified explicitly separated.',
    'direct_is_qualified':'Most recent direct reading is a qualified reference, not merely a remote/model value.',
    'direct_age_s':'Age of actual direct reading; None means missing. Thermal may be remote or qualified; other roles retain their declared sources.',
    'estimate_age_s':'Timestamped pure temperature-CVAE invocation age; None when timestamp unavailable. Legacy30s reconstruction only when explicitly requested for verified E1 tapes.',
    'thermal_disagreement':'Thermal tool difference/20: target rack for thermal, maximum of existing observed/estimated thermal racks in the same public pod otherwise. Not cross-profession uncertainty probability.',
    'prior_disagreement':'Pure existing temperature-CVAE prior mean versus causal nominal model/20; missing on old tapes without raw prior recorder. Not a fault probability.',
    'pod_thermal_signal':'Maximum available fused temperature excess in public same-pod thermal tasks; a cross-system cue, not confirmation.',
    'pod_flow_deficit':'Existing observed same-pod flow deficit; None when not observed.',
    'pod_service_gap':'Existing per-rack public requested/delivered IT power aggregated over known8-rack pod. Service-gap proxy, not voltage/current or contact-hotspot confirmation.',
    'inspection_age_ratio':'Public qualified-check age/deadline, including startup age; distinct from direct remote measurement age.',
    'lead_s':'Earlier completion than continuing protected Tour, from public deterministic route/service model.',
    'shift_s':'Additional public own-tour schedule shift from optional check, not empirical risk or a safety rule.'}
