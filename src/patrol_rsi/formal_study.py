"""Frozen C1-B3.2 splits, aggregation and staged selection (offline only)."""
from __future__ import annotations

import hashlib
import itertools
import json

from .frozen_selection import select_validation

PROTOCOL_SHA256='5018ee1cd7165cf9e501ebb4218fa9be825d1d76398313305cab08834cb56085'
CONTROLS=('tour','evidence','lookahead','conservative')


def load_protocol(path):
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PROTOCOL_SHA256:
        raise ValueError('Original frozen protocol SHA differs')
    return json.loads(raw)


def conditions(protocol,phase,chunk):
    if phase not in ('development','validation','test') or type(chunk) is not int:
        raise ValueError('Frozen phase/chunk required')
    seeds=protocol['physical_seeds'][phase]
    if chunk not in range(len(seeds)):raise ValueError('Frozen physical seed index')
    return [dict(phase=phase,seed=seeds[chunk],scenario=scenario,duration_s=duration,
        fault_start_s=start,noise_c=noise,estimator='history',control='feedback',resources='provisioned')
        for (scenario,duration),start,noise in itertools.product(protocol['fixtures'],protocol['fault_start_s'],protocol['noise_c'])]


def aggregate(rows):
    weight=sum(r['event_weight_sum'] for r in rows)
    return {'cases':len(rows),'segments':sum(r['metrics']['anomaly_segments'] for r in rows),
        'missed':sum(r['metrics']['missed_segments'] for r in rows),
        'weighted_delay_sum':sum(r['weighted_delay_sum'] for r in rows),'event_weight_sum':weight,
        'weighted_wait_s':sum(r['weighted_delay_sum'] for r in rows)/weight if weight else None,
        'travel_m':sum(r['metrics']['travel_m'] for r in rows),
        'service_s':sum(r['metrics']['service_s'] for r in rows),
        'checks':sum(r['metrics']['selected_detours'] for r in rows),
        'violations':sum(r['metrics']['tasks_violating_exact_gap'] for r in rows),
        'wall_s':sum(r['wall_s'] for r in rows)}


def method_definitions(protocol,phase,selection=None):
    candidate={r['id']:r['config'] for r in protocol['revised_candidates']}
    methods={k:{'method':k,'config':None} for k in CONTROLS}
    if phase=='development':
        methods.update({k:{'method':'revised','config':v} for k,v in candidate.items()})
    elif phase=='validation':
        if selection is None or selection['phase']!='development' or len(selection['retained'])!=2:
            raise ValueError('Validation requires the frozen two-candidate development receipt')
        methods.update({k:{'method':'revised','config':candidate[k]} for k in selection['retained']})
    elif phase=='test':
        if selection is None or selection['phase']!='validation' or selection['selected'] not in (*candidate,'TOUR'):
            raise ValueError('Final test requires frozen validation selection')
        chosen=selection['selected']
        methods['revised']={'method':'tour' if chosen=='TOUR' else 'revised','config':None if chosen=='TOUR' else candidate[chosen]}
    else:raise ValueError('Unknown formal phase')
    return methods


def rank_key(identity,summary):
    return (summary['missed'],summary['weighted_wait_s'] if summary['weighted_wait_s'] is not None else 0.,summary['travel_m'],identity)


def development_selection(rows):
    ids=('C0','C1','C2','C3')
    groups={k:aggregate([r for r in rows if r['alias']==k]) for k in ids}
    if any(v['cases']!=32 or v['violations'] for v in groups.values()):raise ValueError('Incomplete or invalid development')
    order=sorted(ids,key=lambda k:rank_key(k,groups[k]))
    return {'phase':'development','ranked':order,'retained':order[:2],'groups':groups,
        'rule':'total misses, pooled consequence-weighted censored delay, total travel, candidate id'}


def validation_selection(rows,retained):
    seeds=('8600','8601')
    groups={k:aggregate([r for r in rows if r['alias']==k]) for k in ('tour',*retained)}
    byseed={k:{s:aggregate([r for r in rows if r['alias']==k and str(r['specification']['seed'])==s]) for s in seeds}
        for k in groups}
    if len(retained)!=2 or any(v['cases']!=32 or v['violations'] for v in groups.values()):
        raise ValueError('Incomplete validation inputs')
    if any(abs(v['event_weight_sum']-groups['tour']['event_weight_sum'])>1e-8 for v in groups.values()):
        raise ValueError('Paired anomaly-weight denominators differ')
    selected=select_validation({k:byseed[k] for k in retained},byseed['tour'])
    rationale={}
    for k in retained:
        worst=max(byseed[k][s]['missed']-byseed['tour'][s]['missed'] for s in seeds)
        delay=groups[k]['weighted_delay_sum']-groups['tour']['weighted_delay_sum']
        strict=groups[k]['missed']<groups['tour']['missed'] or delay < -1e-10
        rationale[k]={'seed_missed_delta':{s:byseed[k][s]['missed']-byseed['tour'][s]['missed'] for s in seeds},
            'pooled_wait_delta_s':groups[k]['weighted_wait_s']-groups['tour']['weighted_wait_s'],
            'at_least_one_strict_improvement':strict,'eligible':worst<=0 and delay<=1e-10 and strict}
    return {'phase':'validation','retained_from_development':list(retained),'selected':selected,
        'fallback':selected=='TOUR','groups':groups,'by_seed':byseed,'rationale':rationale,
        'rule':'zero extra misses per seed; no pooled delay increase; at least one strict improvement; frozen ranking'}
