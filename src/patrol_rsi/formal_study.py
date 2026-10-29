"""Frozen C1-B3.2 splits, aggregation and staged selection (offline only)."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

from .frozen_selection import select_validation

PROTOCOL_SHA256='5018ee1cd7165cf9e501ebb4218fa9be825d1d76398313305cab08834cb56085'
CONTROLS=('tour','evidence','lookahead','conservative')


def persisted_view(value):
    """Use the same finite JSON representation as original trajectory files."""
    return json.loads(json.dumps(value,allow_nan=False))


def verify_case_index(row,record,fresh):
    """Bind saved timing exactly; compare every deterministic replay metric."""
    if row['metrics']!=record['metrics']:raise ValueError('Index metrics differ from hashed original case')
    times={'decision_wall_s','history_estimator_inference_s'}
    for key,value in record['metrics'].items():
        if key in times:continue
        other=fresh['metrics'][key]
        if isinstance(value,float) and isinstance(other,(int,float)):
            import math
            if not math.isclose(value,other,rel_tol=1e-10,abs_tol=1e-10):raise ValueError('Replay metric differs '+key)
        elif value!=other:raise ValueError('Replay metric differs '+key)


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
        if (selection is None or selection['phase']!='development' or len(selection['retained'])!=2
                or len(set(selection['retained']))!=2 or any(k not in candidate for k in selection['retained'])):
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


def audited_phase_rows(run_root,protocol,phase,previous=None):
    """Read only the phase permitted by its already verified predecessor."""
    rows=[];sources=[];methods=method_definitions(protocol,phase,previous)
    previous_hash=(hashlib.sha256((run_root/'development-selection.json').read_bytes()).hexdigest()
        if phase=='validation' else None)
    for chunk,seed in enumerate(protocol['physical_seeds'][phase]):
        folder=run_root/f'{phase}-{chunk}';index=folder/'case-index.json';audit_path=run_root/f'{phase}-{chunk}-audit.json'
        part=json.loads(index.read_bytes());audit=json.loads(audit_path.read_bytes())
        frozen=json.loads((folder/'phase-frozen.json').read_bytes());digest=hashlib.sha256(index.read_bytes()).hexdigest()
        expected=[(k,s) for k in methods for s in conditions(protocol,phase,chunk)]
        if (audit['case_index_sha256']!=digest or audit['actual_cases_replayed']!=len(part) or audit['exact_violations'] or
                [(r['alias'],r['specification']) for r in part]!=expected):
            raise ValueError('Incomplete audited phase index')
        if (frozen['phase']!=phase or frozen['chunk']!=chunk or frozen['protocol_sha256']!=PROTOCOL_SHA256 or
                frozen['conditions']!=conditions(protocol,phase,chunk) or frozen['methods']!=methods or
                frozen['selection_sha256']!=previous_hash or frozen['predecessor_selection']!=previous):
            raise ValueError('Phase source/selection freeze differs')
        if any(r['definition']!=methods[r['alias']] for r in part):raise ValueError('Case method/config differs')
        rows.extend(part);sources.append({'chunk':chunk,'seed':seed,'case_index_sha256':digest,
            'audit_sha256':hashlib.sha256(audit_path.read_bytes()).hexdigest(),'cases':len(part)})
    return rows,sources


def verified_selection(path,protocol,run_root,expected_phase):
    if expected_phase not in ('development','validation'):raise ValueError('Only completed selection stages')
    if path.resolve()!=(run_root/(expected_phase+'-selection.json')).resolve():raise ValueError('Wrong selection location')
    value=json.loads(path.read_bytes())
    if value.get('phase')!=expected_phase or value.get('protocol_sha256')!=PROTOCOL_SHA256 or value.get('final_test_opened') is not False:
        raise ValueError('Missing frozen protocol/phase receipt')
    if not isinstance(value.get('frozen_unix'),(int,float)):raise ValueError('Selection freeze time missing')
    previous=(verified_selection(run_root/'development-selection.json',protocol,run_root,'development')
        if expected_phase=='validation' else None)
    expected_hash=hashlib.sha256((run_root/'development-selection.json').read_bytes()).hexdigest() if previous else None
    if value.get('previous_sha256')!=expected_hash:raise ValueError('Predecessor hash chain differs')
    rows,sources=audited_phase_rows(run_root,protocol,expected_phase,previous)
    result=development_selection(rows) if expected_phase=='development' else validation_selection(rows,previous['retained'])
    if any(value.get(k)!=v for k,v in result.items()) or value.get('audited_sources')!=sources:
        raise ValueError('Selection does not reproduce from audited sources')
    return value
