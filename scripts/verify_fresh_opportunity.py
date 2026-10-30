"""Independent original-measurement, history and exact-ETA verification.

No producer evaluator_labels/sample_targets/completion_lookup functions are used.
All359 public packets are checked; all85+13 columns at59 roots are compared.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import lzma
import math
from pathlib import Path

import numpy as np

from patrol_rsi.active_inspection import graph_for
from patrol_rsi.aisles import build_agents
from patrol_rsi.causal_observation import CHANNELS, validate_public
from patrol_rsi.fresh_opportunity import (ROOTS, KIND_IDS, frozen_inputs, need, read, sha, write)
from patrol_rsi.m9 import arrays_digest
from patrol_rsi.opportunity_data import FEATURE_NAMES, ACTION_FEATURE_NAMES, HORIZONS, SOURCE_CODES
from verify_opportunity_data import direct_labels
from fresh_schedule_audit import GeometryAudit, verify_record


def read_gz(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def verify_condition(folder, group):
    identity = read(folder / 'source-identity-not-features.json')
    spec = identity['specification']
    need(sha(folder / 'tour-record.json.gz') == identity['source_Tour_sha256'], 'Original Tour SHA')
    history_path = folder / identity.get('source_public_history_path', 'public-states.jsonl.gz')
    need(sha(history_path) == identity['source_public_history_sha256'], 'Public history SHA')
    need(sha(group / identity['source_process_path']) == identity['source_process_sha256'], 'Physical source SHA')
    need(sha(group / 'geometry.npz') == identity['source_geometry_sha256'], 'Geometry SHA')
    for name, receipt in read(folder / 'files.json').items():
        need(sha(folder / name) == receipt['sha256'] and (folder / name).stat().st_size == receipt['bytes'], 'Condition bytes')
    with np.load(group / identity['source_process_path']) as data:
        arrays = {k: data[k] for k in data.files}
    need(arrays_digest(arrays) == identity['numeric_process_fingerprint'], 'Physical numerical fingerprint')
    record = read_gz(folder / 'tour-record.json.gz')
    need(record['specification'] == spec and record['trace_fingerprint'] == identity['numeric_process_fingerprint'], 'Tour spec and process binding')
    measured_checks=verify_record(record,arrays,spec['seed'])
    geometry_audit=GeometryAudit(record)
    # Original measurement rules are independent of the producer's references.
    a, ref = direct_labels(arrays, spec['seed'])
    with np.load(folder / 'labels-only.npz') as values:
        labels = {k: values[k] for k in values.files}
    need(np.array_equal(a, labels['A_physical']) and np.array_equal(ref, labels['B_reference']), 'Independent360x176 physical/qualified labels')
    events = read(folder / 'events-evaluator-only.json')
    expected_ids = np.zeros_like(a, dtype=np.int32); expected_retained = expected_ids.copy()
    event_rows = []
    for j in range(176):
        edges = np.diff(np.r_[False, a[:, j], False].astype(int))
        for first, last in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            eid = len(event_rows) + 1
            event_rows.append({'id': eid, 'task': j, 'kind': ('thermal', 'electrical', 'cooling', 'control')[KIND_IDS[j]],
                'first_tick': int(first), 'last_exclusive_tick': int(last), 'onset_s': int(first * 5),
                'end_s': int(last * 5), 'retained_at_least_15s': bool(last-first >= 3),
                'confirmable_positive_bins': int(ref[first:last, j].sum())})
            expected_ids[first:last, j] = eid
            if last-first >= 3:
                expected_retained[first:last, j] = eid
    need(event_rows == events and np.array_equal(expected_ids, labels['event_id']) and
         np.array_equal(expected_retained, labels['retained_event_id']), 'Every maximal raw/retained event exactly recomputed')
    with np.load(folder / 'public-features.npz') as values:
        base, action = values['base'], values['completion']
    with np.load(folder / 'targets-evaluator-only.npz') as values:
        targets = {k: values[k] for k in values.files}
    need(base.shape == (59, 176, 85) and action.shape == (59, 3, 176, 13), 'Frozen59-root shapes')
    need(all(v.shape == (59, 3, 176) for v in targets.values()), 'Targets exactly match feature samples')
    need(np.isfinite(base).all() and np.isfinite(action).all(), 'Finite public features')
    with np.load(group / 'geometry.npz') as data:
        geom = {k: data[k] for k in data.files}
    # Independent public route arithmetic, ordered without observing alarms.
    planned = []
    for v in record['assignments']:
        finish = v['time_s'] + math.ceil((v['distance_m'] + v['service_s']) / 5 - 1e-10) * 5
        if finish <= 1800:
            planned.append((finish, v['task'], v['time_s'], v['agent'], False, -1, finish, 0.))
    for i in range(len(geom['task'])):
        planned.append((float(geom['finish_s'][i]), int(geom['task'][i]), float(geom['dispatch_time_s'][i]),
            int(geom['agent'][i]), True, i, float(geom['tour_target_finish_s'][i]), float(geom['return_slack_s'][i])))
    by_finish = {}
    for item in sorted(planned, key=lambda v: (v[0], v[1], v[2], v[3], v[4])):
        by_finish.setdefault(item[0], []).append(item)
    fcol = {n: i for i, n in enumerate(FEATURE_NAMES)}
    hcol = {n: i for i, n in enumerate(ACTION_FEATURE_NAMES)}
    roles = {v['id']: v['role'] for v in build_agents(graph_for(128, 'open'), 'provisioned')}
    history = [[] for _ in range(176)]
    cursor = frames = kept = available_readings = 0
    history_opener = lzma.open if history_path.suffix == '.xz' else gzip.open
    with history_opener(history_path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            bundle = json.loads(line); validate_public(bundle)
            geometry_audit.append(bundle)
            now = (frames + 1) * 5
            need(bundle['time_s'] == now, 'Every5s public packet clock')
            while cursor < len(record['inspection_events']) and record['inspection_events'][cursor]['time_s'] <= now:
                v = record['inspection_events'][cursor]
                key = {'thermal':'temperature_c', 'electrical':'contact_temperature_c', 'cooling':'pod_flow_kg_s', 'control':'controller_sensor_error_c'}[v['kind']]
                history[v['task']].insert(0, (v['time_s'], v['measurement'][key], v['alarm']))
                cursor += 1
            for task in bundle['tasks']:
                for channel in CHANNELS:
                    v = task[channel]
                    if v['available']:
                        need(0 <= v['context_cutoff_s'] <= v['time_s'] <= now and abs(v['age_s']-now+v['time_s'])<1e-8, 'No future/inconsistent source clock')
                        available_readings += 1
            if now not in ROOTS:
                frames += 1
                continue
            packet = bundle['telemetry']
            expected_base = np.zeros((176, 85), np.float32)
            for j, v in enumerate(bundle['tasks']):
                pod, known = v['pod'], v['last_check_s'] >= 0
                row = [now, *(float(v['kind'] == k) for k in ('thermal','electrical','cooling','control')),
                    pod, pod, pod, v['weight_scaled'], v['service_s'], v['deadline_s'], float(known),
                    v['last_check_s'] if known else 0., now-v['last_check_s'] if known else 0.,
                    float(v['already_alarm']), v['velocity_c_s'], sum(packet['public_requested_it_kw'][pod*8:(pod+1)*8]),
                    sum(packet['public_delivered_it_kw'][pod*8:(pod+1)*8]), packet['pod_pump_command'][pod],
                    packet['secondary_supply_c'][pod], packet['primary_supply_c'], now-packet['time_s']]
                for channel in CHANNELS:
                    val = v[channel]
                    row.extend((1.,val['value'],val['time_s'],val['age_s'],val['context_cutoff_s'],SOURCE_CODES[val['source']]) if val['available'] else (0.,)*6)
                for lag in range(3):
                    old = history[j][lag] if lag < len(history[j]) else None
                    row.extend((1.,old[1],old[0],now-old[0],float(old[2])) if old else (0.,)*5)
                expected_base[j] = row
            need(np.array_equal(base[kept], expected_base), 'All85 root/history/telemetry/nominal/raw/cache columns')
            for h, horizon in enumerate(HORIZONS):
                end = now+horizon
                legal = np.zeros(176, bool); counts = np.zeros(176, np.int16)
                indices = np.full(176, -2, np.int16); selected = [None]*176
                pending = {v['agent']:(v['busy_target'],v['busy_finish_s']) for v in bundle['agents']}
                if end <= 1800:
                    for v in by_finish.get(end, []):
                        finish,j,dispatch,agent,optional,idx,tour,slack = v
                        if dispatch >= now or (not optional and pending[agent] == (j, finish)):
                            need(roles[agent] == bundle['tasks'][j]['required_role'], 'Qualified exact completion')
                            legal[j] = True; counts[j] += 1
                            if selected[j] is None:
                                selected[j] = v; indices[j] = idx
                    expected_a = a[end//5-1].astype(np.int8); expected_ref = ref[end//5-1].astype(np.int8)
                    expected_b = np.where(legal, expected_ref, -1).astype(np.int8)
                else:
                    expected_a = expected_ref = expected_b = np.full(176,-1,np.int8)
                for name, value in (('A',expected_a),('B_reference',expected_ref),('B_action',expected_b),('action_geometry_index',indices),('legal_action_count',counts)):
                    need(np.array_equal(targets[name][kept,h], value), 'Independent sampled '+name)
                expected_action = np.zeros((176,13),np.float32)
                expected_action[:,0] = horizon; expected_action[:,1] = end
                root_values = base[kept]
                expected_action[:,-1] = np.clip(root_values[:,fcol['temperature_measurement.value']] + root_values[:,fcol['temperature_velocity_c_s']] *
                    np.minimum(120., root_values[:,fcol['temperature_measurement.age_s']]+horizon),35.,110.) * root_values[:,fcol['temperature_measurement.available']]
                for j,v in enumerate(selected):
                    if v is not None:
                        finish,_,dispatch,_,optional,_,tour,slack = v
                        expected_action[j,2:12] = (1.,float(optional),dispatch-now,finish-dispatch,tour-now,slack,counts[j],1.,float(optional),float(not optional and dispatch<now))
                need(np.array_equal(action[kept,h], expected_action), 'All13 public ETA/qualification/return-slack fields')
            kept += 1; frames += 1
    need((frames,kept)==(359,59), 'All359 source frames and59 dataset roots verified')
    geometry_count=geometry_audit.finish(geom)
    summary = read(folder / 'summary.json')
    need(summary['feature_audit']['timestamp_available_readings_checked']==available_readings, 'All history timestamp count')
    return {'conditions':1,'history_frames':frames,'sampled_roots':kept,'available_readings':available_readings,
        'physical_label_cells':int(a.size),'sample_rows':int(targets['A'].size),
        'base_values':int(base.size),'action_values':int(action.size),'events':len(events),
        'original_measurements_recomputed':measured_checks,'complete_return_checks_recomputed':geometry_count}


def main():
    p=argparse.ArgumentParser()
    for n in ('root','protocol','manifest','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--group-index',type=int,required=True)
    args=p.parse_args(); protocol,manifest=frozen_inputs(args.protocol,args.manifest)
    groups=[(split,seed) for split,seeds in protocol['new_seed_groups'].items() for seed in seeds]
    split,seed=groups[args.group_index]; group=args.root/split/f'seed-{seed}'
    need(read(group/'complete.json')['conditions']==20,'Complete original group')
    total=Counter()
    for item in (v for v in manifest if v['split']==split and v['group_seed']==seed):
        total.update(verify_condition(group/item['id'],group))
    write(args.output,{'status':'passed','group_index':args.group_index,'split':split,'seed':seed,'counts':dict(total),
        'scope':'Independent original measurements; all physical/events, all85+13 public feature columns, exact legality and source hashes; no model fitting.'})
    print('FRESH_INDEPENDENT_DONE',args.group_index,json.dumps(dict(total)),flush=True)


if __name__=='__main__':main()
