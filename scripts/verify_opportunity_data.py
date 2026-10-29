"""Independent B0 physical/measurement labels, history and sample validity."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from patrol_rsi.active_inspection import ProcessTrace, graph_for, measured_anomaly
from patrol_rsi.aisles import build_agents
from patrol_rsi.causal_observation import CHANNELS, validate_public
from patrol_rsi.opportunity_data import ACTION_FEATURE_NAMES, FEATURE_NAMES, HORIZONS, SOURCE_CODES


FOLDERS = ('development-0', 'development-1', 'validation-0', 'validation-1',
           'test-0', 'test-1', 'test-2', 'test-3')


def need(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def direct_labels(arrays, seed):
    # Original measurement and alarm routines are independent of B0's
    # reference_signals/effective_confirmability convenience functions.
    tasks = graph_for(128, 'open').tasks
    physical = np.concatenate((arrays['rack_temperature_c'] > 65., arrays['contact_temperature_c'] > 60.,
        arrays['rack_flow_kg_s'].reshape(360, 16, 8).sum(2) < 33.6,
        np.abs((arrays['public_sensor_temperature_c'] - arrays['rack_temperature_c']).reshape(360, 16, 8).mean(2)) > 1.), axis=1)
    trace = SimpleNamespace(arrays=arrays, config=graph_for(128, 'open').config, seed=seed)
    confirms = np.zeros_like(physical)
    for tick, task in np.argwhere(physical):
        confirms[tick, task] = measured_anomaly(tasks[int(task)], ProcessTrace.measurement(trace, tasks[int(task)], int(tick)))
    return physical, confirms


def verify_condition(data, prior, source):
    identity = read(data / 'source-identity-not-features.json')
    summary = read(data / 'summary.json')
    spec = identity['condition']
    index = read(source / 'case-index.json')
    old = next(v for v in index if v['alias'] == 'tour' and v['specification'] == spec)
    record = read(source / old['path'])
    need(hashlib.sha256((source / old['path']).read_bytes()).hexdigest() == identity['source_Tour_sha256'], 'Source Tour binding')
    process = read(source / 'process-index.json')[old['process_key']]
    with np.load(source / process['path']) as values:
        arrays = {k: values[k].copy() for k in values.files}
    a, b = direct_labels(arrays, spec['seed'])
    with np.load(data / 'labels-only.npz') as labels:
        need(np.array_equal(a, labels['A_physical']) and np.array_equal(b, labels['B_reference']), 'Independent all professional A/B conditional labels')
        events = read(data / 'events-evaluator-only.json')
        for event in events:
            task, start, end = event['task'], event['first_tick'], event['last_exclusive_tick']
            need(a[start:end, task].all() and (start == 0 or not a[start - 1, task]) and (end == 360 or not a[end, task]), 'Maximal physical event identity')
            need(event['retained_at_least_15s'] == (end - start >= 3), 'Retained event duration')
            need(event['confirmable_positive_bins'] == int(b[start:end, task].sum()), 'Event confirmability')
    with np.load(data / 'public-features.npz') as features:
        base, action = features['base'], features['completion']
    with np.load(data / 'targets-evaluator-only.npz') as labels:
        targets = {k: labels[k].copy() for k in labels.files}
    need(base.shape == (359, 176, len(FEATURE_NAMES)) and action.shape == (359, 3, 176, len(ACTION_FEATURE_NAMES)), 'Full public dataset feature shapes')
    need(all(v.shape == (359, 3, 176) for v in targets.values()), 'Full target shapes')
    need(np.isfinite(base).all() and np.isfinite(action).all(), 'Finite public features with missing masks')
    with np.load(prior.parent / 'geometry.npz') as values:
        geometry = {k: values[k].copy() for k in values.files}
    # Independent baseline planned route arithmetic, not producer completion_lookup.
    planned = []
    for assignment in record['assignments']:
        start = assignment['time_s']
        finish = start + math.ceil((assignment['distance_m'] + assignment['service_s']) / 5 - 1e-10) * 5
        if finish <= 1800:
            planned.append((finish, assignment['task'], start, assignment['agent'],
                            False, -1, finish, 0.))
    for i in range(len(geometry['task'])):
        planned.append((float(geometry['finish_s'][i]), int(geometry['task'][i]),
            float(geometry['dispatch_time_s'][i]), int(geometry['agent'][i]), True, i,
            float(geometry['tour_target_finish_s'][i]), float(geometry['return_slack_s'][i])))
    by_finish = {}
    for row in sorted(planned, key=lambda v: (v[0], v[1], v[2], v[3], v[4])):
        by_finish.setdefault(row[0], []).append(row)
    executor_roles = {v['id']: v['role'] for v in build_agents(graph_for(128, 'open'), 'provisioned')}
    fcol = {name: i for i, name in enumerate(FEATURE_NAMES)}
    hcol = {name: i for i, name in enumerate(ACTION_FEATURE_NAMES)}
    history = [[] for _ in range(176)]
    event_cursor = 0
    total_available = frames = 0
    with gzip.open(prior / 'public-states.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            bundle = json.loads(line); validate_public(bundle)
            now = (frames + 1) * 5
            need(bundle['time_s'] == now and (base[frames, :, fcol['root_time_s']] == now).all(), 'Feature prediction root')
            packet = bundle['telemetry']
            for j, row in enumerate(bundle['tasks']):
                pod = row['pod']; known = row['last_check_s'] >= 0
                values = {'root_time_s': now, 'pod': pod, 'cdu_branch': pod, 'power_branch': pod,
                    'weight_scaled': row['weight_scaled'], 'service_s': row['service_s'],
                    'deadline_s': row['deadline_s'], 'last_check_available': float(known),
                    'last_check_s': row['last_check_s'] if known else 0.,
                    'inspection_age_s': now - row['last_check_s'] if known else 0.,
                    'already_measured_alarm': float(row['already_alarm']),
                    'temperature_velocity_c_s': row['velocity_c_s'],
                    'public_requested_pod_kw': sum(packet['public_requested_it_kw'][pod * 8:(pod + 1) * 8]),
                    'public_delivered_pod_kw': sum(packet['public_delivered_it_kw'][pod * 8:(pod + 1) * 8]),
                    'public_pump_command': packet['pod_pump_command'][pod],
                    'public_secondary_supply_c': packet['secondary_supply_c'][pod],
                    'public_primary_supply_c': packet['primary_supply_c'],
                    'telemetry_age_s': now - packet['time_s']}
                values.update({role: float(row['kind'] == role) for role in ('thermal', 'electrical', 'cooling', 'control')})
                columns = [fcol[name] for name in values]
                need(np.array_equal(base[frames, j, columns], np.asarray(list(values.values()), dtype=np.float32)),
                     'Public task/topology/telemetry/clock feature column differs')
            while event_cursor < len(record['inspection_events']) and record['inspection_events'][event_cursor]['time_s'] <= now:
                event = record['inspection_events'][event_cursor]
                key = {'thermal': 'temperature_c', 'electrical': 'contact_temperature_c',
                       'cooling': 'pod_flow_kg_s', 'control': 'controller_sensor_error_c'}[event['kind']]
                history[event['task']].insert(0, (event['time_s'], event['measurement'][key], event['alarm']))
                event_cursor += 1
            for channel in CHANNELS:
                rows = [row[channel] for row in bundle['tasks']]
                available = np.asarray([v['available'] for v in rows])
                for name in ('available', 'value', 'time_s', 'age_s', 'context_cutoff_s', 'source_code'):
                    expected = [float(v['available']) if name == 'available' else
                        SOURCE_CODES[v['source']] if name == 'source_code' and v['available'] else
                        v[name] if name != 'source_code' and v['available'] else 0. for v in rows]
                    need(np.array_equal(base[frames, :, fcol[channel + '.' + name]], np.asarray(expected, dtype=np.float32)), 'Root-known channel value/source/timestamp changed')
                need(all(not v['available'] or 0 <= v['context_cutoff_s'] <= v['time_s'] <= now for v in rows), 'Future channel content')
                total_available += int(available.sum())
            for j in range(176):
                for lag in range(3):
                    old_event = history[j][lag] if lag < len(history[j]) else None
                    expected = (1., old_event[1], old_event[0], now - old_event[0], float(old_event[2])) if old_event else (0.,) * 5
                    columns = [fcol[f'qualified_history_{lag}.' + key] for key in ('available', 'value', 'time_s', 'age_s', 'alarm')]
                    need(np.array_equal(base[frames, j, columns], np.asarray(expected, dtype=np.float32)), 'Independent qualified-check history differs')
            for h, horizon in enumerate(HORIZONS):
                end = now + horizon
                legal = np.zeros(176, dtype=bool)
                representative = [None] * 176
                multiplicity = np.zeros(176, dtype=np.int16)
                expected_ids = np.full(176, -2, dtype=np.int16)
                if end <= 1800:
                    pending = {v['agent']: (v['busy_target'], v['busy_finish_s']) for v in bundle['agents']}
                    for item in by_finish.get(end, []):
                        finish, task, dispatch, agent, optional, geometry_index, tour_finish, slack = item
                        if dispatch >= now or (not optional and pending[agent] == (task, finish)):
                            legal[task] = True
                            multiplicity[task] += 1
                            if representative[task] is None:
                                representative[task] = item
                                expected_ids[task] = geometry_index
                            need(executor_roles[agent] == bundle['tasks'][task]['required_role'], 'Qualified executor role')
                    tick = int(end / 5) - 1
                    expected_a, expected_ref = a[tick].astype(np.int8), b[tick].astype(np.int8)
                    expected_b = np.where(legal, expected_ref, -1).astype(np.int8)
                else:
                    expected_a = expected_ref = expected_b = np.full(176, -1, dtype=np.int8)
                need(np.array_equal(targets['A'][frames, h], expected_a) and np.array_equal(targets['B_reference'][frames, h], expected_ref), 'Sample A/B reference labels')
                need(np.array_equal(targets['B_action'][frames, h], expected_b), 'All legal completion validity and B labels')
                need(np.array_equal(action[frames, h, :, hcol['legal_completion_available']].astype(bool), legal), 'Legal feature mask differs from root public schedule')
                need((action[frames, h, :, hcol['horizon_s']] == horizon).all() and
                     (action[frames, h, :, hcol['completion_time_s']] == end).all(), 'Known endpoint features')
                need(np.array_equal(targets['action_geometry_index'][frames, h], expected_ids) and
                     np.array_equal(targets['legal_action_count'][frames, h], multiplicity), 'C-sidecar join IDs and legal multiplicity')
                expected_action = np.zeros((176, len(ACTION_FEATURE_NAMES)), dtype=np.float32)
                expected_action[:, hcol['horizon_s']] = horizon
                expected_action[:, hcol['completion_time_s']] = end
                # The public matrix is explicitly float32. Apply the published
                # extrapolation to those same quantized root values, not truth.
                for task in range(176):
                    measurement = base[frames, task, fcol['temperature_measurement.value']]
                    velocity = base[frames, task, fcol['temperature_velocity_c_s']]
                    age = base[frames, task, fcol['temperature_measurement.age_s']]
                    # Array operation order follows the documented float32
                    # feature representation; casting only at the end can
                    # produce a different final ULP.
                    advance = np.float32(velocity * np.minimum(np.float32(120.), np.float32(age + horizon)))
                    predicted = np.float32(measurement + advance)
                    predicted = np.clip(predicted, np.float32(35.), np.float32(110.))
                    expected_action[task, hcol['bounded_trend_at_completion_c']] = predicted * base[frames, task, fcol['temperature_measurement.available']]
                    item = representative[task]
                    if item is not None:
                        finish, _, dispatch, _, optional, _, tour_finish, slack = item
                        fields = {'legal_completion_available': 1., 'optional_check': float(optional),
                            'planned_dispatch_offset_s': dispatch - now,
                            'planned_service_and_travel_s': finish - dispatch,
                            'planned_Tour_target_offset_s': tour_finish - now,
                            'complete_return_slack_s': slack, 'legal_completion_count': multiplicity[task],
                            'qualified_executor_available': 1., 'return_slack_available': float(optional),
                            'inflight_protected_Tour': float(not optional and dispatch < now)}
                        for field, value in fields.items():
                            expected_action[task, hcol[field]] = value
                need(np.array_equal(action[frames, h], expected_action), 'All public completion/ETA/slack/inflight/qualification features')
            frames += 1
    need(frames == 359 and total_available == summary['feature_audit']['timestamp_available_readings_checked'], 'Complete all-root timestamp checks')
    need(len(fcol) == 22 + 8 * 6 + 3 * 5, 'All base feature columns must be explicitly covered by verifier')
    return {'frames': frames, 'available_readings': total_available, 'A_B_physical_cells': int(a.size),
            'public_base_values_verified': int(base.size), 'public_action_values_verified': int(action.size),
            'samples': int(targets['A'].size), 'events': len(events)}


def main():
    parser = argparse.ArgumentParser()
    for name in ('data-root', 'b4a-root', 'source-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    totals = Counter()
    for folder in FOLDERS:
        for i in range(16):
            counts = verify_condition(args.data_root / folder / f'condition-{i:02}',
                                      args.b4a_root / folder / f'condition-{i:02}', args.source_root / folder)
            totals.update(counts)
        print('B4B0_INDEPENDENT_FOLDER_PASSED', folder, dict(totals), flush=True)
    result = {'status': 'passed', 'conditions': 128, 'counts': dict(totals),
        'scope': 'Independent original measurements, professional truth, all exact completion masks, source channel values/timestamps, qualified history and A/B labels. No predictor fit or new process.'}
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print('B4B0_INDEPENDENT_ALL_PASSED', flush=True)


if __name__ == '__main__':
    main()
