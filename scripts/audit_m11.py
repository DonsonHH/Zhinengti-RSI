"""Full M11 actual-policy replay, independent route/clock/event verification."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from patrol_rsi.aisles import ROLES, build_agents
from patrol_rsi.estimation_data import ATOMS
from patrol_rsi.inspection_model import TaskPolicy
from patrol_rsi.m11 import (frozen_predictor, seed_digest, test_specs, case_name,
                            run_episode, validation_specs)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def equivalent(left, right, context=''):
    if isinstance(left, dict):
        require(set(left) == set(right), 'Keys ' + context)
        for key in left:
            if key in ('decision_CPU_seconds', 'history_estimator_inference_s'):
                continue
            equivalent(left[key], right[key], context + '/' + key)
    elif isinstance(left, list):
        require(len(left) == len(right), 'List ' + context)
        for i, (a, b) in enumerate(zip(left, right)):
            equivalent(a, b, context + '/' + str(i))
    elif isinstance(left, float):
        require(np.isclose(left, right, rtol=1e-10, atol=1e-9), 'Numeric ' + context)
    else:
        require(left == right, 'Value ' + context)


def independent_schedule(record, env):
    graph, tasks = env.graph, env.tasks
    agents = build_agents(graph, env.spec['resources'])
    positions = [a['position'] for a in agents]
    finished = [5.] * len(agents)
    reservations, assignments_by_agent = {}, {}
    events = {(e['agent'], e['task'], e['time_s']): e for e in record['inspection_events']}
    matched = set()
    for assignment in record['assignments']:
        i, j, start = assignment['agent'], assignment['task'], assignment['time_s']
        task = tasks[j]
        require(task.required_role == agents[i]['role'] and task.access == 'external_sensing', 'Role/access')
        require(j in agents[i]['allowed'], 'Fixed work area')
        require(start >= finished[i] - 1e-9 and tuple(assignment['source']) == positions[i], 'Busy or wrong position')
        route = [tuple(p) for p in assignment['route']]
        require(route[0] == positions[i] and route[-1] == task.position_m, 'Route endpoints')
        distance = 0.
        for a, b in zip(route, route[1:]):
            require(b in graph.adjacency[a], 'Through-rack or closed-aisle traversal')
            distance += graph.adjacency[a][b]
        require(np.isclose(distance, assignment['distance_m'], atol=1e-9), 'Route distance')
        duration = np.ceil((distance + task.service_s) / 5 - 1e-10) * 5
        finish = start + duration
        require(finish <= 1800 and assignment['service_s'] == task.service_s, 'Service/horizon')
        require(start >= reservations.get(j, 0) - 1e-9, 'Duplicate task reservation')
        key = (i, j, finish)
        require(key in events, 'Travel/service incomplete before measurement')
        event = events[key]
        require(event['role'] == task.required_role and event['calibrated'] and event['access'] == task.access, 'Completion eligibility')
        expected = env.trace.measurement(task, int(finish / 5) - 1)
        equivalent(expected, event['measurement'], 'actual_measurement')
        reservations[j], finished[i], positions[i] = finish, finish, task.position_m
        assignments_by_agent.setdefault(i, []).append(assignment)
        matched.add(key)
    require(set(events) == matched and len(events) == len(record['inspection_events']), 'All completions matched once')
    for agent in record['team']['per_agent']:
        require(np.isclose(agent['travel_m'] + agent['service_s'] + agent['idle_s'], 1795), 'Physical time conservation')
        completed = assignments_by_agent.get(agent['id'], [])
        require(np.isclose(sum(a['distance_m'] for a in completed), agent['travel_m'], atol=1e-7), 'Actual walking')
        require(np.isclose(sum(a['service_s'] for a in completed), agent['service_s'], atol=1e-7), 'Actual service')
        require(agent['completed'] == len(completed), 'Completion count')
    last, ever, cursor = np.full(len(tasks), -1.), np.zeros(len(tasks), dtype=bool), 0
    ordered_events = sorted(record['inspection_events'], key=lambda e: e['time_s'])
    maximum_age, overdue = 0., []
    for row in record['scalar_trace']:
        t = row['time_s']
        while cursor < len(ordered_events) and ordered_events[cursor]['time_s'] <= t:
            j = ordered_events[cursor]['task']
            last[j], ever[j] = ordered_events[cursor]['time_s'], True
            cursor += 1
        age = np.where(last >= 0, t - last, t)
        ratio = age / env.deadlines
        require(np.isclose(row['maximum_age_s'], age.max()), 'Maximum waiting')
        require(np.isclose(row['coverage'], ever.mean()), 'Coverage')
        require(np.isclose(row['mean_age_ratio'], np.minimum(ratio, 6).mean()), 'Age ratio')
        require(np.isclose(row['overdue_fraction'], (ratio > 1).mean()), 'Overdue')
        if 'public_reward' in row:
            expected = (2 * row['new_coverage'] - .1 * row['overdue_fraction'] -
                        .02 * row['mean_age_ratio'] - .02 * row['public_risk_age'] +
                        .02 * row['completed_alarm_count'] / len(agents) -
                        .0001 * row['travel_increment_m'] / len(agents))
            require(np.isclose(row['public_reward'], expected, atol=1e-12), 'Public reward formula')
        for role in ROLES:
            mask = [task.kind == role for task in tasks]
            require(np.isclose(row['per_role_overdue'][role], (ratio[mask] > 1).mean()), 'Role overdue')
            require(np.isclose(row['per_role_maximum_age_s'][role], age[mask].max()), 'Role waiting')
        maximum_age = max(maximum_age, age.max())
        overdue.append((ratio > 1).mean())
    require(np.isclose(record['metrics']['maximum_inspection_age_s'], maximum_age), 'Episode longest wait')
    require(np.isclose(record['metrics']['mean_overdue_fraction'], np.mean(overdue)), 'Episode overdue')
    require(record['metrics']['invalid_assignments'] == 0 and record['metrics']['duplicate_reservations'] == 0, 'Illegal operation metric')


def independent_events(record, env):
    a, c, tasks = env.trace.arrays, env.config, env.tasks
    raw = a['rack_temperature_c']
    contact = a['contact_temperature_c']
    flows = a['rack_flow_kg_s'].reshape(360, c.pods, 8).sum(2)
    biases = abs((a['public_sensor_temperature_c'] - raw).reshape(360, c.pods, 8).mean(2))
    intervals, labels = [], []
    for j, task in enumerate(tasks):
        if task.kind == 'thermal':
            value = raw[:, task.index]
            flags, amplitude = value > 65, np.maximum(value - 65, 0) / 20
        elif task.kind == 'electrical':
            value = contact[:, task.index]
            flags, amplitude = value > 60, np.maximum(value - 60, 0) / 30
        elif task.kind == 'cooling':
            value = flows[:, task.index]
            flags, amplitude = value < 33.6, np.maximum(1 - value / 33.6, 0)
        else:
            value = biases[:, task.index]
            flags, amplitude = value > 1, value / 18
        labels.append(flags)
        start = None
        for tick in range(361):
            positive = tick < 360 and bool(flags[tick])
            if positive and start is None:
                start = tick
            if not positive and start is not None:
                if tick - start >= 3:
                    alarms = [e['time_s'] for e in record['inspection_events'] if e['task'] == j and e['alarm']
                              and start * 5 <= e['time_s'] <= tick * 5
                              and flags[min(359, int(e['time_s'] / 5) - 1)]]
                    intervals.append({'task': j, 'task_id': task.task_id, 'kind': task.kind,
                        'onset_s': float(start * 5), 'end_s': float(tick * 5), 'samples': tick - start,
                        'weight': float(env.asset_weights[j] * (1 + amplitude[start:tick].mean())),
                        'first_confirmation_s': min(alarms) if alarms else None,
                        'detected': bool(alarms),
                        'delay_or_censor_s': float((min(alarms) if alarms else tick * 5) - start * 5)})
                start = None
    equivalent(intervals, record['anomaly_intervals'], 'independent_anomaly_segments')
    m = record['metrics']
    count, detected = len(intervals), sum(r['detected'] for r in intervals)
    require((m['anomaly_segments'], m['confirmed_segments'], m['missed_segments']) == (count, detected, count - detected), 'Detection counts')
    if count:
        weighted = sum(r['weight'] * r['delay_or_censor_s'] for r in intervals) / sum(r['weight'] for r in intervals)
        require(np.isclose(weighted, m['consequence_weighted_censored_delay_s']), 'Censored weighted delay')
        require(np.isclose(1 - detected / count, m['miss_rate']), 'Miss rate')
    else:
        require(m['miss_rate'] is None and m['consequence_weighted_censored_delay_s'] is None, 'No-event nulls')
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--estimator-weights', type=Path, required=True)
    parser.add_argument('--estimator-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    predictor = frozen_predictor(args.estimator_weights, args.estimator_manifest)
    policies, training, initial = {}, {}, {}
    for mode in ('nominal', 'history'):
        for seed in (0, 1, 2):
            folder = args.models / f'{mode}-seed-{seed}'
            record = json.loads((folder / 'training.json').read_text())
            require(not record['smoke'] and record['episodes'] == 48 and record['decision_steps'] == 17232, 'Formal policy budget')
            require(record['PPO_epochs'] == 192 and len(record['learning']) == 48, 'Update budget')
            rng = np.random.default_rng(15000 + seed)
            for episode, row in enumerate(record['learning']):
                spec = row['process']
                require(spec['seed'] == 1000 + seed * 100 + episode and spec['scenario'] == str(rng.choice(ATOMS)), 'Paired physical training plan')
                require(row['update']['team_steps'] == 359 and len(row['update']['losses']) == 4, 'Actual rollout and update epochs')
            candidates = [row for row in record['learning'] if 'validation_public_return' in row]
            require([r['episode'] for r in candidates] == [12, 24, 36, 48], 'Validation schedule')
            best = max(candidates, key=lambda r: r['validation_public_return'])
            require(best['episode'] == record['selected_episode'] and best['validation_public_return'] == record['best_validation_public_return'], 'Validation-only selection')
            torch.manual_seed(seed)
            frozen_predictor(args.estimator_weights, args.estimator_manifest)
            model = TaskPolicy()
            require(seed_digest(model) == record['initial_parameters_sha256'], 'Actual initial parameters')
            initial.setdefault(seed, set()).add(record['initial_parameters_sha256'])
            path = folder / 'selected.pt'
            require(hashlib.sha256(path.read_bytes()).hexdigest() == record['checkpoint_sha256'], 'Selected weight bytes')
            pack = torch.load(path, map_location='cpu', weights_only=True)
            model.load_state_dict(pack['state_dict'])
            require(seed_digest(model) != record['initial_parameters_sha256'], 'Policy changed by training')
            model.eval()
            validation = [run_episode(spec, predictor, 'ppo_guard', model)[0]['metrics']['public_return'] for spec in validation_specs(mode)]
            require(np.allclose(validation, best['per_validation_case'], atol=1e-7, rtol=1e-9), 'Actual selected-policy validation replay')
            policies[(mode, seed)], training[(mode, seed)] = model, record
    require(all(len(v) == 1 for v in initial.values()), 'Paired initialization across estimators')
    for seed in (0, 1, 2):
        for left, right in zip(training[('nominal', seed)]['learning'], training[('history', seed)]['learning']):
            a = {k: v for k, v in left['process'].items() if k != 'estimator'}
            b = {k: v for k, v in right['process'].items() if k != 'estimator'}
            require(a == b, 'Same environment budget and cases')
    index = json.loads((args.results / 'case-index.json').read_text())
    require(len(index) == 1136 and {r['case'] for r in index} == {case_name(s) for s in test_specs()}, 'Frozen formal scope')
    process_index = json.loads((args.results / 'process-index.json').read_text())
    processes, event_count, completed_count, assignments, physical_maxima = set(), 0, 0, 0, {'electric_kw': 0., 'energy_kj': 0.}
    for number, item in enumerate(index):
        path = args.results / item['record']
        require(path.stat().st_size == item['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], 'Stored case bytes')
        record = json.loads(gzip.decompress(path.read_bytes()))
        spec = item['specification']
        repeated, env, _ = run_episode(spec, predictor, spec['method'], policies.get((spec['estimator'], spec['training_seed'])))
        equivalent(repeated, record, item['case'])
        equivalent(item['metrics'], record['metrics'], 'index_metrics')
        independent_schedule(record, env)
        event_count += independent_events(record, env)
        completed_count += len(record['inspection_events'])
        assignments += len(record['assignments'])
        if item['process_key'] not in processes:
            row = process_index[item['process_key']]
            rawpath = args.results / row['path']
            require(rawpath.stat().st_size == row['bytes'] and hashlib.sha256(rawpath.read_bytes()).hexdigest() == row['sha256'], 'Full process arrays')
            with np.load(rawpath, allow_pickle=False) as pack:
                require(set(pack.files) == set(env.trace.arrays), 'Process channel schema')
                for key in pack.files:
                    np.testing.assert_allclose(pack[key], env.trace.arrays[key], rtol=1e-11, atol=1e-9)
                c = env.config
                energy = (c.chip_capacity_kj_k * pack['rack_temperature_c'].sum(1) +
                          c.secondary_capacity_kj_k * pack['secondary_supply_c'].sum(1) +
                          c.primary_capacity_per_pod_kj_k * c.pods * pack['primary_supply_c'] +
                          c.contact_capacity_kj_k * pack['contact_temperature_c'].sum(1))
                scalar = pack['process_scalars']
                residual = energy - np.r_[env.trace.metadata['initial_energy_kj'], energy[:-1]] - 5 * (scalar[:, 2] - scalar[:, 4])
                electric = scalar[:, 2] - scalar[:, 5] - scalar[:, 6] - scalar[:, 7] - scalar[:, 8]
                require(abs(residual).max() < 1e-4 and abs(electric).max() < 1e-6, 'Independent process balances')
                physical_maxima['energy_kj'] = max(physical_maxima['energy_kj'], float(abs(residual).max()))
                physical_maxima['electric_kw'] = max(physical_maxima['electric_kw'], float(abs(electric).max()))
            require(row['process']['numeric_digest'] == env.trace.fingerprint, 'Same process across policies')
            if env.config.racks == 3200:
                require(env.trace.metadata['peak_source_input_MW'] <= 1000 and env.trace.metadata['IT_nameplate_MW'] == 800, 'GW units and input')
            processes.add(item['process_key'])
        if number % 40 == 0:
            print('M11_AUDITED', number + 1, '/', len(index), flush=True)
    result = {'all_test_episodes_replayed': len(index), 'task_clock_steps': len(index) * 359,
              'unique_full_process_arrays_read_and_replayed': len(processes),
              'unique_process_physical_steps': len(processes) * 360,
              'actual_selected_policy_weights_read': 6, 'actual_fixed_temperature_estimator_read': True,
              'paired_initialization_and_training_cases_verified': True,
              'actual_selected_policy_validation_replayed': True,
              'all_assignments_independently_verified': assignments,
              'all_completed_measurements_independently_verified': completed_count,
              'all_anomaly_segments_independently_verified': event_count,
              'graph_paths_roles_zones_reservations_and_service_clocks_verified': True,
              'all_ages_overdue_and_censoring_recomputed': True,
              'independent_physical_balance_maxima': physical_maxima,
              'scope': 'Complete synthetic closed-loop task/policy/measurement audit; no site calibration, optimality or RSI claim'}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print('M11_FULL_AUDIT_PASSED', len(index), 'closed-loop episodes;', len(processes), 'full process arrays')


if __name__ == '__main__':
    main()
