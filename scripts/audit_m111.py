"""Whole pilot replay and independent event-time/fallback-plan verification."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from functools import lru_cache
import gzip
import hashlib
import heapq
import json
import math
from pathlib import Path
import runpy

import numpy as np
import torch

from patrol_rsi.aisles import build_agents
from patrol_rsi.covered_inspection import run_covered
from patrol_rsi.m11 import frozen_predictor, seed_digest
from patrol_rsi.m111 import specs, name, untrained_scorer


def require(value, message):
    if not value:
        raise RuntimeError(message)


def close(a, b):
    return abs(float(a) - float(b)) <= 1e-9 + 1e-10 * abs(float(b))


def equivalent(a, b, context=''):
    if isinstance(a, dict):
        require(set(a) == set(b), 'Keys ' + context)
        for key in a:
            if key in ('decision_CPU_seconds', 'history_estimator_inference_s'):
                continue
            equivalent(a[key], b[key], context + '/' + key)
    elif isinstance(a, list):
        require(len(a) == len(b), 'Length ' + context)
        for x, y in zip(a, b):
            equivalent(x, y, context)
    elif isinstance(a, float):
        require(close(a, b), 'Value ' + context)
    else:
        require(a == b, 'Value ' + context)


@lru_cache(maxsize=50000)
def distance(graph, source, target):
    """Independent Dijkstra, rather than the production A* / matrix."""
    queue, best = [(0., source)], {source: 0.}
    while queue:
        cost, node = heapq.heappop(queue)
        if cost > best[node]:
            continue
        if node == target:
            return cost
        for other, length in graph.adjacency[node].items():
            candidate = cost + length
            if candidate < best.get(other, float('inf')) - 1e-10:
                best[other] = candidate
                heapq.heappush(queue, (candidate, other))
    raise RuntimeError('Unreachable independent path')


def independent_certificates(record, env):
    agents = build_agents(env.graph, 'provisioned')
    by_task = {j: [] for j in range(len(env.tasks))}
    for event in record['inspection_events']:
        by_task[event['task']].append(event['time_s'])
    for values in by_task.values():
        values.sort()
    cursors = [0] * len(agents)
    selected = record['selected_certificates']
    require(len(selected) == len(record['assignments']), 'One selected plan per actual assignment')

    def duration(position, j):
        return math.ceil((distance(env.graph, tuple(position), env.tasks[j].position_m) +
                          env.tasks[j].service_s) / 5 - 1e-10) * 5.

    def check(clock, position, last, cursor, tour, candidate=None):
        virtual = {j: max(0., last[j]) for j in tour}
        due, finish = None, None
        next_cursor = cursor
        if candidate is not None:
            due = virtual[candidate] + env.deadlines[candidate]
            finish = clock + duration(position, candidate)
            if finish > due + 1e-9:
                return {'ok': False}
            clock, position = finish, env.tasks[candidate].position_m
            virtual[candidate] = finish
            if candidate == tour[cursor]:
                next_cursor = (cursor + 1) % len(tour)
        order = tour[next_cursor:] + tour[:next_cursor]
        times, dues = [], []
        for j in order:
            clock += duration(position, j)
            deadline = virtual[j] + env.deadlines[j]
            if clock > deadline + 1e-9:
                return {'ok': False}
            times.append(float(clock))
            dues.append(float(deadline))
            virtual[j], position = clock, env.tasks[j].position_m
        cycle = sum(duration(env.tasks[a].position_m, b) for a, b in zip(tour[-1:] + tour[:-1], tour))
        require(all(cycle <= env.deadlines[j] for j in tour), 'Repeat continuation period')
        return {'ok': True, 'candidate_finish_s': finish,
                'candidate_due_s': float(due) if due is not None else None,
                'cursor_after': next_cursor, 'fallback_order': order,
                'fallback_completion_s': times, 'fallback_due_s': dues,
                'repeating_cycle_s': cycle}

    for i, (agent, initial) in enumerate(zip(agents, record['initial_certificates'])):
        expected = check(5., agent['position'], {j: 0. for j in agent['tour']}, 0, agent['tour'])
        require(expected['ok'] and initial['ok'], 'Initial backlog')
        for key in ('fallback_order', 'fallback_completion_s', 'fallback_due_s', 'repeating_cycle_s'):
            equivalent(initial[key], expected[key], 'initial/' + key)
    accepted, detours = 0, 0
    for selected_plan, assignment in zip(selected, record['assignments']):
        i, j, clock = assignment['agent'], assignment['task'], assignment['time_s']
        tour = agents[i]['tour']
        require(j in tour and selected_plan['task'] == j and selected_plan['agent'] == i, 'Owned candidate')
        require(selected_plan['clock_s'] == clock and selected_plan['position'] == assignment['source'], 'Plan dispatch boundary')
        cursor = cursors[i]
        require(selected_plan['baseline_next'] == tour[cursor], 'Protected cursor')
        last = {}
        for k in tour:
            count = bisect_right(by_task[k], clock)
            last[k] = by_task[k][count - 1] if count else 0.
        expected = check(clock, assignment['source'], last, cursor, tour, j)
        stored = selected_plan['certificate']
        require(expected['ok'] == stored['ok'], 'Independent complete-round predicate')
        if selected_plan['protection_enabled']:
            require(expected['ok'], 'Protected dispatch must have a valid complete return')
        if expected['ok']:
            accepted += 1
            for key in ('candidate_finish_s', 'candidate_due_s', 'cursor_after', 'fallback_order',
                        'fallback_completion_s', 'fallback_due_s', 'repeating_cycle_s'):
                equivalent(stored[key], expected[key], 'selected/' + key)
            slacks = [d - t for d, t in zip(expected['fallback_due_s'], expected['fallback_completion_s'])]
            slacks.append(expected['candidate_due_s'] - expected['candidate_finish_s'])
            require(close(stored['minimum_projected_slack_s'], min(slacks)), 'Certificate slack')
        detours += j != tour[cursor]
        if j == tour[cursor]:
            cursors[i] = (cursor + 1) % len(tour)
    maxima = []
    gaps = record['exact_task_gaps']
    for j in range(len(env.tasks)):
        times = [0.] + by_task[j] + [1800.]
        maximum = max(b - a for a, b in zip(times, times[1:]))
        require(gaps[j]['task'] == j and gaps[j]['completion_times_s'] == by_task[j], 'Raw completion times')
        require(close(gaps[j]['maximum_gap_including_initial_and_terminal_s'], maximum), 'Exact head/interior/tail gap')
        maxima.append(maximum)
    ratios = np.asarray(maxima) / env.deadlines
    violations = int((ratios > 1 + 1e-9).sum())
    require(violations == record['metrics']['tasks_violating_exact_gap'], 'Exact gap violation count')
    require(close(max(ratios), record['metrics']['maximum_exact_gap_ratio']), 'Exact gap ratio')
    if env.protected:
        require(violations == 0, 'Protected comparator failed declared coverage')
    require(detours == record['metrics']['selected_detours'], 'Extra inspection count')
    return accepted, len(selected)


def independent_power_events(env, record):
    """Rebuild service evidence directly from the public packet sequence."""
    bad, good = [0] * 16, [0] * 16
    active, last_report = [False] * 16, [-float('inf')] * 16
    events = []
    for packet in env.trace.packets:
        t = packet['time_s']
        for pod in range(16):
            req = float(sum(packet['public_requested_it_kw'][pod * 8:(pod + 1) * 8]))
            got = float(sum(packet['public_delivered_it_kw'][pod * 8:(pod + 1) * 8]))
            gap = max(req - got, 0) / req if req > 1 else 0.
            flag = gap > .1 and req > 1
            bad[pod], good[pod] = (bad[pod] + 1, 0) if flag else (0, good[pod] + 1)
            kind = None
            if bad[pod] >= 3 and not active[pod]:
                active[pod], kind = True, 'new_measured_alert'
            elif bad[pod] >= 3 and t - last_report[pod] >= 60:
                kind = 'persistent_recheck'
            elif good[pod] >= 3 and active[pod]:
                active[pod], kind = False, 'recovery_confirmation'
            if kind:
                last_report[pod] = t
                events.append({'time_s': t, 'pod': pod, 'kind': kind,
                    'task': 'IT_service_availability', 'requested_kw': req, 'delivered_kw': got,
                    'service_gap_ratio': gap, 'calibration_declared_in_simulation': True,
                    'root_cause_diagnosed': False, 'voltage_or_current_measured': False})
    equivalent(events, record['service_evidence_events'], 'independent_public_power_evidence')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--estimator-weights', type=Path, required=True)
    p.add_argument('--estimator-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    predictor = frozen_predictor(args.estimator_weights, args.estimator_manifest)
    scorer, untrained, original = untrained_scorer()
    helpers = runpy.run_path(str(Path(__file__).with_name('audit_m11.py')))
    index = json.loads((args.results / 'case-index.json').read_text())
    require(len(index) == 128 and {row['case'] for row in index} == {name(s) for s in specs('pilot')}, 'Frozen pilot scope')
    processes = json.loads((args.results / 'process-index.json').read_text())
    checked_processes, certificates, dispatches, completions = set(), 0, 0, 0
    protected_count, maximum_ratio, maximum_energy, maximum_electric = 0, 0., 0., 0.
    for number, row in enumerate(index):
        path = args.results / row['path']
        require(hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'] and path.stat().st_size == row['bytes'], 'Raw case fingerprint')
        record = json.loads(gzip.decompress(path.read_bytes()))
        repeated, env = run_covered(row['specification'], predictor, scorer)
        equivalent(repeated, record)
        helpers['independent_schedule'](record, env)
        helpers['independent_events'](record, env)
        passed, count = independent_certificates(record, env)
        independent_power_events(env, record)
        certificates += passed
        dispatches += count
        completions += len(record['inspection_events'])
        if env.protected:
            protected_count += 1
            maximum_ratio = max(maximum_ratio, record['metrics']['maximum_exact_gap_ratio'])
        require(seed_digest(untrained) == original, 'No scoring network training')
        if row['process_key'] not in checked_processes:
            process = processes[row['process_key']]
            source = args.results / process['path']
            require(hashlib.sha256(source.read_bytes()).hexdigest() == process['sha256'], 'Process array bytes')
            with np.load(source, allow_pickle=False) as pack:
                for key in pack.files:
                    np.testing.assert_allclose(pack[key], env.trace.arrays[key], rtol=1e-11, atol=1e-9)
                c, scalar = env.config, pack['process_scalars']
                energy = (c.chip_capacity_kj_k * pack['rack_temperature_c'].sum(1) +
                    c.secondary_capacity_kj_k * pack['secondary_supply_c'].sum(1) +
                    c.primary_capacity_per_pod_kj_k * c.pods * pack['primary_supply_c'] +
                    c.contact_capacity_kj_k * pack['contact_temperature_c'].sum(1))
                residual = energy - np.r_[env.trace.metadata['initial_energy_kj'], energy[:-1]] - 5 * (scalar[:, 2] - scalar[:, 4])
                electric = scalar[:, 2] - scalar[:, 5] - scalar[:, 6] - scalar[:, 7] - scalar[:, 8]
                require(abs(residual).max() < 1e-4 and abs(electric).max() < 1e-6, 'Independent physical balance')
                maximum_energy = max(maximum_energy, float(abs(residual).max()))
                maximum_electric = max(maximum_electric, float(abs(electric).max()))
            checked_processes.add(row['process_key'])
        if number % 16 == 0:
            print('M111_AUDITED', number + 1, '/', len(index), flush=True)
    result = {'all_pilot_cases_replayed': len(index), 'unique_process_arrays_replayed': len(checked_processes),
        'fixed_estimator_bytes_verified': True, 'untrained_scorer_seed_and_no_updates_verified': True,
        'accepted_complete_round_certificates_independently_checked': certificates,
        'all_actual_dispatches_independently_checked': dispatches,
        'all_completed_measurements_independently_checked': completions,
        'protected_cases': protected_count, 'maximum_protected_exact_gap_ratio': maximum_ratio,
        'all_exact_completion_gaps_including_initial_and_terminal_recomputed': True,
        'all_public_power_evidence_lifecycles_independently_rebuilt': True,
        'maximum_independent_energy_residual_kj': maximum_energy,
        'maximum_independent_electric_residual_kw': maximum_electric,
        'scope': 'Startup prototype and deterministic known-layout certificates, not completeM11.1 policy research, a general safety theorem or field validation'}
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print('M111_FULL_STARTUP_AUDIT_PASSED', len(index), 'cases')


if __name__ == '__main__':
    main()
