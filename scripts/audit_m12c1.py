"""Actual C1-A replay plus independent public candidates and finite witnesses."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

import patrol_rsi.opportunity_run as runner
from patrol_rsi.improver_task import PatrolProgram
from patrol_rsi.m12 import actor_scorer, load_actors
from patrol_rsi.m11 import frozen_predictor, seed_digest
from patrol_rsi.m12c1 import specs, name
from audit_m11 import independent_schedule, independent_events
from audit_m111 import equivalent, independent_certificates, independent_power_events, distance
from audit_m12 import independent_public_and_quota


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


ORIGINAL_SNAPSHOT = runner.snapshot
ORIGINAL_CHOOSE = runner.choose
ACTIVE_PROBE = None
CANDIDATES_CHECKED = 0
CHOICES_CHECKED = 0


def independent_snapshot(env, scorer):
    """Rebuild legal masks including unselected options, with independent paths."""
    global CANDIDATES_CHECKED
    packet, obs = env.current_packet, env.state.observer
    temperature = obs.predicted_temperature if env.mode == 'nominal' else env.current_history
    disagreement = np.zeros(128) if env.mode == 'nominal' else abs(temperature - obs.predicted_temperature) / 20
    thermal = np.maximum(temperature - 65, 0) / 20
    pod_thermal = thermal.reshape(16, 8).max(1)
    raw_gap = 1 - packet['public_delivered_it_kw'] / np.maximum(packet['public_requested_it_kw'], 1e-12)
    pod_gap = raw_gap.reshape(16, 8).mean(1)
    remote = dict(zip(packet['temperature_indices'].tolist(), packet['temperature_c'].tolist()))
    groups = []
    def duration(a, target):
        length = distance(env.graph, tuple(a), env.tasks[target].position_m)
        return math.ceil((length + env.tasks[target].service_s) / 5 - 1e-10) * 5.
    for raw_i in np.roll(np.arange(len(env.agents)), -(env.tick % len(env.agents))):
        i = int(raw_i);agent = env.agents[i]
        if agent['task'] is not None:
            continue
        tour = tuple(agent['tour']);cursor = int(env.protected_cursor[i]);baseline = tour[cursor]
        period = sum(duration(env.tasks[a].position_m, b) for a, b in zip(tour[-1:] + tour[:-1], tour))
        need(all(period <= env.deadlines[j] + 1e-9 for j in tour), 'Independent repeating-tour feasibility')
        rows, feature_rows = [], []
        for j in tour:
            t = env.tasks[j]
            need(t.required_role == agent['role'] and t.access == 'external_sensing', 'Profession/access before candidate creation')
            if j != baseline and (env.detours_used[i] >= 2 or
                env.last_inspection[j] >= 0 and env.clock - env.last_inspection[j] < 60):
                continue
            length = distance(env.graph, tuple(agent['position']), t.position_m)
            eta = duration(agent['position'], j);finish = env.clock + eta
            if finish > 1800:
                continue
            virtual = {k: max(0., float(env.last_inspection[k])) for k in tour}
            own_due = virtual[j] + env.deadlines[j]
            if finish > own_due + 1e-9:
                continue
            slack = own_due - finish
            virtual[j] = finish
            next_cursor = (cursor + 1) % len(tour) if j == baseline else cursor
            order = tour[next_cursor:] + tour[:next_cursor]
            clock, position, valid = finish, t.position_m, True
            for k in order:
                clock += duration(position, k)
                due = virtual[k] + env.deadlines[k]
                if clock > due + 1e-9:
                    valid = False
                    break
                slack = min(slack, due - clock)
                virtual[k], position = clock, env.tasks[k].position_m
            if not valid:
                continue
            d = env.deadlines[j]
            age = env.clock - env.last_inspection[j] if env.last_inspection[j] >= 0 else env.clock
            rack = t.index if t.kind == 'thermal' else t.index * 8
            stamp = obs.temperature_stamp[rack]
            measured_age = env.clock - stamp if stamp >= 0 else env.clock
            if t.kind == 'thermal':
                severity = thermal[t.index] + .25 * raw_gap[t.index]
            elif t.kind == 'electrical':
                severity = max(obs.last_contact[t.index] - 60, 0) / 30 + pod_gap[t.index] + .1 * pod_thermal[t.index]
            elif t.kind == 'cooling':
                severity = max(1 - obs.last_flow[t.index] / 33.6, 0) + .25 * pod_thermal[t.index]
            else:
                severity = float(obs.sensor_bias_verified[t.index]) + .25 * pod_thermal[t.index]
            f = np.asarray([min(age / d, 6), eta / d, min(severity, 4), float(not env.ever[j]),
                min(measured_age / 300, 6), min(disagreement[rack], 4), env.asset_weights[j] / 2,
                t.service_s / d, length / d, *[float(t.kind == r) for r in ('thermal', 'electrical', 'cooling', 'control')],
                env.clock / 1800, (t.index % 8) / 7, 1 - env.ever.mean()], dtype=np.float32)
            signals = {'thermal_excess': max(temperature[t.index] - 65, 0) / 20 if t.kind == 'thermal' else 0.,
                'thermal_disagreement': abs(temperature[t.index] - obs.predicted_temperature[t.index]) / 20 if t.kind == 'thermal' else 0.,
                'contact_heat': max(obs.last_contact[t.index] - 60, 0) / 30 if t.kind == 'electrical' else 0.,
                'flow_deficit': max(1 - obs.last_flow[t.index] / 33.6, 0) if t.kind == 'cooling' else 0.,
                'verified_sensor_bias': float(obs.sensor_bias_verified[t.index]) if t.kind == 'control' else 0.}
            signals['raw_remote_thermal_excess'] = max(remote[t.index] - 65, 0) / 20 if t.kind == 'thermal' and t.index in remote else 0.
            row = {'task': j, 'task_id': t.task_id, 'profession': t.kind,
                'features': f.tolist(), 'signals': {k: float(v) for k, v in signals.items()},
                'known_measured_alarm': bool(env.measured_alert.get(j, False)), 'cue_strength': float(max(signals.values())),
                'asset_weight_scaled': float(f[6]), 'eta_over_deadline': float(f[1]),
                'candidate_finish_s': finish, 'minimum_return_slack_s': float(slack),
                'reference_measurement_stamp_s': float(env.last_inspection[j]),
                'raw_remote_temperature_c': float(remote[t.index]) if t.kind == 'thermal' and t.index in remote else None,
                'raw_remote_temperature_time_s': float(packet['time_s']) if t.kind == 'thermal' and t.index in remote else None,
                'nominal_temperature_c': float(obs.predicted_temperature[t.index]) if t.kind == 'thermal' else None,
                'history_temperature_c': float(env.current_history[t.index]) if t.kind == 'thermal' and env.mode == 'history' else None,
                'baseline_next': baseline, 'is_optional': j != baseline}
            feature_rows.append(f);rows.append(row)
        if not any(r['task'] == baseline for r in rows):
            rows, feature_rows = [], []
        if rows:
            scores = scorer(np.stack(feature_rows))
            for row, score in zip(rows, scores):
                row['actor_score'] = float(score)
        CANDIDATES_CHECKED += len(rows)
        groups.append({'time_s': float(env.clock), 'agent': i, 'role': agent['role'], 'baseline_next': baseline,
            'position_m': list(agent['position']), 'detours_used': int(env.detours_used[i]),
            'baseline_completed': int(env.baseline_completed[i]), 'candidates': rows})
    return groups


def checked_snapshot(env, scorer):
    actual = ORIGINAL_SNAPSHOT(env, scorer)
    expected = independent_snapshot(env, scorer)
    equivalent(actual, expected, 'Independent public snapshot and full candidate mask')
    if ACTIVE_PROBE is not None:
        ACTIVE_PROBE.observe(env, expected)
    return actual


def checked_choose(groups, method):
    global CHOICES_CHECKED
    actual, logs = ORIGINAL_CHOOSE(groups, method)
    choices, expected_logs = {}, []
    for g in groups:
        rows = g['candidates']
        if not rows:
            choices[g['agent']] = None
            continue
        actor_index = int(np.argmax([r['actor_score'] for r in rows]))
        baseline = next(r for r in rows if r['task'] == g['baseline_next'])
        if method == 'actor':
            selected, tool = rows[actor_index], 'ppo'
        elif method == 'tour':
            selected, tool = baseline, 'return_tour'
        else:
            cued = [r for r in rows if r['cue_strength'] >= .05 and not r['known_measured_alarm']]
            if not cued:
                selected, tool = baseline, 'return_tour'
            else:
                priorities = [(.01 + r['cue_strength']) * r['asset_weight_scaled'] / (.03 + r['eta_over_deadline']) for r in cued]
                selected, tool = cued[int(np.argmax(priorities))], 'public_cue'
        choices[g['agent']] = selected['task']
        expected_logs.append({'time_s': g['time_s'], 'agent': g['agent'], 'task': selected['task'],
            'baseline_next': g['baseline_next'], 'actor_task': rows[actor_index]['task'], 'tool': tool,
            'rule_index': 0 if tool == 'public_cue' else None, 'legal_candidates': len(rows),
            'cue_strength': selected['cue_strength'], 'known_measured_alarm': selected['known_measured_alarm'],
            'detours_used_before': g['detours_used'], 'baseline_completed_before': g['baseline_completed']})
        CHOICES_CHECKED += 1
    need(actual == choices, 'Independent public controller actions')
    equivalent(logs, expected_logs, 'Independent public tool attribution')
    return actual, logs


class IndependentProbe:
    def __init__(self, intervals):
        self.intervals = intervals
        self.free, self.candidates = 0, 0
        self.rows = [{'segment_index': i, 'task': r['task'], 'task_id': r['task_id'], 'profession': r['kind'],
            'onset_s_evaluator_only': r['onset_s'], 'end_s_evaluator_only': r['end_s'],
            'witness_count': 0, 'any_cue_witness_count': 0, 'unconfirmed_cue_witness_count': 0,
            'first_witness': None, 'first_unconfirmed_cue_witness': None} for i, r in enumerate(intervals)]

    def observe(self, env, groups):
        self.free += len(groups)
        for g in groups:
            for candidate in g['candidates']:
                self.candidates += 1
                if not candidate['is_optional']:
                    continue
                j, finish = candidate['task'], candidate['candidate_finish_s']
                matching = [i for i, r in enumerate(self.intervals) if r['task'] == j and r['onset_s'] < finish <= r['end_s']]
                if not matching:
                    continue
                item, arrays = env.tasks[j], env.trace.arrays
                tick = int(finish / 5) - 1
                if item.kind == 'thermal':
                    noise = np.random.default_rng(3000000 + env.spec['seed'] * 10000 + item.index * 361 + tick).normal(0, .1)
                    measurement = {'rack_index': item.index, 'temperature_c': float(arrays['rack_temperature_c'][tick, item.index] + noise)}
                    alarm = measurement['temperature_c'] > 65
                elif item.kind == 'electrical':
                    measurement = {'pod_index': item.index, 'contact_temperature_c': float(arrays['contact_temperature_c'][tick, item.index])}
                    alarm = measurement['contact_temperature_c'] > 60
                elif item.kind == 'cooling':
                    measurement = {'pod_index': item.index, 'pod_flow_kg_s': float(arrays['rack_flow_kg_s'][tick].reshape(16, 8)[item.index].sum())}
                    alarm = measurement['pod_flow_kg_s'] < 33.6
                else:
                    error = arrays['public_sensor_temperature_c'][tick] - arrays['rack_temperature_c'][tick]
                    measurement = {'pod_index': item.index, 'controller_sensor_error_c': float(error.reshape(16, 8)[item.index].mean())}
                    alarm = abs(measurement['controller_sensor_error_c']) > 1
                if not alarm:
                    continue
                row = self.rows[matching[0]]
                cued = candidate['cue_strength'] >= .05
                uncued_alarm = cued and not candidate['known_measured_alarm']
                witness = {'dispatch_time_s': g['time_s'], 'agent': g['agent'], 'position_m': g['position_m'],
                    'candidate': candidate, 'completion_measurement_evaluator_only': measurement}
                row['witness_count'] += 1;row['any_cue_witness_count'] += int(cued)
                row['unconfirmed_cue_witness_count'] += int(uncued_alarm)
                for key, allowed in [('first_witness', True), ('first_unconfirmed_cue_witness', uncued_alarm)]:
                    if allowed and (row[key] is None or (g['time_s'], g['agent']) < (row[key]['dispatch_time_s'], row[key]['agent'])):
                        row[key] = witness

    def finish(self):
        for row, interval in zip(self.rows, self.intervals):
            row['baseline_detected'] = interval['detected']
            row['baseline_first_confirmation_s'] = interval['first_confirmation_s']
            row['opportunity_class'] = ('one_detour_with_prespecified_unconfirmed_cue' if row['unconfirmed_cue_witness_count'] else
                'one_detour_without_prespecified_unconfirmed_cue' if row['witness_count'] else 'no_one_detour_witness_on_original_tour_states')
        return {'scope': 'Evaluator-only one-detour witnesses on original tour states; not an optimal schedule or global bound',
            'scanned_free_agent_frames': self.free, 'scanned_legal_candidates': self.candidates, 'segments': self.rows}


def main():
    global ACTIVE_PROBE
    p = argparse.ArgumentParser()
    for key in ('results', 'actors', 'estimator-weights', 'estimator-manifest', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    summary = json.loads((args.results / 'summary.json').read_text())
    index = json.loads((args.results / 'case-index.json').read_text())
    branches = json.loads((args.results / 'branch-index.json').read_text())
    process_index = json.loads((args.results / 'process-index.json').read_text())
    actors, fingerprints = load_actors(args.actors)
    predictor = frozen_predictor(args.estimator_weights, args.estimator_manifest)
    scorer = actor_scorer(actors[0])
    main_rows = [r for r in index if r['counterfactual_source'] is None]
    need([r['specification'] for r in main_rows] == specs(summary['phase']), 'All frozen main cases in declared order')
    need(summary['main_cases'] == len(main_rows) and summary['evaluator_branches'] == len(branches), 'Scope accounting')
    verified_probes, baseline_records, checked_processes = {}, {}, set()
    expected_branches = {}
    dispatches, measurements, certificates = 0, 0, 0
    maximum_ratio, max_energy, max_electric = 0., 0., 0.
    runner.snapshot, runner.choose = checked_snapshot, checked_choose
    try:
        for number, row in enumerate(index):
            path = args.results / row['path']
            need(sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Raw case bytes')
            original = json.loads(gzip.decompress(path.read_bytes()))
            source = row['counterfactual_source']
            if source is None:
                probe_on = row['specification']['controller'] == 'tour'
                ACTIVE_PROBE = IndependentProbe(original['anomaly_intervals']) if probe_on else None
                replay, env = runner.run_public(row['specification'], predictor, scorer,
                    row['specification']['controller'], probe_enabled=probe_on)
            else:
                ACTIVE_PROBE = None
                need(row['case'] in expected_branches, 'Only predeclared first witnesses may create branches')
                equivalent(source, expected_branches[row['case']], 'Counterfactual source and exact one-action rule')
                replay, env = runner.run_evaluator_branch(row['specification'], predictor, scorer, source['intervention'])
            equivalent(replay, original, 'Actual public control or one-action counterfactual replay')
            independent_schedule(original, env)
            independent_events(original, env)
            passed, count = independent_certificates(original, env)
            independent_power_events(env, original)
            independent_public_and_quota(original, env)
            dispatches += count;measurements += len(original['inspection_events']);certificates += passed
            maximum_ratio = max(maximum_ratio, original['metrics']['maximum_exact_gap_ratio'])
            if ACTIVE_PROBE is not None:
                probe = ACTIVE_PROBE.finish()
                equivalent(probe, original['evaluator_probe'], 'Independent full finite candidate enumeration and cue classification')
                verified_probes[row['case']] = probe
                baseline_records[row['case']] = original
                for segment in probe['segments']:
                    if segment['baseline_detected'] or segment['first_witness'] is None:
                        continue
                    witness = segment['first_witness']
                    branch_case = row['case'] + '--branch--' + str(segment['segment_index'])
                    expected_branches[branch_case] = {'baseline_case': row['case'], 'segment_index': segment['segment_index'],
                        'intervention': {'dispatch_time_s': witness['dispatch_time_s'], 'agent': witness['agent'], 'task': segment['task']},
                        'scope': 'One independent evaluator-only action; not a deployable policy or combined rescue count'}
            elif source is not None:
                base = baseline_records[source['baseline_case']]
                segment = source['segment_index']
                need(not base['anomaly_intervals'][segment]['detected'] and original['anomaly_intervals'][segment]['detected'], 'Targeted missed segment actually confirmed in branch')
                override = source['intervention']
                for old, new in zip(base['action_log'], original['action_log']):
                    if old['time_s'] < override['dispatch_time_s']:
                        equivalent(old, new, 'No pre-intervention action changes')
                    elif old['time_s'] == override['dispatch_time_s']:
                        for agent, target in old['choices'].items():
                            need(new['choices'][agent] == (override['task'] if int(agent) == override['agent'] else target), 'Only one action differs at intervention boundary')
                        break
                need(sum(d['tool'] == 'evaluator_one_detour' for d in original['workflow_decisions']) == 1, 'Exactly one explicit override')
                branch = next(r for r in branches if r['case'] == row['case'])
                equivalent(branch['baseline_intervals'], base['anomaly_intervals'])
                equivalent(branch['branch_intervals'], original['anomaly_intervals'])
                need(branch['target_detected'] and branch['total_missed_delta'] == original['metrics']['missed_segments'] - base['metrics']['missed_segments'], 'Branch target and collateral miss counts')
                need(abs(branch['travel_delta_m'] - original['metrics']['travel_m'] + base['metrics']['travel_m']) < 1e-7, 'Actual branch travel change')
                need(abs(branch['service_delta_s'] - original['metrics']['service_s'] + base['metrics']['service_s']) < 1e-7, 'Actual branch service change')
            if row['process_key'] not in checked_processes:
                info = process_index[row['process_key']];source_path = args.results / info['path']
                need(sha(source_path) == info['sha256'], 'Physical array file bytes')
                with np.load(source_path, allow_pickle=False) as pack:
                    for key in pack.files:
                        np.testing.assert_allclose(pack[key], env.trace.arrays[key], rtol=1e-11, atol=1e-9)
                    c, scalar = env.config, pack['process_scalars']
                    energy = (c.chip_capacity_kj_k * pack['rack_temperature_c'].sum(1) +
                        c.secondary_capacity_kj_k * pack['secondary_supply_c'].sum(1) +
                        c.primary_capacity_per_pod_kj_k * c.pods * pack['primary_supply_c'] +
                        c.contact_capacity_kj_k * pack['contact_temperature_c'].sum(1))
                    residual = energy - np.r_[env.trace.metadata['initial_energy_kj'], energy[:-1]] - 5 * (scalar[:, 2] - scalar[:, 4])
                    electric = scalar[:, 2] - scalar[:, 5] - scalar[:, 6] - scalar[:, 7] - scalar[:, 8]
                    need(abs(residual).max() < 1e-4 and abs(electric).max() < 1e-6, 'Independent physical balances')
                    max_energy = max(max_energy, float(abs(residual).max()));max_electric = max(max_electric, float(abs(electric).max()))
                checked_processes.add(row['process_key'])
            if number % 16 == 0:
                print('C1A_AUDIT', number + 1, '/', len(index), flush=True)
    finally:
        runner.snapshot, runner.choose, ACTIVE_PROBE = ORIGINAL_SNAPSHOT, ORIGINAL_CHOOSE, None
    need({r['case'] for r in index if r['counterfactual_source'] is not None} == set(expected_branches), 'No missing or added counterfactual branches')
    need({r['case'] for r in branches} == set(expected_branches), 'Branch catalogue exactly matches first witnesses')
    for seed, model in actors.items():
        need(seed_digest(model) == fingerprints[str(seed)]['parameters_sha256'], 'Actor parameters unchanged')
    result = {'all_actual_cases_replayed': len(index), 'main_cases_replayed': len(main_rows),
        'evaluator_one_detour_branches_replayed': len(expected_branches), 'physical_arrays_verified': len(checked_processes),
        'actual_dispatches': dispatches, 'completed_measurements': measurements, 'complete_return_plans_checked': certificates,
        'all_public_legal_candidates_independently_rebuilt': CANDIDATES_CHECKED,
        'public_controller_choices_independently_checked': CHOICES_CHECKED,
        'all_finite_probe_counts_first_witnesses_and_cues_independently_rebuilt': True,
        'only_one_declared_action_changed_and_target_confirmed_in_each_branch': True,
        'maximum_exact_gap_ratio': maximum_ratio,
        'maximum_independent_energy_residual_kj': max_energy, 'maximum_independent_electric_residual_kw': max_electric,
        'scope': 'Local one-detour diagnosis on original tour states; no global optimal bound, no proof of unavoidable misses and no deployable future-label policy'}
    with args.output.open('x') as f:
        json.dump(result, f, indent=2);f.write('\n')
    print('C1A_FULL_AUDIT_PASSED', len(index), flush=True)


if __name__ == '__main__':
    main()
