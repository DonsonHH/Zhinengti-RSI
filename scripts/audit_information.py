"""Reconstruct archived Tour observations and audit factorized candidates."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from patrol_rsi.causal_observation import ObservedTask, capture_public
from patrol_rsi.improver_task import PatrolProgram
from patrol_rsi.information_audit import (KINDS, SOURCE_NAMES, PublicSignals,
    analytic_continuations, audited_best, current_legal_geometry, pack_geometry,
    reference_signals, rescore_candidates, score_candidates)
from patrol_rsi.m11 import frozen_predictor
from patrol_rsi.m9 import arrays_digest
from patrol_rsi.public_stopping import StopConfig, public_frontier


BINS = ('0-30', '30-60', '60-120', '120-300', '300-600', '600-1800')


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def same(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(v, b[k]) for k, v in a.items())
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return isinstance(b, (int, float)) and np.isclose(a, b, rtol=1e-10, atol=1e-9)
    return a == b


def coordinate(row):
    spec = row['specification']
    return tuple(spec[k] for k in ('seed', 'scenario', 'duration_s', 'fault_start_s', 'noise_c'))


class ErrorStatistics:
    def __init__(self):
        self.groups = {}

    def add(self, label, predicted, actual, mask=None):
        good = np.isfinite(predicted) & np.isfinite(actual)
        if mask is not None:
            good &= mask
        p, a = predicted[good], actual[good]
        if not len(p):
            return
        row = self.groups.setdefault(label, {'n': 0, 'sum_error': 0., 'sum_absolute_error': 0.,
            'sum_squared_error': 0., 'predicted_positive': 0, 'reference_positive': 0,
            'true_positive': 0, 'false_positive': 0, 'false_negative': 0})
        error = p - a
        row['n'] += len(p)
        row['sum_error'] += float(error.sum())
        row['sum_absolute_error'] += float(np.abs(error).sum())
        row['sum_squared_error'] += float(np.square(error).sum())
        for key, values in [('predicted_positive', p > 0), ('reference_positive', a > 0),
                            ('true_positive', (p > 0) & (a > 0)),
                            ('false_positive', (p > 0) & (a <= 0)),
                            ('false_negative', (p <= 0) & (a > 0))]:
            row[key] += int(values.sum())


def geometry_key(geometry, i):
    return (int(geometry['dispatch_time_s'][i]), int(geometry['agent'][i]), int(geometry['task'][i]))


def rebuild_tour(record, predictor, arrays):
    """Forced saved actions; old physical reconstruction must equal all arrays."""
    env = ObservedTask(record['specification'], PatrolProgram(), predictor)
    need(set(env.trace.arrays) == set(arrays) and all(np.array_equal(env.trace.arrays[k], arrays[k]) for k in arrays),
         'Old physical reconstruction differs from archived process')
    bundles, geometry_rows, stamps = [], [], {}
    for action in record['action_log']:
        need(env.clock == action['time_s'], 'Recorded replay time differs')
        bundle = capture_public(env)
        bundles.append(bundle)
        geometry_rows.extend(current_legal_geometry(bundle))
        stamps[env.clock] = hashlib.sha256(json.dumps(bundle, sort_keys=True).encode()).hexdigest()
        env.step({int(k): v for k, v in action['choices'].items()})
    need(env.events == record['inspection_events'] and env.assignments == record['assignments'],
         'Forced original Tour measurements or assignments differ')
    need(len(bundles) == 359 and env.estimator_calls == record['metrics']['estimator_calls'], 'Incomplete history reconstruction')
    return bundles, pack_geometry(geometry_rows), stamps


def bind_prefix(record, tour, bundles, stamps):
    extra = next((v for v in record['workflow_decisions'] if v['task'] != v['baseline_next']), None)
    stop = extra['time_s'] if extra else 1800.
    for action, expected in zip(record['action_log'], tour['action_log']):
        if action['time_s'] < stop:
            need(action == expected, 'Source did not share the common unspent-budget Tour prefix')
    need([v for v in record['inspection_events'] if v['time_s'] <= stop] ==
         [v for v in tour['inspection_events'] if v['time_s'] <= stop], 'Common prefix measurements differ')
    count = 0
    for row in record['causal_snapshot_stamps']:
        need(row['time_s'] <= stop and stamps[row['time_s']] == row['sha256'], 'Archived public snapshot SHA differs')
        count += 1
    for snapshot in record['selected_public_snapshots']:
        need(snapshot == bundles[int(snapshot['time_s'] / 5) - 1], 'Original selected full public state differs')
    return extra, count


def validate_executed_branch(row, record, tour, geometry, oracle, lookup):
    action = next((v for v in record['workflow_decisions'] if v['task'] != v['baseline_next']), None)
    if not action:
        need(row['metrics']['missed_segments'] == tour['metrics']['missed_segments'], 'No-check outcomes differ from Tour')
        return None
    key = (int(action['time_s']), action['agent'], action['task'])
    need(key in lookup, 'Saved actual optional action missing from full legal universe')
    i = lookup[key]
    need(int(oracle['missed_delta'][i]) == record['metrics']['missed_segments'] - tour['metrics']['missed_segments'],
         'Analytic whole-mission miss effect differs from actual original branch')
    if tour['anomaly_intervals']:
        observed = record['metrics']['consequence_weighted_censored_delay_s'] - tour['metrics']['consequence_weighted_censored_delay_s']
        need(np.isclose(oracle['weighted_wait_delta_s'][i], observed, rtol=1e-10, atol=1e-8),
             'Analytic whole-mission delay differs from actual original branch')
    actual_first = [v['first_confirmation_s'] if v['detected'] else -1 for v in record['anomaly_intervals']]
    need(oracle['first_confirmation_s'][i].tolist() == actual_first, 'Analytic event-level confirmations differ')
    return {'case': row['case'], 'geometry_index': i, 'action': key,
            'missed_delta': int(oracle['missed_delta'][i]),
            'weighted_wait_delta_s': float(oracle['weighted_wait_delta_s'][i]) if tour['anomaly_intervals'] else None}


def analyze_condition(rows, records, folder, process, output, predictor, shared_geometry):
    tour_row = next(v for v in rows if v['alias'] == 'tour')
    tour = records[tour_row['case']]
    with np.load(folder / process['path']) as data:
        arrays = {k: data[k].copy() for k in data.files}
    need(sha(folder / process['path']) == process['sha256'], 'Original process SHA differs')
    started = time.perf_counter()
    bundles, geometry, stamps = rebuild_tour(tour, predictor, arrays)
    if shared_geometry is not None:
        need(set(geometry) == set(shared_geometry) and all(np.array_equal(geometry[k], shared_geometry[k]) for k in geometry),
             'Common known geometry is not common across original conditions')
    else:
        np.savez_compressed(output.parent / 'geometry.npz', **geometry)
    output.mkdir(exist_ok=False)
    with gzip.open(output / 'public-states.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for bundle in bundles:
            stream.write(json.dumps(bundle, allow_nan=False) + '\n')
    ref = reference_signals(arrays)
    oracle = analytic_continuations(tour, geometry, ref)
    np.savez_compressed(output / 'analytic-continuations.npz', **oracle)
    lookup = {geometry_key(geometry, i): i for i in range(len(geometry['task']))}
    source_stamps = branches = 0
    checks = []
    revised = []
    source_rows = []
    for row in rows:
        record = records[row['case']]
        extra, count = bind_prefix(record, tour, bundles, stamps)
        source_stamps += count
        check = validate_executed_branch(row, record, tour, geometry, oracle, lookup)
        if check:
            checks.append(check)
            branches += 1
        if row['definition']['method'] == 'revised':
            revised.append((row, StopConfig(**row['definition']['config']), record))
        source_rows.append({'case': row['case'], 'source_sha256': row['sha256'], 'source_bytes': row['bytes'],
                            'common_prefix_until_s': extra['time_s'] if extra else 1800.,
                            'archived_public_SHA_frames_verified': count})
    statistics = ErrorStatistics()
    layer_counts = defaultdict(Counter)
    endpoint_sources = Counter()
    forecasts = 0
    weak_history_roots = unique_history_roots = 0
    winner_checks = 0
    frame_path = output / 'revised-frame-diagnostics.jsonl.gz'
    with gzip.open(frame_path, 'wt', encoding='utf-8') as frames:
        for root_index, bundle in enumerate(bundles):
            public = PublicSignals(bundle)
            base_values = score_candidates(public, geometry)
            index = base_values['geometry_index']
            target = geometry['task'][index]
            delta = geometry['finish_s'][index] - bundle['time_s']
            truth = ref[np.minimum(359, (geometry['finish_s'][index] / 5).astype(int) - 1), target]
            signal = base_values['target_signal']
            forecasts += len(index)
            bins = np.digitize(delta, [30, 60, 120, 300, 600], right=True)
            for kind, name in enumerate(KINDS):
                for bin_index, bin_name in enumerate(BINS):
                    statistics.add('all_legal/' + name + '/' + bin_name, signal, truth,
                                   (public.kind[target] == kind) & (bins == bin_index))
            thermal = public.kind[target] == 0
            pieces = public.thermal_components(target, delta)
            for source, temperature in pieces.items():
                predicted = np.maximum(temperature - 65., 0.) / 20
                for bin_index, bin_name in enumerate(BINS):
                    statistics.add('thermal_source/' + source + '/' + bin_name, predicted, truth,
                                   thermal & (bins == bin_index))
            endpoint_sources.update(SOURCE_NAMES[int(v)] for v in base_values['target_source'])
            # Monotone bounded trend + two held channels: checking endpoints
            # suffices for whether held history can uniquely increase any signal.
            ids = np.arange(128)
            possible_change = False
            for endpoint in (0., 1800. - bundle['time_s']):
                full, _ = public.at(ids, endpoint)
                removed, _ = public.at(ids, endpoint, drop_history=True)
                if np.any((full > 0) & ((~np.isfinite(removed)) | (full > removed + 1e-12))):
                    possible_change = True
            unique_history_roots += int(possible_change)
            weak_history_roots += int(not possible_change)
            for row, config, original in revised:
                before = original['stopping_decisions'][root_index]
                if before['action'] == 'TOUR':
                    continue
                values = rescore_candidates(bundle, geometry, base_values, config)
                current_id, current = audited_best(bundle, geometry, values, True, config)
                later_id, later = audited_best(bundle, geometry, values, False, config)
                qcheck = current['net_score'] if current else 0.
                qwait = later['net_score'] if later else 0.
                choose = current is not None and qcheck >= qwait - 1e-12
                need(before['action'] == ('CHECK' if choose else 'WAIT') and
                     same(before['q_check'], qcheck) and same(before['q_wait'], qwait), 'Frozen actual CHECK/WAIT decision differs')
                need(int(values['positive_signal_eligible'].sum()) == before['candidate_count'], 'Full positive frontier count differs')
                need(same(before.get('future_selected'), later), 'Saved best future score/provenance differs')
                if choose:
                    need(same(before['selected'], current), 'Saved CHECK score/provenance differs')
                # On fixed roots, also compare full original positive frontier
                # identity and value, rather than relying on winner agreement.
                if root_index in (0, 65, 98, 149, 269, 358) and row['alias'] in ('C0',):
                    old = public_frontier(bundle, config)
                    own_keys = {geometry_key(geometry, int(v)) for v in index[values['positive_signal_eligible']]}
                    need(own_keys == {(int(v['dispatch_time_s']), v['agent'], v['task']) for v in old}, 'Full original future frontier membership differs')
                    by_key = {(int(v['dispatch_time_s']), v['agent'], v['task']): v for v in old}
                    for key, point in by_key.items():
                        position = np.flatnonzero(index == lookup[key])[0]
                        need(np.isclose(values['net_score'][position], point['net_score'], rtol=1e-10, atol=1e-9), 'Nonwinner candidate score differs')
                    winner_checks += len(old)
                current_mask = values['current']
                helpful = oracle['helpful'][index] & current_mask
                any_helpful = bool(helpful.any())
                considered = helpful & values['positive_signal_eligible']
                retained = considered & values['admissible']
                if not any_helpful:
                    layer = 'no_net_helpful_current_action'
                elif not considered.any():
                    layer = 'helpful_current_lacks_positive_available_unalarmed_signal'
                elif not retained.any():
                    layer = 'helpful_current_rejected_by_cost_threshold_or_veto'
                elif current_id is not None and not oracle['helpful'][current_id]:
                    layer = 'current_proxy_ranks_nonhelpful_action'
                elif choose:
                    layer = 'CHECK_executes_net_helpful_action'
                else:
                    layer = 'WAIT_defers_net_helpful_current_action'
                layer_counts[row['alias']][layer] += 1
                current_truth = int(oracle['missed_delta'][current_id]) if current_id is not None else None
                future_truth = int(oracle['missed_delta'][later_id]) if later_id is not None else None
                frame = {'case': row['case'], 'alias': row['alias'], 'time_s': bundle['time_s'],
                    'actual_action': before['action'], 'episode_never_checked': original['metrics']['selected_detours'] == 0,
                    'q_check': qcheck, 'q_wait': qwait, 'current_geometry_index': current_id,
                    'future_geometry_index': later_id, 'net_helpful_current_exists': any_helpful,
                    'best_current_missed_delta': current_truth, 'best_future_missed_delta': future_truth,
                    'best_current_wait_delta_s': float(oracle['weighted_wait_delta_s'][current_id]) if current_id is not None and tour['anomaly_intervals'] else None,
                    'best_future_wait_delta_s': float(oracle['weighted_wait_delta_s'][later_id]) if later_id is not None and tour['anomaly_intervals'] else None,
                    'layer': layer, 'helpful_current_count': int(helpful.sum()),
                    'helpful_positive_signal_count': int(considered.sum()),
                    'helpful_admissible_count': int(retained.sum()),
                    'helpful_missing_signal_count': int((helpful & ~np.isfinite(signal)).sum()),
                    'helpful_zero_signal_count': int((helpful & np.isfinite(signal) & (signal <= 0)).sum()),
                    'helpful_known_alarm_excluded_count': int((helpful & public.alarm[target]).sum()),
                    'helpful_vetoed_count': int((considered & values['tour_slack_veto']).sum()),
                    'helpful_below_threshold_count': int((considered & (values['net_score'] <= config.threshold)).sum()),
                    'future_selected_predicted_signal': later['target_signal'] if later else None,
                    'future_selected_reference_signal': float(ref[min(359, int(later['finish_s'] / 5) - 1), later['task']]) if later else None,
                    'hard_legal_remaining_candidates': len(index),
                    'old_positive_signal_frontier_count': int(values['positive_signal_eligible'].sum())}
                frames.write(json.dumps(frame, allow_nan=False) + '\n')
                if later_id is not None:
                    p = int(np.flatnonzero(index == later_id)[0])
                    statistics.add('selected_future/' + row['alias'], signal[[p]], truth[[p]])
    oracle_summary = {'hard_legal_dispatch_candidates': len(geometry['task']),
        'net_helpful_candidates': int(oracle['helpful'].sum()), 'net_harmful_candidates': int(oracle['harmful'].sum()),
        'minimum_missed_delta': int(oracle['missed_delta'].min()) if len(oracle['missed_delta']) else 0,
        'executed_original_branches_verified': branches, 'executed_branch_checks': checks}
    summary = {'condition': dict(tour_row['specification']), 'source_cases': source_rows,
        'common_Tour_history_frames': len(bundles), 'original_public_SHA_frames_verified': source_stamps,
        'factorized_candidate_root_pairs': forecasts, 'positive_frontier_nonwinner_scores_checked': winner_checks,
        'error_statistics': statistics.groups, 'actual_revised_decision_layers': {k: dict(v) for k, v in layer_counts.items()},
        'all_candidate_winner_source_counts': dict(endpoint_sources),
        'history_channel_can_uniquely_increase_positive_signal_roots': unique_history_roots,
        'history_channel_endpoint_no_positive_influence_proofs': weak_history_roots,
        'oracle': oracle_summary, 'geometry_numeric_digest': arrays_digest(geometry),
        'wall_s': time.perf_counter() - started,
        'scope': 'All hard-legal current/future Tour dispatch candidates; exact factorized public states and geometry. Evaluator-only one-CHECK-then-Tour analytic outcomes, validated against every saved executed detour. No new policy selection or trained predictors.'}
    write(output / 'summary.json', summary)
    return geometry, summary


def main():
    parser = argparse.ArgumentParser()
    for name in ('formal-root', 'output', 'estimator-weights', 'estimator-manifest'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--folder', required=True)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(exist_ok=False)
    folder = args.formal_root / args.folder
    index = read(folder / 'case-index.json')
    processes = read(folder / 'process-index.json')
    cases = [row for row in index if row['alias'] == 'tour']
    if args.pilot:
        need(args.folder == 'development-0', 'Pilot uses two declared old development conditions')
        cases = [cases[i] for i in (0, 14)]
    conditions = {coordinate(v) for v in cases}
    source = [v for v in index if coordinate(v) in conditions]
    records = {}
    for row in source:
        path = folder / row['path']
        need(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Original case SHA/size differs')
        records[row['case']] = read(path)
    predictor = frozen_predictor(args.estimator_weights, args.estimator_manifest)
    frozen = {'original_folder': args.folder, 'original_case_index_sha256': sha(folder / 'case-index.json'),
              'original_process_index_sha256': sha(folder / 'process-index.json'),
              'CVAE_sha256': sha(args.estimator_weights), 'pilot': args.pilot,
              'old_observation_conditions': [v['specification'] for v in cases],
              'new_physical_instances': 0, 'new_policy_trials': 0, 'neural_updates': 0, 'LLM_calls': 0}
    write(args.output / 'input-binding.json', frozen)
    geometry, summaries = None, []
    for number, row in enumerate(cases):
        own = [v for v in source if coordinate(v) == coordinate(row)]
        geometry, summary = analyze_condition(own, records, folder, processes[row['process_key']],
            args.output / f'condition-{number:02}', predictor, geometry)
        summaries.append(summary)
        print('INFORMATION_CONDITION_DONE', args.folder, number, row['case'], summary['wall_s'], flush=True)
    write(args.output / 'summary.json', {'folder': args.folder, 'conditions': len(cases),
        'source_cases': len(source), 'geometry_candidates_per_condition': len(geometry['task']),
        'total_factorized_candidate_root_pairs': sum(v['factorized_candidate_root_pairs'] for v in summaries),
        'original_public_SHA_frames_verified': sum(v['original_public_SHA_frames_verified'] for v in summaries),
        'executed_original_branches_verified': sum(v['oracle']['executed_original_branches_verified'] for v in summaries),
        'new_physical_instances': 0, 'new_policy_trials': 0, 'neural_updates': 0, 'LLM_calls': 0})
    print('INFORMATION_AUDIT_DONE', args.folder, len(cases), len(source), flush=True)


if __name__ == '__main__':
    main()
