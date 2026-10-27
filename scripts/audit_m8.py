"""M8 factorial budgets, validation selection and full physical replay."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from patrol_rsi.deadline import measured_mean_estimate
from patrol_rsi.m6 import rank
from patrol_rsi.m8 import VARIANTS, average, contract, training_case
from patrol_rsi.relations import RelationDispatchEnv
from patrol_rsi.m7 import PLANNER_SETTINGS
from patrol_rsi.stress import PairCyclePlanner, planner_choices


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(directory):
    raw = (directory / 'summary.json.gz').read_bytes()
    original = gzip.decompress(raw)
    fingerprint = json.loads((directory / 'summary-fingerprint.json').read_text())
    require(len(raw) == fingerprint['stored_bytes'] and hashlib.sha256(raw).hexdigest() == fingerprint['stored_sha256'], 'Stored bytes fingerprint')
    require(len(original) == fingerprint['uncompressed_bytes'] and hashlib.sha256(original).hexdigest() == fingerprint['uncompressed_sha256'], 'Uncompressed fingerprint')
    return json.loads(original)


def replay(row, cache):
    case = {key: row[key] for key in ('split', 'family', 'layout_seed', 'field_seed', 'profile', 'noise')}
    method = row['method']
    use_vae = method in ('learned', 'untrained_network') or PLANNER_SETTINGS[method][1]
    key = json.dumps(case, sort_keys=True) + str(use_vae) + json.dumps([(event['choices'], event['target_estimates']) for event in row['events']])
    fields = ['target_coverage', 'deadline_violation', 'late_deadline_violation', 'weighted_idleness', 'public_cost',
              'peak_age', 'movement_distance', 'target_prediction_mse', 'target_assignments', 'duplicate_assignment_fraction',
              'first_all_seen_step', 'target_ages', 'positions', 'target_deadlines', 'prediction_mse_series', 'weighted_idleness_series']
    signature_data = {name: row[name] for name in fields}
    signature_data['events'] = [{name: value for name, value in event.items() if name != 'behavior_logp'} for event in row['events']]
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    require(len(row['events']) == 96 and len(row['positions']) == len(row['target_ages']) == 97, 'Incomplete trace')
    if key in cache and method not in PLANNER_SETTINGS:
        require(cache[key] == signature, 'Same public controls produce differing metrics')
        return
    wrapper = RelationDispatchEnv(case)
    env = wrapper.env
    pair_planner = PairCyclePlanner(wrapper) if method == 'pair_cycle' else None
    idle, mse, rewards, violations = [], [], [], []
    first_seen = 0 if env.ever_measured[env.scenario.targets].all() else None
    require(np.array_equal(env.positions, row['positions'][0]) and np.array_equal(env.raw_age[env.scenario.targets], row['target_ages'][0]), 'Initial state')
    for step, event in enumerate(row['events']):
        estimate = np.asarray(event['target_estimates'], dtype=np.float32)
        require(estimate.shape == (8,) and np.isfinite(estimate).all() and ((estimate >= 0) & (estimate <= 1)).all(), 'Bad public estimate')
        predicted = np.full_like(env.values, .5)
        predicted[env.scenario.targets] = estimate
        env.set_estimate(predicted)
        require(np.array_equal(env.estimate[env.scenario.targets], estimate), 'Measured estimate override')
        if not use_vae:
            require(np.array_equal(estimate, measured_mean_estimate(env)[env.scenario.targets]), 'Measurement-hold estimate')
        state = wrapper.prepare()
        require(state['free'].tolist() == event['free_executors'] and state['order'].tolist() == event['order'], 'Commitment/order replay')
        choice = np.asarray(event['choices'], dtype=int)
        if method in PLANNER_SETTINGS:
            expected = planner_choices(state, PLANNER_SETTINGS[method][0], pair_planner)
            require(np.array_equal(choice, expected), 'Stated planner decision')
        mse.append(env.evaluation()['reconstruction_mse'])
        reward, repeated = wrapper.step(choice, state, True)
        for name, value in repeated.items():
            require(np.isclose(value, event[name]) if isinstance(value, float) else value == event[name], f'Physical event {step}/{name}')
        require(np.array_equal(env.positions, row['positions'][step + 1]), 'Position trace')
        require(np.array_equal(env.raw_age[env.scenario.targets], row['target_ages'][step + 1]), 'Age trace')
        idle.append(env.evaluation()['normalized_weighted_idleness'])
        rewards.append(reward)
        violations.append(event['deadline_violation'])
        if first_seen is None and env.ever_measured[env.scenario.targets].all():
            first_seen = step + 1
    metrics = {'target_coverage': env.evaluation()['target_coverage'], 'deadline_violation': np.mean(violations),
               'late_deadline_violation': np.mean(violations[32:]), 'weighted_idleness': np.mean(idle),
               'public_cost': -np.mean(rewards), 'peak_age': np.max(row['target_ages']),
               'movement_distance': env.evaluation()['total_movement_distance'], 'target_prediction_mse': np.mean(mse),
               'target_assignments': wrapper.assignments, 'duplicate_assignment_fraction': 0,
               'first_all_seen_step': first_seen if first_seen is not None else 97}
    for name, value in metrics.items():
        require(np.isclose(row[name], value), f'Recomputed {name}')
    require(row['first_all_seen_censored'] == (first_seen is None), 'Coverage censoring')
    require(row['target_deadlines'] == wrapper.deadlines.tolist(), 'Deadline vector')
    require(np.allclose(row['prediction_mse_series'], mse) and np.allclose(row['weighted_idleness_series'], idle), 'Metric series')
    for name, value in wrapper.initial_access_bounds().items():
        require(row[name] == value, 'Initial bound')
    require(row['deadline_violation'] + 1e-12 >= row['startup_violation_lower_bound'], 'Impossible startup bound')
    require(row['estimator_calls'] == (96 if use_vae else 0), 'Estimator budget')
    if pair_planner:
        require(row['pair_partition'] == [list(pair) for pair in pair_planner.pairs], 'Partition')
        require(np.isclose(row['pair_proxy_objective'], pair_planner.proxy_objective), 'Pair proxy')
        require(np.isclose(row['cross_pair_assignment_fraction'], pair_planner.cross_pair_assignments / max(1, pair_planner.assignments)), 'Pair fallback')
    cache[key] = signature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configuration = contract()
    initial, schedule = {}, {}
    tasks, cache, count = [], {}, 0
    for variant in VARIANTS:
        for seed in range(3):
            directory = args.results / f'{variant}-seed-{seed}'
            summary = read(directory)
            require(json.loads((directory / 'contract.json').read_text()) == configuration, 'Frozen contract')
            require(summary['variant'] == variant and summary['seed'] == seed and not summary['preflight'] and not summary['smoke'], 'Training provenance')
            require(summary['episodes'] == 1200 and summary['gradient_updates'] == 3600 and summary['parameters'] == 22594, 'Matched budget/capacity')
            require(summary['behavior_logp_max_error'] <= 2e-5 and summary['fresh_checkpoint_trace_replay_matches'] and not summary['test_used_for_selection'], 'Likelihood/reload/selection gate')
            require(summary['initial_parameter_sha256'] != summary['selected_parameter_sha256'], 'No parameter update')
            initial.setdefault(seed, summary['initial_parameter_sha256'])
            require(initial[seed] == summary['initial_parameter_sha256'], 'Paired initializations')
            rng = np.random.default_rng(seed)
            expected_schedule = hashlib.sha256()
            for episode in range(1200):
                _, row = training_case(configuration, variant, seed, episode, rng)
                expected_schedule.update(json.dumps(row).encode() + b'\n')
            require(expected_schedule.hexdigest() == summary['schedule_sha256'], 'Actual schedule fingerprint')
            schedule.setdefault(seed, summary['schedule_sha256'])
            require(schedule[seed] == summary['schedule_sha256'], 'Paired index/field/profile schedules')
            curve = [json.loads(line) for line in (directory / 'learning.jsonl').read_text().splitlines()]
            require([row['episode'] for row in curve] == list(range(100, 1201, 100)), 'Complete curve')
            selected = min(curve, key=lambda row: rank(row['validation']))
            require(selected['episode'] == summary['selected_episode'] and selected['validation'] == summary['best_validation'], 'Validation-only chosen checkpoint')
            require(set(summary['test']) == {'learned', 'untrained_network'}, 'All controls')
            for method, group in summary['test'].items():
                require(len(group['cases']) == 96, 'All final conditions')
                for case, row in zip(configuration['test'], group['cases']):
                    require(all(row[key] == value for key, value in case.items()) and row['method'] == method and row['variant'] == variant, 'Final case identity')
                    replay(row, cache)
                    count += 1
                for key, value in average(group['cases']).items():
                    require(np.isclose(value, group['metrics'][key]), 'Method mean')
            tasks.append({'variant': variant, 'seed': seed, 'selected_episode': summary['selected_episode'],
                          'validation_selection': True, 'paired_initial_and_schedule': True, 'records': 192})
    common = args.results / 'common-planners'
    planners = read(common)
    require(planners['planners'] and not planners['training'] and set(planners['test']) == set(PLANNER_SETTINGS), 'Common planner controls')
    require(json.loads((common / 'contract.json').read_text()) == configuration, 'Planner contract')
    for method, group in planners['test'].items():
        require(len(group['cases']) == 96, 'Planner conditions')
        for case, row in zip(configuration['test'], group['cases']):
            require(all(row[key] == value for key, value in case.items()) and row['method'] == method, 'Planner case identity')
            replay(row, cache)
            count += 1
        for key, value in average(group['cases']).items():
            require(np.isclose(value, group['metrics'][key]), 'Planner mean')
    result = {'from_scratch_training_tasks': 12, 'paired_training_seeds_per_group': 3,
              'common_planner_task': 1, 'method_condition_records_replayed': count,
              'distinct_case_choices_and_estimates': len(cache), 'parameters_each': 22594,
              'team_training_microsteps': 12 * 1200 * 96, 'gradient_minibatches': 12 * 3600,
              'paired_initializations_and_index_field_profile_schedules': True, 'validation_only_selection': True,
              'actual_weights_checked_separately': True, 'tasks': tasks,
              'scope': 'Complete physical/public reward/metric and planner replay; unmeasured VAE predictions accepted as recorded; actual-policy first-case replay checked separately'}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
    print('M8_FULL_AUDIT', count, 'records;', len(cache), 'distinct public controls', flush=True)


if __name__ == '__main__':
    main()
