"""Recompute M7 public decisions and physical traces, independent of summaries."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from patrol_rsi.deadline import measured_mean_estimate
from patrol_rsi.m6 import METRICS
from patrol_rsi.m7 import EXTRA_METRICS, LEARNED_SETTINGS, PLANNER_SETTINGS, contract
from patrol_rsi.stress import PairCyclePlanner, StressDispatchEnv, planner_choices


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def tasks():
    return [(name, seed) for name in LEARNED_SETTINGS for seed in range(3)] + [(name, 0) for name in PLANNER_SETTINGS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--fingerprints', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fingerprints = json.loads(args.fingerprints.read_text())
    cached = {}
    count = 0
    settings = []
    for setting, seed in tasks():
        directory = args.results / f'{setting}-seed-{seed}'
        raw = directory / 'summary.json.gz'
        stored = json.loads((directory / 'summary-fingerprint.json').read_text())
        require = lambda condition, message: check(condition, f'{setting}/{seed}: {message}')
        require(hashlib.sha256(raw.read_bytes()).hexdigest() == stored['stored_sha256'], 'Stored fingerprint')
        original = gzip.decompress(raw.read_bytes())
        require(len(original) == stored['uncompressed_bytes'] and hashlib.sha256(original).hexdigest() == stored['uncompressed_sha256'], 'Lossless byte fingerprint')
        summary = json.loads(original)
        require(not summary['smoke'] and not summary['training'] and not summary['checkpoint_reselection'] and not summary['rsi'], 'Frozen deployment scope')
        require(json.loads((directory / 'contract.json').read_text()) == contract(), 'Frozen design')
        require(list(summary['groups']) == [setting], 'Single setting result')
        group = summary['groups'][setting]
        if setting in LEARNED_SETTINGS:
            variant, use_vae, coordinated = LEARNED_SETTINGS[setting]
            require(group['checkpoint_sha256'] == fingerprints[f'{variant}-seed-{seed}'] and group['fresh_checkpoint_trace_matches'], 'Frozen actual weight/reload')
        else:
            method, use_vae = PLANNER_SETTINGS[setting]
            coordinated = True
        require(len(group['cases']) == 96, '96 final conditions')
        for case, row in zip(contract()['test'], group['cases']):
            require(all(row[key] == value for key, value in case.items()), 'Final case order')
            require(row['setting'] == setting and row['seed'] == seed, 'Case provenance')
            require(len(row['events']) == 96 and len(row['positions']) == len(row['target_ages']) == 97, 'Complete trace')
            key = json.dumps(case, sort_keys=True) + str((coordinated, use_vae)) + json.dumps(
                [(event['choices'], event['target_estimates']) for event in row['events']])
            signature = hashlib.sha256(json.dumps({name: value for name, value in row.items()
                if name not in ('setting', 'seed', 'selection_seconds', 'estimation_seconds', 'episode_seconds',
                                'pair_partition', 'pair_proxy_objective', 'cross_pair_assignment_fraction')}, sort_keys=True).encode()).hexdigest()
            # Replay planner decisions even if a learned method happened to pick
            # the same sequence, since the planner's stated objective is a claim.
            if key not in cached or setting in PLANNER_SETTINGS:
                wrapper = StressDispatchEnv(case)
                env = wrapper.env
                pair_planner = PairCyclePlanner(wrapper) if setting == 'pair_cycle' else None
                mse, idle, reward_values, violations = [], [], [], []
                first_all_seen = 0 if env.ever_measured[env.scenario.targets].all() else None
                require(np.array_equal(env.positions, row['positions'][0]), 'Initial positions')
                require(np.array_equal(env.raw_age[env.scenario.targets], row['target_ages'][0]), 'Initial ages')
                for step, event in enumerate(row['events']):
                    estimates = np.asarray(event['target_estimates'], dtype=np.float32)
                    require(estimates.shape == (8,) and np.isfinite(estimates).all() and ((estimates >= 0) & (estimates <= 1)).all(), 'Public estimate range')
                    predicted = np.full_like(env.values, .5)
                    predicted[env.scenario.targets] = estimates
                    env.set_estimate(predicted)
                    require(np.array_equal(env.estimate[env.scenario.targets], estimates), 'Measured values must override predictions')
                    if not use_vae:
                        require(np.array_equal(estimates, measured_mean_estimate(env)[env.scenario.targets]), 'No-VAE measurement hold')
                    state = wrapper.prepare()
                    require(state['free'].tolist() == event['free_executors'] and state['order'].tolist() == event['order'], 'Commitment release/order')
                    choices = np.asarray(event['choices'], dtype=int)
                    if setting in PLANNER_SETTINGS:
                        require(np.array_equal(planner_choices(state, method, pair_planner), choices), 'Planning rule decision')
                    mse.append(env.evaluation()['reconstruction_mse'])
                    reward, repeated = wrapper.step(choices, state, coordinated)
                    for name, value in repeated.items():
                        require(np.isclose(value, event[name]) if isinstance(value, float) else value == event[name], f'Physical event {step}/{name}')
                    require(np.array_equal(env.positions, row['positions'][step + 1]), 'Motion trace')
                    require(np.array_equal(env.raw_age[env.scenario.targets], row['target_ages'][step + 1]), 'Inspection age trace')
                    idle.append(env.evaluation()['normalized_weighted_idleness'])
                    reward_values.append(reward)
                    violations.append(event['deadline_violation'])
                    if first_all_seen is None and env.ever_measured[env.scenario.targets].all():
                        first_all_seen = step + 1
                metric = {'target_coverage': env.evaluation()['target_coverage'], 'deadline_violation': float(np.mean(violations)),
                          'late_deadline_violation': float(np.mean(violations[32:])), 'weighted_idleness': float(np.mean(idle)),
                          'public_cost': -float(np.mean(reward_values)), 'peak_age': int(np.max(row['target_ages'])),
                          'movement_distance': env.evaluation()['total_movement_distance'], 'target_prediction_mse': float(np.mean(mse)),
                          'target_assignments': wrapper.assignments,
                          'duplicate_assignment_fraction': wrapper.duplicate_assignments / max(1, wrapper.assignments),
                          'first_all_seen_step': first_all_seen if first_all_seen is not None else 97}
                for name, value in metric.items():
                    require(np.isclose(row[name], value), f'Recomputed metric {name}')
                require(np.allclose(row['prediction_mse_series'], mse) and np.allclose(row['weighted_idleness_series'], idle), 'Complete metric series')
                require(row['target_deadlines'] == wrapper.deadlines.tolist(), 'Declared deadline vector')
                for name, value in wrapper.initial_access_bounds().items():
                    require(row[name] == value, 'Startup lower bound')
                if pair_planner:
                    require(row['pair_partition'] == [list(pair) for pair in pair_planner.pairs], 'Pair partition')
                    require(np.isclose(row['pair_proxy_objective'], pair_planner.proxy_objective), 'Pair objective')
                    require(np.isclose(row['cross_pair_assignment_fraction'], pair_planner.cross_pair_assignments / max(1, pair_planner.assignments)), 'Fallback assignment count')
                cached[key] = signature
            else:
                require(cached[key] == signature, 'Identical controls have identical outcome')
            require(row['estimator_calls'] == (96 if use_vae else 0), 'Estimator budget')
            require(row['deadline_violation'] + 1e-12 >= row['startup_violation_lower_bound'], 'Cannot beat earliest-access bound')
            if coordinated:
                require(row['duplicate_assignment_fraction'] == 0, 'Explicit dedup rule')
            count += 1
        for name in METRICS + EXTRA_METRICS:
            require(np.isclose(group['metrics'][name], np.mean([row[name] for row in group['cases']])), f'Group mean {name}')
        settings.append({'setting': setting, 'seed': seed, 'records': 96, 'verified': True})
    result = {'deployment_tasks': 27, 'learned_deployment_tasks': 21, 'planner_tasks': 6,
              'method_condition_records_replayed': count, 'unique_case_choices_and_estimates': len(cached),
              'planner_decisions_recomputed': 576 * 96, 'frozen_weights_only': True,
              'scope': 'All physical/public rewards/metrics and planning rules replayed; VAE unobserved predictions are not regenerated by this CPU audit; each learned deployment separately passed fresh-weight first-case replay',
              'settings': settings}
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
    print('M7 VERIFIED', count, 'records;', len(cached), 'distinct controls')


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


if __name__ == '__main__':
    main()
