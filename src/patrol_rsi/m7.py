"""Frozen M6 checkpoints under M7 stress and deployment crossovers."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .deadline import measured_mean_estimate
from .m6 import METRICS, refresh
from .stress import PairCyclePlanner, StressDispatchEnv, planner_choices


LEARNED_SETTINGS = {
    'joint_vae': ('joint_vae', True, True),
    'joint_vae_tool_off': ('joint_vae', False, True),
    'no_vae': ('no_vae', False, True),
    'no_vae_tool_on': ('no_vae', True, True),
    'independent_targets': ('independent_targets', True, False),
    'joint_vae_mask_off': ('joint_vae', True, False),
    'independent_mask_on': ('independent_targets', True, True),
}
PLANNER_SETTINGS = {
    'risk_vae': ('risk_age_distance', True),
    'risk_hold': ('risk_age_distance', False),
    'matching_vae': ('matching', True),
    'matching_hold': ('matching', False),
    'slack': ('slack', False),
    'pair_cycle': ('pair_cycle', False),
}
EXTRA_METRICS = ('late_deadline_violation', 'first_all_seen_step', 'startup_violation_lower_bound',
                 'selection_seconds', 'estimation_seconds', 'episode_seconds', 'cross_pair_assignment_fraction')


def contract():
    development = []
    tests = []
    for family, dev_start, test_start, index_offset in [('rack', 30000, 31000, 0), ('baffle', 30100, 31100, 8)]:
        for index in range(2):
            development.append({'split': 'm7_development', 'family': family, 'layout_seed': dev_start + index,
                                'field_seed': 33000000 + index_offset + index, 'profile': 'balanced', 'noise': 0.0})
        for index in range(8):
            for profile in ('balanced', 'tight', 'pressure'):
                for noise in (0.0, .03):
                    tests.append({'split': 'm7_test', 'family': family, 'layout_seed': test_start + index,
                                  'field_seed': 34000000 + index_offset + index, 'profile': profile, 'noise': noise})
    return {'development': development, 'test': tests, 'horizon': 96, 'late_steps': [33, 96],
            'learned_settings': {key: list(value) for key, value in LEARNED_SETTINGS.items()},
            'planner_settings': {key: list(value) for key, value in PLANNER_SETTINGS.items()},
            'paired_checkpoint_seeds': [0, 1, 2], 'training': False, 'checkpoint_reselection': False,
            'noise': 'Full cell noise field keyed by field seed and absolute step; same noise for same cell/time across methods',
            'matching': {'lookahead': 24, 'discount': .99, 'objective': 'Additive one-refresh public benefit, exact assignment enumeration; no collision/route optimality claim'},
            'pair_cycle': '105 pair partitions by optimistic squared period/deadline ratios, then 24 start assignments; legal age/distance fallback',
            'layout_test_access': 'Public geometry inspected for connectivity/deduplication; no test observations, scores or checkpoint selection before the frozen final run',
            'scope': 'Static synthetic importance; centralized dispatch; human-engineered common low-level controller; not a new training or RSI experiment'}


def episode(case, setting, seed, estimator, policy, device, trace=True):
    import torch
    from .dispatch_model import tensors
    wrapper = StressDispatchEnv(case)
    env = wrapper.env
    learned = setting in LEARNED_SETTINGS
    if learned:
        _, use_vae, coordinated = LEARNED_SETTINGS[setting]
        method = 'learned'
    else:
        method, use_vae = PLANNER_SETTINGS[setting]
        coordinated = True
    start = time.perf_counter()
    pair_planner = PairCyclePlanner(wrapper) if method == 'pair_cycle' else None
    selection_seconds = time.perf_counter() - start  # includes pair partition construction
    estimation_seconds = 0.0
    ages = [env.raw_age[env.scenario.targets].copy()]
    positions = [env.positions.copy()]
    estimates = []
    mse = []
    idle = []
    violations = []
    rewards = []
    events = []
    calls = 0
    all_seen = 0 if env.ever_measured[env.scenario.targets].all() else None
    for step in range(96):
        tick = time.perf_counter()
        if use_vae:
            calls += refresh(env, estimator, device, 'joint_vae', case['field_seed'], step)
        else:
            env.set_estimate(measured_mean_estimate(env))
        estimation_seconds += time.perf_counter() - tick
        mse.append(env.evaluation()['reconstruction_mse'])
        estimates.append(env.estimate[env.scenario.targets].copy())
        tick = time.perf_counter()
        state = wrapper.prepare()
        if learned:
            with torch.no_grad():
                choices, _, _, _ = policy.joint(**tensors([state], device), coordinated=coordinated, deterministic=True)
            choices = choices[0].cpu().numpy()
        else:
            choices = planner_choices(state, method, pair_planner)
        selection_seconds += time.perf_counter() - tick
        reward, event = wrapper.step(choices, state, coordinated)
        if trace:
            events.append(event | {'choices': choices.tolist(), 'free_executors': state['free'].tolist(),
                                   'order': state['order'].tolist(), 'target_estimates': estimates[-1].tolist()})
        ages.append(env.raw_age[env.scenario.targets].copy())
        positions.append(env.positions.copy())
        violations.append(event['deadline_violation'])
        rewards.append(reward)
        idle.append(env.evaluation()['normalized_weighted_idleness'])
        if all_seen is None and env.ever_measured[env.scenario.targets].all():
            all_seen = step + 1
    metric = case | {'setting': setting, 'seed': seed, 'target_coverage': env.evaluation()['target_coverage'],
                     'deadline_violation': float(np.mean(violations)), 'weighted_idleness': float(np.mean(idle)),
                     'public_cost': -float(np.mean(rewards)), 'peak_age': int(np.max(ages)),
                     'movement_distance': env.evaluation()['total_movement_distance'], 'estimator_calls': calls,
                     'target_prediction_mse': float(np.mean(mse)), 'target_assignments': wrapper.assignments,
                     'duplicate_assignment_fraction': wrapper.duplicate_assignments / max(1, wrapper.assignments),
                     'target_deadlines': wrapper.deadlines.tolist(), 'late_deadline_violation': float(np.mean(violations[32:])),
                     'first_all_seen_step': all_seen if all_seen is not None else 97,
                     'first_all_seen_censored': all_seen is None,
                     'selection_seconds': selection_seconds, 'estimation_seconds': estimation_seconds,
                     'episode_seconds': time.perf_counter() - start,
                     'cross_pair_assignment_fraction': pair_planner.cross_pair_assignments / max(1, pair_planner.assignments) if pair_planner else 0.0}
    metric.update(wrapper.initial_access_bounds())
    if pair_planner:
        metric['pair_partition'] = [list(pair) for pair in pair_planner.pairs]
        metric['pair_proxy_objective'] = pair_planner.proxy_objective
    if trace:
        metric.update(target_ages=[row.tolist() for row in ages], positions=[row.tolist() for row in positions],
                      events=events, prediction_mse_series=mse, weighted_idleness_series=idle)
    return metric


def stable_trace(row):
    return {key: value for key, value in row.items() if key not in ('selection_seconds', 'estimation_seconds', 'episode_seconds')}


def load_policy(weights_root, setting, seed, device, fingerprints):
    import torch
    from .dispatch_model import DispatchActorCritic
    variant = LEARNED_SETTINGS[setting][0]
    path = weights_root / f'{variant}-seed-{seed}' / 'best-policy.pt'
    expected = fingerprints[f'{variant}-seed-{seed}']
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError('M6 frozen checkpoint fingerprint mismatch')
    state = torch.load(path, map_location=device, weights_only=True)
    if state['variant'] != variant or state['seed'] != seed:
        raise RuntimeError('M6 checkpoint metadata mismatch')
    policy = DispatchActorCritic().to(device).eval()
    policy.load_state_dict(state['model'])
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights-root', type=Path, required=True)
    parser.add_argument('--fingerprints', type=Path, required=True)
    parser.add_argument('--estimator', type=Path, required=True)
    parser.add_argument('--setting', choices=list(LEARNED_SETTINGS) + list(PLANNER_SETTINGS))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    import torch
    from .models import ConditionalVAEUNet
    from .train import setup
    device = setup(args.seed)
    fingerprints = json.loads(args.fingerprints.read_text())
    estimator = ConditionalVAEUNet().to(device).eval()
    if hashlib.sha256(args.estimator.read_bytes()).hexdigest() != 'fe70299e4a6006520c719c9b55b2194241a042534eb3ab348ef1134b91b4b881':
        raise RuntimeError('Frozen VAE fingerprint mismatch')
    estimator.load_state_dict(torch.load(args.estimator, map_location=device, weights_only=True)['model'])
    args.output.mkdir(parents=True, exist_ok=False)
    settings = list(LEARNED_SETTINGS) + list(PLANNER_SETTINGS) if args.smoke else [args.setting]
    if settings == [None]:
        raise ValueError('Setting required for final evaluation')
    cases = contract()['development'] if args.smoke else contract()['test']
    (args.output / 'contract.json').write_text(json.dumps(contract(), indent=2))
    groups = {}
    started = time.perf_counter()
    for setting in settings:
        policy = load_policy(args.weights_root, setting, args.seed, device, fingerprints) if setting in LEARNED_SETTINGS else None
        rows = [episode(case, setting, args.seed, estimator, policy, device) for case in cases]
        if policy is not None:
            fresh = load_policy(args.weights_root, setting, args.seed, device, fingerprints)
            repeated = episode(cases[0], setting, args.seed, estimator, fresh, device)
            if stable_trace(repeated) != stable_trace(rows[0]):
                raise RuntimeError('Fresh M6 checkpoint M7 trace replay mismatch')
        metrics = {key: float(np.mean([row[key] for row in rows])) for key in METRICS + EXTRA_METRICS}
        groups[setting] = {'metrics': metrics, 'cases': rows, 'fresh_checkpoint_trace_matches': policy is not None,
                           'checkpoint_sha256': fingerprints[f'{LEARNED_SETTINGS[setting][0]}-seed-{args.seed}'] if policy is not None else None}
        print(json.dumps({'setting': setting, 'seed': args.seed, 'cases': len(rows), 'metrics': metrics}), flush=True)
    result = {'smoke': args.smoke, 'seed': args.seed, 'groups': groups, 'seconds': time.perf_counter() - started,
              'job_id': os.environ.get('SLURM_JOB_ID'), 'gpu': torch.cuda.get_device_name(),
              'training': False, 'checkpoint_reselection': False, 'rsi': False}
    raw = (json.dumps(result) + '\n').encode()
    (args.output / 'summary.json.gz').write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
    (args.output / 'summary-fingerprint.json').write_text(json.dumps({'uncompressed_bytes': len(raw), 'uncompressed_sha256': hashlib.sha256(raw).hexdigest(),
                                                                 'stored_sha256': hashlib.sha256((args.output / 'summary.json.gz').read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
