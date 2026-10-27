"""M8 matched actor-information by training-distribution factorial."""
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
from .m6 import METRICS, parameter_hash, rank, refresh
from .m7 import EXTRA_METRICS, PLANNER_SETTINGS
from .relations import RelationDispatchEnv
from .stress import PairCyclePlanner, planner_choices


VARIANTS = ('rack_local', 'rack_relation', 'mixed_local', 'mixed_relation')
VAE_SHA256 = 'fe70299e4a6006520c719c9b55b2194241a042534eb3ab348ef1134b91b4b881'


def contract():
    validation, tests = [], []
    for family, valid_start, test_start, offset in (('rack', 43000, 44000, 0), ('baffle', 43100, 44100, 8)):
        for index in range(2):
            for profile in ('balanced', 'tight', 'pressure'):
                validation.append({'split': 'm8_validation', 'family': family, 'layout_seed': valid_start + index,
                                   'field_seed': 45000000 + offset + index, 'profile': profile, 'noise': 0.0})
        for index in range(8):
            for profile in ('balanced', 'tight', 'pressure'):
                for noise in (0.0, .03):
                    tests.append({'split': 'm8_test', 'family': family, 'layout_seed': test_start + index,
                                  'field_seed': 46000000 + offset + index, 'profile': profile, 'noise': noise})
    return {'preflight': {'split': 'm8_preflight', 'family': 'baffle', 'layout_seed': 39000,
                          'field_seed': 41000000, 'profile': 'balanced', 'noise': 0.0},
            'rack_train': [['rack', 40000 + index] for index in range(64)],
            'mixed_train': [['rack', 40000 + index] for index in range(32)] + [['baffle', 41000 + index] for index in range(32)],
            'validation': validation, 'test': tests, 'variants': list(VARIANTS), 'training_seeds': [0, 1, 2],
            'field_rule': '42000000 + seed*100000 + zero_based_episode', 'horizon': 96,
            'training_profiles': ['balanced', 'tight', 'pressure'], 'episodes': 1200, 'preflight_episodes': 256,
            'validation_interval': 100, 'episodes_per_update': 4, 'ppo_epochs': 4, 'minibatch': 128,
            'discount': .99, 'gae_lambda': .95, 'clip': .2, 'entropy_coefficient': .01,
            'value_half_mse_coefficient': .5, 'effective_value_mse_coefficient': .25, 'learning_rate': .0003,
            'actor_ablation': 'Identical topology/parameter count; local zeros encoded edge/team actor context; same full-public-state critic',
            'relation_features': '8 target-edge and 8 team-edge public features; shared MLP and mean pooling; not attention',
            'joint_protocol': 'Same conditional deduplication and new-commitment updates during collection and replay',
            'common_control': 'M7 sensing, paired cell/time noise, commitments, fixed shortest-distance and cyclic avoidance; static synthetic importance',
            'estimator_sha256': VAE_SHA256, 'frozen_estimator_all_training_groups': True,
            'selection': 'Shared mixed-layout validation; rank coverage loss, violation, truth-weighted idleness rounded to 6 decimals; never select by test',
            'test_access': 'Public geometry may be deduplicated in advance; final fields/observations/scores first evaluated after checkpoint freeze',
            'RSI': False, 'paper_numerical_reproduction': False}


def training_case(configuration, variant, seed, episode_index, rng):
    index = int(rng.integers(64))
    profile = configuration['training_profiles'][int(rng.integers(3))]
    family, layout = configuration['mixed_train' if variant.startswith('mixed') else 'rack_train'][index]
    case = {'split': 'm8_train', 'family': family, 'layout_seed': layout,
            'field_seed': 42000000 + seed * 100000 + episode_index, 'profile': profile, 'noise': 0.0}
    return case, [index, case['field_seed'], profile, case['noise']]


def episode(estimator, policy, case, device, variant, method='learned', collect=False, trace=False):
    import torch
    from .relation_model import tensors
    wrapper = RelationDispatchEnv(case)
    env = wrapper.env
    use_vae = True if method in ('learned', 'untrained_network') else PLANNER_SETTINGS[method][1]
    planner_method = PLANNER_SETTINGS[method][0] if method in PLANNER_SETTINGS else None
    start = time.perf_counter()
    pair_planner = PairCyclePlanner(wrapper) if planner_method == 'pair_cycle' else None
    selection_seconds = time.perf_counter() - start
    estimation_seconds = 0.0
    ages, positions = [env.raw_age[env.scenario.targets].copy()], [env.positions.copy()]
    idle, violations, rewards, mse, events, rollout = [], [], [], [], [], []
    calls = 0
    first_all_seen = 0 if env.ever_measured[env.scenario.targets].all() else None
    for step in range(96):
        tick = time.perf_counter()
        if use_vae:
            calls += refresh(env, estimator, device, 'joint_vae', case['field_seed'], step)
        else:
            env.set_estimate(measured_mean_estimate(env))
        estimation_seconds += time.perf_counter() - tick
        estimate = env.estimate[env.scenario.targets].copy()
        mse.append(env.evaluation()['reconstruction_mse'])
        tick = time.perf_counter()
        state = wrapper.prepare()
        if planner_method is None:
            with torch.no_grad():
                chosen, logp, _, value = policy.joint(**tensors([state], device), deterministic=not collect)
            choices = chosen[0].cpu().numpy()
            old_logp, old_value = float(logp[0]), float(value[0])
        else:
            choices = planner_choices(state, planner_method, pair_planner)
            old_logp = old_value = 0.0
        selection_seconds += time.perf_counter() - tick
        reward, event = wrapper.step(choices, state, True)
        if collect:
            rollout.append({'state': state, 'choices': choices.copy(), 'logp': old_logp, 'value': old_value,
                            'reward': reward, 'done': step == 95})
        if trace:
            events.append(event | {'choices': choices.tolist(), 'free_executors': state['free'].tolist(),
                                   'order': state['order'].tolist(), 'target_estimates': estimate.tolist(),
                                   'behavior_logp': old_logp})
        ages.append(env.raw_age[env.scenario.targets].copy())
        positions.append(env.positions.copy())
        idle.append(env.evaluation()['normalized_weighted_idleness'])
        violations.append(event['deadline_violation'])
        rewards.append(reward)
        if first_all_seen is None and env.ever_measured[env.scenario.targets].all():
            first_all_seen = step + 1
    row = case | {'variant': variant, 'method': method, 'target_coverage': env.evaluation()['target_coverage'],
                  'deadline_violation': float(np.mean(violations)), 'weighted_idleness': float(np.mean(idle)),
                  'public_cost': -float(np.mean(rewards)), 'peak_age': int(np.max(ages)),
                  'movement_distance': env.evaluation()['total_movement_distance'], 'estimator_calls': calls,
                  'target_prediction_mse': float(np.mean(mse)), 'target_assignments': wrapper.assignments,
                  'duplicate_assignment_fraction': wrapper.duplicate_assignments / max(1, wrapper.assignments),
                  'late_deadline_violation': float(np.mean(violations[32:])),
                  'first_all_seen_step': first_all_seen if first_all_seen is not None else 97,
                  'first_all_seen_censored': first_all_seen is None, 'target_deadlines': wrapper.deadlines.tolist(),
                  'selection_seconds': selection_seconds, 'estimation_seconds': estimation_seconds,
                  'episode_seconds': time.perf_counter() - start,
                  'cross_pair_assignment_fraction': pair_planner.cross_pair_assignments / max(1, pair_planner.assignments) if pair_planner else 0.0}
    row.update(wrapper.initial_access_bounds())
    if pair_planner:
        row.update(pair_partition=[list(pair) for pair in pair_planner.pairs], pair_proxy_objective=pair_planner.proxy_objective)
    if trace:
        row.update(events=events, target_ages=[a.tolist() for a in ages], positions=[p.tolist() for p in positions],
                   prediction_mse_series=mse, weighted_idleness_series=idle)
    return row, rollout


def average(rows):
    return {key: float(np.mean([row[key] for row in rows])) for key in METRICS + EXTRA_METRICS}


def evaluate(estimator, policy, cases, device, variant, method='learned', trace=False):
    rows = [episode(estimator, policy, case, device, variant, method, trace=trace)[0] for case in cases]
    return average(rows), rows


def stable_trace(row):
    return {key: value for key, value in row.items() if key not in ('selection_seconds', 'estimation_seconds', 'episode_seconds')}


def store_summary(output, summary):
    raw = (json.dumps(summary) + '\n').encode()
    stored = gzip.compress(raw, compresslevel=6, mtime=0)
    (output / 'summary.json.gz').write_bytes(stored)
    (output / 'summary-fingerprint.json').write_text(json.dumps({'uncompressed_bytes': len(raw),
        'uncompressed_sha256': hashlib.sha256(raw).hexdigest(), 'stored_bytes': len(stored),
        'stored_sha256': hashlib.sha256(stored).hexdigest()}, indent=2))


def train(output, estimator_path, seed, variant, preflight=False, smoke=False, planners=False):
    import torch
    from .models import ConditionalVAEUNet
    from .relation_model import RelationActorCritic, ppo_update
    from .train import setup
    configuration = contract()
    device = setup(seed)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'contract.json').write_text(json.dumps(configuration, indent=2))
    if hashlib.sha256(estimator_path.read_bytes()).hexdigest() != VAE_SHA256:
        raise RuntimeError('Frozen estimator fingerprint mismatch')
    estimator = ConditionalVAEUNet().to(device).eval()
    estimator.load_state_dict(torch.load(estimator_path, map_location=device, weights_only=True)['model'])
    start = time.perf_counter()
    if planners:
        tests = {}
        for method in PLANNER_SETTINGS:
            score, rows = evaluate(estimator, None, configuration['test'], device, 'common_planner', method, True)
            tests[method] = {'metrics': score, 'cases': rows}
        store_summary(output, {'planners': True, 'test': tests, 'training': False, 'seconds': time.perf_counter() - start,
                               'job_id': os.environ.get('SLURM_JOB_ID'), 'gpu': torch.cuda.get_device_name(), 'rsi': False})
        return
    relational = variant.endswith('_relation')
    policy = RelationActorCritic(relational).to(device)
    initial_state = {key: value.detach().clone() for key, value in policy.state_dict().items()}
    initial_hash = parameter_hash(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=configuration['learning_rate'])
    rng = np.random.default_rng(seed)
    episodes = 4 if smoke else configuration['preflight_episodes'] if preflight else configuration['episodes']
    validation = [configuration['preflight']] if smoke or preflight else configuration['validation']
    interval = 4 if smoke else 32 if preflight else 100
    initial_score, _ = evaluate(estimator, policy, validation, device, variant)
    (output / 'initial-validation.json').write_text(json.dumps(initial_score))
    best, selected_episode = None, None
    buffer, updates, behavior_error = [], 0, 0.0
    schedule = hashlib.sha256()
    with (output / 'learning.jsonl').open('x') as log:
        for index in range(episodes):
            if smoke or preflight:
                case = configuration['preflight']
            else:
                case, schedule_row = training_case(configuration, variant, seed, index, rng)
                schedule.update(json.dumps(schedule_row).encode() + b'\n')
            _, transitions = episode(estimator, policy, case, device, variant, collect=True)
            buffer += transitions
            if (index + 1) % 4 == 0:
                last = ppo_update(policy, optimizer, buffer, device)
                updates += last['gradient_updates']
                behavior_error = max(behavior_error, last['behavior_logp_max_error'])
                buffer = []
            if (index + 1) % interval == 0 or index + 1 == episodes:
                cpu_rng, gpu_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
                score, _ = evaluate(estimator, policy, validation, device, variant)
                torch.random.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(gpu_rng)
                row = {'episode': index + 1, 'team_steps': (index + 1) * 96, 'gradient_updates': updates,
                       'validation': score, 'last_ppo': last, 'seconds': time.perf_counter() - start}
                log.write(json.dumps(row) + '\n')
                log.flush()
                print(json.dumps(row), flush=True)
                if best is None or rank(score) < rank(best):
                    best, selected_episode = score, index + 1
                    torch.save({'model': policy.state_dict(), 'seed': seed, 'variant': variant,
                                'relational': relational, 'selected_episode': selected_episode, 'validation': best,
                                'contract_sha256': hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()}, output / 'best-policy.pt')
    checkpoint = torch.load(output / 'best-policy.pt', map_location=device, weights_only=True)
    policy.load_state_dict(checkpoint['model'])
    selected_hash = parameter_hash(policy)
    if selected_hash == initial_hash:
        raise RuntimeError('Policy parameters did not update')
    tests = {}
    if not smoke and not preflight:
        score, rows = evaluate(estimator, policy, configuration['test'], device, variant, trace=True)
        tests['learned'] = {'metrics': score, 'cases': rows}
        untrained = RelationActorCritic(relational).to(device)
        untrained.load_state_dict(initial_state)
        score, rows = evaluate(estimator, untrained, configuration['test'], device, variant, 'untrained_network', True)
        tests['untrained_network'] = {'metrics': score, 'cases': rows}
    replay_case = validation[0] if smoke or preflight else configuration['test'][0]
    expected = episode(estimator, policy, replay_case, device, variant, trace=True)[0] if smoke or preflight else tests['learned']['cases'][0]
    fresh = RelationActorCritic(relational).to(device)
    fresh.load_state_dict(torch.load(output / 'best-policy.pt', map_location=device, weights_only=True)['model'])
    repeated = episode(estimator, fresh, replay_case, device, variant, trace=True)[0]
    if stable_trace(expected) != stable_trace(repeated):
        raise RuntimeError('Fresh M8 checkpoint trace replay mismatch')
    if smoke or preflight:
        (output / 'preflight-trace.json').write_text(json.dumps(expected))
    store_summary(output, {'seed': seed, 'variant': variant, 'planners': False, 'preflight': preflight, 'smoke': smoke,
        'episodes': episodes, 'initial_validation': initial_score, 'best_validation': best, 'selected_episode': selected_episode,
        'preflight_improved_vs_initial': rank(best) < rank(initial_score), 'test': tests, 'gradient_updates': updates,
        'behavior_logp_max_error': behavior_error, 'fresh_checkpoint_trace_replay_matches': True,
        'initial_parameter_sha256': initial_hash, 'selected_parameter_sha256': selected_hash,
        'checkpoint_sha256': hashlib.sha256((output / 'best-policy.pt').read_bytes()).hexdigest(),
        'estimator_sha256': VAE_SHA256, 'parameters': sum(p.numel() for p in policy.parameters()),
        'schedule_sha256': schedule.hexdigest(), 'seconds': time.perf_counter() - start,
        'job_id': os.environ.get('SLURM_JOB_ID'), 'gpu': torch.cuda.get_device_name(),
        'test_used_for_selection': False, 'rsi': False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--estimator', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--variant', choices=VARIANTS, default='mixed_relation')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--planners', action='store_true')
    args = parser.parse_args()
    train(args.output, args.estimator, args.seed, args.variant, args.preflight, args.smoke, args.planners)


if __name__ == '__main__':
    main()
