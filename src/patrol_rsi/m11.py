"""M11 frozen resource/route, shared-PPO and active-inspection experiment."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .active_inspection import ActiveInspection, CANDIDATE_FEATURES, heuristic_decisions
from .aisles import AisleGraph, feasibility
from .estimation_data import ATOMS, COMBINATIONS
from .estimation_model import TemperatureCVAE
from .facility import FacilityConfig
from .inspection_metrics import evaluate_episode
from .inspection_model import TaskPolicy, policy_decisions, ppo_update

BASE_SCENARIOS = ('healthy', 'coldplate_fouling', 'aux_feeder_loss',
                  'sensor_bias', 'it_feeder_loss', 'tower_derate') + COMBINATIONS
RULES = ('cycle', 'urgency', 'risk', 'rolling')


def contract():
    return {'phase': 'M11', 'protocol_version': 2, 'clock_s': 5, 'decision_steps': 359, 'physical_steps': 360,
            'first_decision_s': 5, 'end_s': 1800, 'training_episodes_each': 48,
            'formal_training_runs': 6, 'training_seeds': [0, 1, 2],
            'estimator_modes': ['nominal', 'history'], 'actor_features': list(CANDIDATE_FEATURES),
            'rules': list(RULES), 'guard_threshold_projected_age_ratio': .9,
            'validation_interval_episodes': 12, 'PPO_epochs_each_episode': 4,
            'PPO_gamma': .99, 'GAE_lambda': .95, 'PPO_clip': .2, 'learning_rate': .0003,
            'test_base_process_conditions': 52, 'test_small_episodes': 1040,
            'test_GW_episodes': 96, 'formal_test_episodes': 1136,
            'scope': 'Synthetic qualified zoned task scheduling; fixed plant, no repair, calibrated probability or recursive improver claim'}


def frozen_predictor(path, manifest):
    info = json.loads(manifest.read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != info['checkpoint_sha256']:
        raise RuntimeError('Frozen M10 estimator bytes changed')
    pack = torch.load(path, map_location='cpu', weights_only=True)
    model = TemperatureCVAE(pack['inputs'])
    model.load_state_dict(pack['state_dict'])
    model.eval()
    def predict(context):
        with torch.inference_mode():
            return model.predict(torch.as_tensor(context)).numpy().squeeze(-1) * 20 + 50
    return predict


def seed_digest(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def validation_specs(mode):
    return [dict(racks=128, variant='open', resources='provisioned', estimator=mode,
                 seed=2000 + j, scenario=scenario,
                 control='feedback' if j % 2 == 0 else 'fixed_commands')
            for j, scenario in enumerate(('healthy', 'coldplate_fouling', 'sensor_bias', 'aux_feeder_loss'))]


def test_specs():
    specs = []
    for scenario in BASE_SCENARIOS:
        for control in ('feedback', 'fixed_commands'):
            for seed in (3000, 3001):
                specs.append(dict(group='main', racks=128, variant='open', resources='provisioned',
                                  scenario=scenario, control=control, seed=seed))
    for scenario in ('coldplate_fouling', 'sensor_bias', COMBINATIONS[-1]):
        for control in ('feedback', 'fixed_commands'):
            for seed in (3000, 3001):
                specs.append(dict(group='scarce', racks=128, variant='open', resources='scarce',
                                  scenario=scenario, control=control, seed=seed))
    for scenario in ('coldplate_fouling', COMBINATIONS[-1]):
        for seed in (3002, 3003):
            specs.append(dict(group='aisle_change', racks=128, variant='restricted', resources='provisioned',
                              scenario=scenario, control='feedback', seed=seed))
    for scenario in ('healthy', 'coldplate_fouling', COMBINATIONS[-1]):
        for seed in (4000, 4001):
            for level in ('scarce', 'provisioned'):
                specs.append(dict(group='gw', racks=3200, variant='open', resources=level,
                                  scenario=scenario, control='feedback', seed=seed))
    rows = []
    for spec in specs:
        for estimator in ('nominal', 'history'):
            strategies = list(RULES) if spec['group'] != 'gw' else ['cycle', 'rolling']
            methods = [(rule, None) for rule in strategies]
            for seed in ((0,) if spec['group'] == 'gw' else (0, 1, 2)):
                methods.extend([('ppo_free', seed), ('ppo_guard', seed)])
            for method, seed in methods:
                rows.append(dict(spec, estimator=estimator, method=method, training_seed=seed))
    return rows


def case_name(spec):
    keys = ('group', 'racks', 'resources', 'variant', 'scenario', 'control', 'seed', 'estimator', 'method', 'training_seed')
    return '--'.join(str(spec[k]) for k in keys)


def run_episode(spec, predictor, method, model=None, training=False):
    env = ActiveInspection(spec, predictor=predictor)
    rollout, action_log, guard_forced = [], [], 0
    decision_seconds = 0.
    while not env.done:
        public = env.value_features()
        start = time.perf_counter()
        if method.startswith('ppo'):
            with torch.inference_mode():
                value = float(model.values(torch.as_tensor(public)))
            choices, samples, lp, _, forced = policy_decisions(
                env, model, guard=method == 'ppo_guard', stochastic=training)
            guard_forced += forced
        else:
            choices = heuristic_decisions(env, method)
            value, samples, lp = 0., [], 0.
        decision_seconds += time.perf_counter() - start
        action_log.append({'time_s': env.clock, 'choices': {str(k): v for k, v in choices.items()}})
        reward = env.step(choices)
        if training:
            rollout.append(dict(public_value=public, value=value, samples=samples, logprob=lp, reward=reward))
    if training:
        metrics = {'public_return': float(sum(r.get('public_reward', 0) for r in env.scalar_trace))}
        by_role, intervals = {}, []
    else:
        metrics, by_role, intervals = evaluate_episode(env)
    metrics.update(decision_CPU_seconds=decision_seconds, guard_forced_choices=guard_forced)
    record = {'specification': dict(spec), 'metrics': metrics, 'by_role': by_role, 'anomaly_intervals': intervals,
              'action_log': action_log, 'assignments': env.assignments, 'inspection_events': env.events,
              'scalar_trace': env.scalar_trace, 'process': env.trace.metadata,
              'team': {'count': len(env.agents), 'per_agent': [
                  {key: a[key] for key in ('id', 'role', 'zone', 'travel_m', 'service_s', 'idle_s', 'completed')}
                  for a in env.agents]}}
    return record, env, rollout


def write_json(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def train(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    predictor = frozen_predictor(args.estimator_weights, args.estimator_manifest)
    model = TaskPolicy()
    initial = seed_digest(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    rng = np.random.default_rng(15000 + args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    episodes = 2 if args.smoke else 48
    interval = 1 if args.smoke else 12
    learning, best, selected = [], -float('inf'), None
    started = time.perf_counter()
    for episode in range(episodes):
        spec = dict(racks=128, variant='open', estimator=args.estimator,
                    seed=(90000 if args.smoke else 1000) + args.seed * 100 + episode,
                    resources='scarce' if episode % 2 == 0 else 'provisioned',
                    scenario=str(rng.choice(ATOMS)), control='feedback' if episode % 4 in (0, 3) else 'fixed_commands')
        record, _, rollout = run_episode(spec, predictor, 'ppo_guard', model, training=True)
        update = ppo_update(model, optimizer, rollout)
        row = dict(episode=episode + 1, process=spec, update=update, public_return=record['metrics']['public_return'])
        if (episode + 1) % interval == 0:
            validation = [run_episode(spec, predictor, 'ppo_guard', model)[0]['metrics']['public_return']
                          for spec in validation_specs(args.estimator)]
            score = float(np.mean(validation))
            row['validation_public_return'] = score
            row['per_validation_case'] = validation
            if score > best:
                best, selected = score, episode + 1
                torch.save({'state_dict': model.state_dict(), 'estimator': args.estimator, 'seed': args.seed,
                            'selected_episode': selected, 'features': list(CANDIDATE_FEATURES)}, args.output / 'selected.pt')
            print(json.dumps({'mode': args.estimator, 'seed': args.seed, 'episode': episode + 1,
                              'validation_public_return': score}), flush=True)
        learning.append(row)
    result = {'smoke': args.smoke, 'estimator': args.estimator, 'seed': args.seed,
              'episodes': episodes, 'decision_steps': episodes * 359, 'PPO_epochs': episodes * 4,
              'initial_parameters_sha256': initial, 'parameters': sum(p.numel() for p in model.parameters()),
              'best_validation_public_return': best, 'selected_episode': selected,
              'checkpoint_sha256': hashlib.sha256((args.output / 'selected.pt').read_bytes()).hexdigest(),
              'learning': learning, 'seconds': time.perf_counter() - started,
              'device': 'CPU', 'torch': torch.__version__, 'numpy': np.__version__,
              'job_id': os.environ.get('SLURM_JOB_ID'), 'array_task': os.environ.get('SLURM_ARRAY_TASK_ID'),
              'estimator_checkpoint_sha256': hashlib.sha256(args.estimator_weights.read_bytes()).hexdigest(),
              'reward_scope': 'Public observation and qualified completion only; no evaluation true hazards'}
    write_json(args.output / 'training.json', result)
    print('M11_TRAIN_COMPLETE', args.estimator, args.seed, 'episodes', episodes, flush=True)


def evaluate(args):
    torch.set_num_threads(1)
    predictor = frozen_predictor(args.estimator_weights, args.estimator_manifest)
    models, weights = {}, []
    for mode in ('nominal', 'history'):
        for seed in (0, 1, 2):
            folder = args.models / f'{mode}-seed-{seed}'
            metadata = json.loads((folder / 'training.json').read_text())
            path = folder / 'selected.pt'
            if metadata['smoke'] or metadata['episodes'] != 48 or hashlib.sha256(path.read_bytes()).hexdigest() != metadata['checkpoint_sha256']:
                raise RuntimeError('Incomplete selected policy')
            pack = torch.load(path, map_location='cpu', weights_only=True)
            model = TaskPolicy()
            model.load_state_dict(pack['state_dict'])
            model.eval()
            models[(mode, seed)] = model
            weights.append({'mode': mode, 'seed': seed, 'training': metadata, 'path': str(path)})
    specs = test_specs()
    if len(specs) != 1136:
        raise RuntimeError('Frozen evaluation set changed')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'cases').mkdir()
    (args.output / 'process-traces').mkdir()
    index, process_index = [], {}
    for number, spec in enumerate(specs):
        record, env, _ = run_episode(spec, predictor, spec['method'],
                                    models.get((spec['estimator'], spec['training_seed'])))
        name = case_name(spec)
        path = args.output / 'cases' / (name + '.json.gz')
        raw = (json.dumps(record) + '\n').encode()
        path.write_bytes(gzip.compress(raw, mtime=0))
        process_key = '--'.join(str(spec[k]) for k in ('racks', 'scenario', 'control', 'seed'))
        if process_key not in process_index:
            process_path = args.output / 'process-traces' / (process_key + '.npz')
            np.savez_compressed(process_path, **env.trace.arrays)
            process_index[process_key] = {'path': str(process_path.relative_to(args.output)),
                'bytes': process_path.stat().st_size, 'sha256': hashlib.sha256(process_path.read_bytes()).hexdigest(),
                'process': env.trace.metadata}
        index.append({'case': name, 'specification': spec, 'record': str(path.relative_to(args.output)),
                      'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'metrics': record['metrics'], 'by_role': record['by_role'], 'process_key': process_key})
        if number % 20 == 0:
            print('M11_EVALUATED', number + 1, '/', len(specs), name, flush=True)
    write_json(args.output / 'case-index.json', index)
    write_json(args.output / 'process-index.json', process_index)
    write_json(args.output / 'selected-policies.json', weights)
    write_json(args.output / 'contract.json', contract())
    write_json(args.output / 'summary.json', {'episodes': len(index), 'process_conditions': len(process_index),
               'job_id': os.environ.get('SLURM_JOB_ID'), 'physical_model_frozen': True, 'new_policy_training_runs': 6})
    print('M11_SUITE_COMPLETE', len(index), 'episodes', flush=True)


def capacity(args):
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for racks, variant in ((128, 'open'), (128, 'restricted'), (3200, 'open')):
        graph = AisleGraph(FacilityConfig(racks=racks), variant)
        for level in ('scarce', 'provisioned'):
            rows.append(feasibility(graph, level))
        if racks == 128:
            write_json(args.output / ('aisles-' + variant + '.json'), graph.description())
    write_json(args.output / 'feasibility.json', rows)
    print('M11_CAPACITY_COMPLETE', len(rows), 'resource/layout combinations', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('train', 'evaluate', 'capacity'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--estimator', choices=('nominal', 'history'), default='nominal')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--estimator-weights', type=Path)
    parser.add_argument('--estimator-manifest', type=Path)
    parser.add_argument('--models', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    else:
        capacity(args)


if __name__ == '__main__':
    main()
