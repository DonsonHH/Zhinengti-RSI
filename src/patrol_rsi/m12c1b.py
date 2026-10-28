"""C1-B0 bounded cue-quality and learned net-check-value experiment."""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .m12 import actor_scorer, load_actors, sha, write
from .m11 import frozen_predictor, seed_digest
from .m12b import specification, DEVELOPMENT, TEST
from .inspection_value import FEATURE_NAMES
from .value_dataset import Collector, labels
from .value_learning import fit, ridge, scorer, identity
from .value_run import run

THRESHOLDS = (0., .1, .3, 'off')


def contract():
    return {'stage': 'C1-B0 supervised one-global-check skill and public cue quality',
        'train_seeds': [7800, 7801, 7802, 7803], 'validation_seeds': [7850, 7851], 'test_seeds': [7900, 7901],
        'smoke_seed': 7899, 'train_physical': 16, 'validation_physical': 8, 'test_physical': 16,
        'test_observation_conditions': 32, 'noise_c_test': [.2, 1.], 'global_optional_budget': 1,
        'natural_train_caps': {'unconfirmed_cue': 6, 'no_cue_age_at_least_half_deadline': 3, 'remaining': 3},
        'teacher_cap_per_baseline': 4, 'heldout_natural_caps': {'cue': 2, 'other': 2},
        'feature_names': list(FEATURE_NAMES), 'initialization_seeds': [0, 1, 2], 'updates': 400,
        'optimizer': 'Adam', 'learning_rate': .001, 'hidden_widths': [32, 32], 'activation': 'Tanh',
        'thresholds_in_tie_order': list(THRESHOLDS), 'selection_order': ['total_missed_segments', 'pooled_weighted_wait_s', 'total_travel_m'],
        'net_label': 'baseline_missed - branch_missed + pooled_weighted_wait_gain_s/3000 - added_travel_m/100000',
        'target_label': 'target_new_confirmations + max(target_weighted_wait_gain_s,0)/3000; ignores all collateral losses',
        'estimation': 'Current estimates on identical tour histories; hold, clipped linear history, nominal, frozen CVAE',
        'actor_control': 'Best positive optional-vs-baseline frozen actor advantage at first eligible frame, one global action',
        'LLM_calls': 0, 'PPO_or_CVAE_retrained': False, 'improver_updated': False,
        'scope': 'Supervised skill prototype; not continuous PPO, full RSI or field validation'}


def specs(split, smoke=False):
    if smoke:
        return [specification('smoke', a, b, 7899) for a, b in DEVELOPMENT[:2]]
    fixtures, seeds, noises = (DEVELOPMENT, range(7800, 7804), (.2,)) if split == 'train' else (
        (DEVELOPMENT, (7850, 7851), (.2,)) if split == 'validation' else (TEST, (7900, 7901), (.2, 1.)))
    return [specification(split, a, b, seed, noise) for a, b in fixtures for seed in seeds for noise in noises]


def case_name(spec):
    return '--'.join(str(spec[k]) for k in ('phase', 'scenario', 'duration_s', 'seed', 'noise_c', 'estimator'))


def aggregate(rows):
    weight = sum(r['event_weight_sum'] for r in rows)
    return {'cases': len(rows), 'segments': sum(r['metrics']['anomaly_segments'] for r in rows),
        'missed': sum(r['metrics']['missed_segments'] for r in rows),
        'weighted_wait_s': sum(r['weighted_delay_sum'] for r in rows) / weight if weight else None,
        'travel_m': sum(r['metrics']['travel_m'] for r in rows), 'service_s': sum(r['metrics']['service_s'] for r in rows),
        'extra_checks': sum(r['metrics']['selected_detours'] for r in rows),
        'false_measured_confirmations': sum(r['metrics']['false_measured_confirmations'] for r in rows),
        'exact_violations': sum(r['metrics']['tasks_violating_exact_gap'] for r in rows)}


def rank(row):
    return (row['missed'], row['weighted_wait_s'] if row['weighted_wait_s'] is not None else 0., row['travel_m'])


class Store:
    def __init__(self, output, predictor, actors, fingerprints, estimator):
        self.output, self.predictor, self.actors = output, predictor, actors
        self.actor_score = actor_scorer(actors[0])
        self.rows, self.processes = [], {}
        output.mkdir(parents=True, exist_ok=False)
        (output / 'cases').mkdir();(output / 'process-traces').mkdir()
        write(output / 'contract.json', contract())
        write(output / 'immutable-inputs.json', {'actors': fingerprints, 'estimator_sha256': sha(estimator),
            'job_id': os.environ.get('SLURM_JOB_ID'), 'torch': torch.__version__, 'numpy': np.__version__})

    def execute(self, spec, tag, method='tour', **kwargs):
        key = case_name(spec) + '--' + tag
        record, env = run(spec, self.predictor, self.actor_score, method, **kwargs)
        path = self.output / 'cases' / (key + '.json.gz')
        if path.exists():
            raise RuntimeError('No silent case overwrite')
        path.write_bytes(gzip.compress((json.dumps(record) + '\n').encode(), mtime=0))
        process_key = '--'.join(str(spec[k]) for k in ('scenario', 'duration_s', 'seed', 'fault_start_s'))
        if process_key not in self.processes:
            dest = self.output / 'process-traces' / (process_key + '.npz')
            np.savez_compressed(dest, **env.trace.arrays)
            self.processes[process_key] = {'path': str(dest.relative_to(self.output)), 'sha256': sha(dest),
                'bytes': dest.stat().st_size, 'numeric_digest': env.trace.fingerprint}
        elif self.processes[process_key]['numeric_digest'] != env.trace.fingerprint:
            raise RuntimeError('Policy changed physical process')
        row = {'case': key, 'tag': tag, 'specification': spec, 'deployment': record['deployment'],
            'path': str(path.relative_to(self.output)), 'sha256': sha(path), 'bytes': path.stat().st_size,
            'process_key': process_key, 'metrics': record['metrics'], 'anomaly_intervals': record['anomaly_intervals'],
            'event_weight_sum': sum(r['weight'] for r in record['anomaly_intervals']),
            'weighted_delay_sum': sum(r['weight'] * r['delay_or_censor_s'] for r in record['anomaly_intervals'])}
        self.rows.append(row)
        if len(self.rows) % 16 == 0:
            print('C1B_CASES', len(self.rows), tag, flush=True)
        return row, record

    def finish(self, **extra):
        write(self.output / 'case-index.json', self.rows)
        write(self.output / 'process-index.json', self.processes)
        write(self.output / 'summary.json', {'actual_cases': len(self.rows), 'physical_processes': len(self.processes),
            'LLM_calls': 0, 'job_id': os.environ.get('SLURM_JOB_ID'), **extra})


def read(path):
    return json.loads(path.read_text())


def data_stage(store, smoke):
    dataset, quality = [], []
    for split in ('train', 'validation') if not smoke else ('train',):
        for spec in specs(split, smoke):
            observer = Collector()
            base_row, base = store.execute(spec, 'tour-data', observer=observer, probe_enabled=True)
            quality.append({'case': base_row['case'], 'split': split, 'confusion': observer.confusion, 'temperature': observer.quality})
            if split != 'train':
                continue
            selected = observer.sample(base)
            if smoke:
                selected = selected[:3]
            for i, item in enumerate(selected):
                row, branch = store.execute(spec, 'label-' + str(i), intervention=item['intervention'])
                dataset.append({**item, 'base_case': base_row['case'], 'branch_case': row['case'],
                    'labels': labels(base, branch, item['intervention']['task'])})
    write(store.output / 'dataset.json', dataset)
    write(store.output / 'quality.json', quality)
    store.finish(stage='data', smoke=smoke, labeled_actions=len(dataset))


def fit_stage(store, data, trial, smoke):
    dataset = read(data / 'dataset.json')
    models, logs = fit(dataset, trial, updates=5 if smoke else 400)
    if trial == 0:
        models['ridge'] = ridge(dataset)
    (store.output / 'models').mkdir()
    selected = {}
    for name, pack in models.items():
        model_id = identity(pack)
        write(store.output / 'models' / (name + '.json'), pack)
        values = []
        for threshold in THRESHOLDS:
            group = []
            for spec in specs('validation', smoke):
                row, _ = store.execute(spec, name + '-threshold-' + str(threshold), name,
                    score=scorer(pack), threshold=threshold, model_id=model_id)
                group.append(row)
            values.append({'threshold': threshold, 'case_ids': [r['case'] for r in group], 'validation': aggregate(group)})
        winner = min(values, key=lambda v: rank(v['validation']))
        selected[name] = {'model_id': model_id, 'threshold': winner['threshold'], 'validation': winner['validation'], 'all_candidates': values}
    write(store.output / 'training.json', {'trial': trial, 'dataset_sha256': sha(data / 'dataset.json'),
        'updates': 5 if smoke else 400, 'seconds_includes_validation': None, 'training': logs})
    write(store.output / 'selections-frozen.json', {'trial': trial, 'models': selected,
        'test_has_not_started': True, 'frozen_at_unix': time.time()})
    store.finish(stage='fit', smoke=smoke, trial=trial)


def verify_frozen_all(run_root):
    frozen = []
    for trial in range(3):
        folder = run_root / ('fit-' + str(trial))
        selection = read(folder / 'selections-frozen.json')
        for name, row in selection['models'].items():
            if identity(read(folder / 'models' / (name + '.json'))) != row['model_id']:
                raise RuntimeError('Selected weights changed before test')
        frozen.append({'trial': trial, 'selection_sha256': sha(folder / 'selections-frozen.json'), 'frozen_at_unix': selection['frozen_at_unix']})
    return {'all_three_trials_frozen': frozen, 'test_start_unix': time.time()}


def test_stage(store, run_root, trial, common):
    write(store.output / 'test-start-gate.json', verify_frozen_all(run_root))
    fit_folder = run_root / ('fit-' + str(0 if common else trial))
    selected = read(fit_folder / 'selections-frozen.json')['models']
    groups, sampled = {}, []
    methods = ('tour', 'cue_rule', 'actor', 'ridge') if common else ('untrained', 'target', 'net', 'net-nominal')
    for name in methods:
        group = []
        source = 'net' if name == 'net-nominal' else name
        pack = read(fit_folder / 'models' / (source + '.json')) if source in selected else None
        for spec in specs('test'):
            if name == 'net-nominal':
                spec = dict(spec, estimator='nominal')
            observer = Collector() if name == 'tour' else None
            args = {'score': scorer(pack), 'threshold': selected[source]['threshold'], 'model_id': identity(pack)} if pack else {}
            if observer is not None:
                args.update(observer=observer, probe_enabled=True)
            row, record = store.execute(spec, name, source, **args)
            group.append(row)
            if observer is not None:
                sampled.append({'case': row['case'], 'specification': spec, 'samples': observer.sample(record, test=True),
                    'confusion': observer.confusion, 'temperature': observer.quality})
        groups[name] = {'case_ids': [r['case'] for r in group], 'aggregate': aggregate(group)}
    if common:
        write(store.output / 'test-tour-samples-and-quality.json', sampled)
    write(store.output / 'test-groups.json', groups)
    store.finish(stage='test-common' if common else 'test-trial', trial=trial)


def heldout_stage(store, run_root):
    predictions, examples = [], read(run_root / 'test-common' / 'test-tour-samples-and-quality.json')
    models = {}
    for trial in range(3):
        folder = run_root / ('fit-' + str(trial))
        for name in ('net', 'target', 'untrained') + (('ridge',) if trial == 0 else ()):
            models[str(trial) + '/' + name] = scorer(read(folder / 'models' / (name + '.json')))
    # Save predictions before any held-out branch outcomes exist.
    for example in examples:
        for i, sample in enumerate(example['samples']):
            x = np.asarray([sample['x']], dtype=np.float32)
            predictions.append({'baseline': example['case'], 'specification': example['specification'], 'sample_number': i, **sample,
                'predictions': {name: float(f(x)[0]) for name, f in models.items()}})
    write(store.output / 'predictions-frozen-before-labels.json', {'saved_at_unix': time.time(), 'samples': predictions})
    result = []
    for item in predictions:
        base_path = run_root / 'test-common/cases' / (item['baseline'] + '.json.gz')
        base = json.loads(gzip.decompress(base_path.read_bytes()))
        row, branch = store.execute(item['specification'], 'heldout-label-' + str(item['sample_number']), intervention=item['intervention'])
        result.append({**item, 'branch_case': row['case'], 'labels': labels(base, branch, item['intervention']['task'])})
    write(store.output / 'heldout-labels.json', result)
    store.finish(stage='heldout', samples=len(result))


def main():
    p = argparse.ArgumentParser()
    for key in ('output', 'actors', 'estimator-weights', 'estimator-manifest'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--stage', choices=('data', 'fit', 'test-common', 'test-trial', 'heldout'), required=True)
    p.add_argument('--data', type=Path);p.add_argument('--run-root', type=Path)
    p.add_argument('--trial', type=int, choices=(0, 1, 2), default=0);p.add_argument('--smoke', action='store_true')
    args = p.parse_args();torch.set_num_threads(1)
    started = time.perf_counter()
    actors, fingerprints = load_actors(args.actors)
    predictor = frozen_predictor(args.estimator_weights, args.estimator_manifest)
    store = Store(args.output, predictor, actors, fingerprints, args.estimator_weights)
    if args.stage == 'data':
        data_stage(store, args.smoke)
    elif args.stage == 'fit':
        fit_stage(store, args.data, args.trial, args.smoke)
    elif args.stage in ('test-common', 'test-trial'):
        test_stage(store, args.run_root, args.trial, args.stage == 'test-common')
    else:
        heldout_stage(store, args.run_root)
    for seed, model in actors.items():
        if seed_digest(model) != fingerprints[str(seed)]['parameters_sha256']:
            raise RuntimeError('Frozen old actor changed')
    write(args.output / 'completed.json', {'stage': args.stage, 'seconds': time.perf_counter() - started,
        'old_actors_unchanged': True, 'estimator_sha256': sha(args.estimator_weights), 'LLM_calls': 0})
    print('C1B_COMPLETE', args.stage, len(store.rows), flush=True)


if __name__ == '__main__':
    main()
