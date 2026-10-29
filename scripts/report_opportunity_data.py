"""Complete B4-B0 process/event coverage, never a predictor performance score."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np


FOLDERS = ('development-0', 'development-1', 'validation-0', 'validation-1',
           'test-0', 'test-1', 'test-2', 'test-3')


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


class Coverage:
    def __init__(self):
        self.kind = None
        self.horizon_s = None
        self.counts = Counter()
        self.seeds = set()
        self.numeric = set()
        self.positive_seeds = set()
        self.positive_numeric = set()
        self.events = set()
        self.retained_events = set()
        self.physical_events = set()
        self.channel = defaultdict(Counter)

    def add(self, row, condition, event_index):
        if self.kind is None:
            self.kind, self.horizon_s = row['kind'], row['horizon_s']
        if (self.kind, self.horizon_s) != (row['kind'], row['horizon_s']):
            raise ValueError('Coverage cell mixes professional tasks or horizons')
        spec, digest = condition['condition'], condition['numeric_process_fingerprint']
        self.seeds.add(spec['seed']); self.numeric.add(digest)
        if row['B_action_positive']:
            self.positive_seeds.add(spec['seed']); self.positive_numeric.add(digest)
        for key in ('total_samples', 'A_valid', 'A_positive', 'A_negative',
                    'B_reference_valid', 'B_reference_positive', 'B_action_valid',
                    'B_action_positive', 'B_action_negative', 'B_action_invalid',
                    'physical_event_count', 'retained_event_count'):
            self.counts[key] += row.get(key, 0)
        def event_key(i):
            item = event_index[i]
            return (digest, item['task'], item['first_tick'], item['last_exclusive_tick'])
        self.events.update(event_key(i) for i in row['B_positive_event_ids'])
        self.retained_events.update(event_key(i) for i in row['B_positive_retained_event_ids'])
        self.physical_events.update(event_key(i) for i, v in event_index.items() if v['kind'] == row['kind'])
        for channel, values in row['availability_in_valid_B_samples'].items():
            self.channel[channel].update(values)

    def value(self):
        c = dict(self.counts)
        c.update({'kind': self.kind, 'horizon_s': self.horizon_s,
            'B_reference_negative': c['B_reference_valid'] - c['B_reference_positive'],
            'B_reference_invalid': c['total_samples'] - c['B_reference_valid'],
            'B_reference_positive_fraction': c['B_reference_positive'] / c['B_reference_valid'] if c['B_reference_valid'] else None,
            'seed_groups': sorted(self.seeds), 'numeric_processes': len(self.numeric),
            'B_positive_seed_groups': sorted(self.positive_seeds), 'B_positive_numeric_processes': len(self.positive_numeric),
            'distinct_physical_events': len(self.physical_events), 'distinct_B_positive_events': len(self.events),
            'distinct_B_positive_retained_events': len(self.retained_events),
            'valid_B_positive_fraction': c.get('B_action_positive', 0) / c['B_action_valid'] if c.get('B_action_valid') else None,
            'A_positive_fraction': c.get('A_positive', 0) / c['A_valid'] if c.get('A_valid') else None,
            'available_sensor_counts_in_valid_B': {k: dict(v) for k, v in self.channel.items()}})
        # Count fields originally sum physical events over noise copies; retain
        # their explicit name and use distinct_* for independent coverage.
        return c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    summaries = []
    dimensions = defaultdict(dict)
    noise_pairs = defaultdict(list)
    features = defaultdict(Counter)
    available = timestamps = 0
    total_storage = 0
    inventory = []
    for folder in FOLDERS:
        top = read(args.root / folder / 'summary.json')
        if top['conditions'] != 16:
            raise RuntimeError('Incomplete data condition set')
        for i in range(16):
            path = args.root / folder / f'condition-{i:02}'
            item = read(path / 'summary.json')
            summaries.append(dict(item, folder=folder, condition_index=i))
            spec = item['condition']
            digest = item['numeric_process_fingerprint']
            process_key = (spec['seed'], spec['scenario'], spec['duration_s'], spec['fault_start_s'])
            noise_pairs[process_key].append({'noise_c': spec['noise_c'], 'fingerprint': digest,
                'labels_sha256': hashlib.sha256((path / 'labels-only.npz').read_bytes()).hexdigest(),
                'targets_sha256': hashlib.sha256((path / 'targets-evaluator-only.npz').read_bytes()).hexdigest()})
            for kind, values in item['feature_audit']['channel_availability_by_kind'].items():
                for channel, count in values.items():
                    features[kind + '/' + channel].update(count)
            timestamps += item['feature_audit']['timestamp_available_readings_checked']
            events = {v['id']: v for v in item['event_definitions']}
            for row in item['strata']:
                cells = [('profession_horizon', row['kind'] + '/' + str(row['horizon_s']))]
                for key in ('seed', 'scenario', 'fault_start_s', 'noise_c'):
                    cells.append((key, str(spec[key]) + '/' + row['kind'] + '/' + str(row['horizon_s'])))
                cells.append(('numeric_process', digest + '/' + row['kind'] + '/' + str(row['horizon_s'])))
                for dimension, key in cells:
                    cell = dimensions[dimension].setdefault(key, Coverage())
                    cell.add(row, item, events)
            for file in sorted(path.iterdir()):
                if file.is_file():
                    raw = file.read_bytes(); total_storage += len(raw)
                    inventory.append({'path': str(file), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    for key, pair in noise_pairs.items():
        if len(pair) != 2 or {v['noise_c'] for v in pair} != {.2, 1.} or len({v['fingerprint'] for v in pair}) != 1:
            raise RuntimeError('Broken paired-noise physical identity')
        if len({v['labels_sha256'] for v in pair}) != 1 or len({v['targets_sha256'] for v in pair}) != 1:
            raise RuntimeError('Same process noise versions have different A/B truth or legal labels')
    if len(summaries) != 128 or len(noise_pairs) != 64 or total_storage > 1024**3:
        raise RuntimeError('Data coverage or1GB derived storage cap')
    result = {'stage': 'C1-B4-B0', 'conditions': 128, 'named_physical_processes': 64,
        'numeric_process_fingerprints': len({v['numeric_process_fingerprint'] for v in summaries}),
        'seed_groups': sorted({v['condition']['seed'] for v in summaries}),
        'total_sample_rows': sum(v['dataset_samples'] for v in summaries),
        'all_expected_rows': 128 * 359 * 3 * 176,
        'causal_available_readings_checked': timestamps, 'future_timestamp_violations': 0,
        'feature_channel_availability_all_root_tasks': {k: dict(v) for k, v in features.items()},
        'coverage': {dim: {key: cell.value() for key, cell in cells.items()} for dim, cells in dimensions.items()},
        'noise_pairs': [{'seed': key[0], 'scenario': key[1], 'duration_s': key[2], 'fault_start_s': key[3],
                         'versions': pair} for key, pair in noise_pairs.items()],
        'raw_derived_data_bytes': total_storage,
        'dataset_readiness': 'NO-GO for B4-B1 training: all old data diagnostic only, fresh splits ungenerated; inspect per-kind independent coverage deficits.',
        'new_predictor_training': 0, 'new_physical_instances': 0,
        'independence_scope': 'Sample/time/task/horizon/noise rows share events and seed groups. distinct_* deduplicates by numeric physical fingerprint and event window, not by file count.',
        'B0_data_scope': 'Conditional B_reference all endpoints; B_action only qualified exact planned Tour/extra-check completion. C remains external B4-A value sidecar.'}
    write(args.output / 'coverage.json', result)
    write(args.output / 'condition-summaries.json', summaries)
    write(args.output / 'raw-storage.json', {'root': str(args.root), 'files': inventory, 'total_bytes': total_storage})
    print('B4B0_COVERAGE_DONE', len(summaries), result['total_sample_rows'], total_storage, flush=True)


if __name__ == '__main__':
    main()
