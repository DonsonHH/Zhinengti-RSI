"""Materialize one exact candidate from factorized public/evaluator records."""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np

from patrol_rsi.information_audit import scalar_candidate
from patrol_rsi.public_stopping import StopConfig


def json_finite(value):
    """Undefined event waiting remains null, never a fabricated zero."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_finite(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, required=True, help='Downloaded development-0 etc.')
    parser.add_argument('--condition', type=int, required=True)
    parser.add_argument('--root-time', type=int, required=True)
    parser.add_argument('--candidate-index', type=int, required=True)
    parser.add_argument('--candidate', choices=('C0', 'C1', 'C2', 'C3'), default='C0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.root_time % 5 or not 5 <= args.root_time <= 1795:
        raise ValueError('A saved five-second public decision root is required')
    folder = args.folder / f'condition-{args.condition:02}'
    bundle = None
    with gzip.open(folder / 'public-states.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            value = json.loads(line)
            if value['time_s'] == args.root_time:
                bundle = value
                break
    if bundle is None:
        raise ValueError('Missing saved public root')
    with np.load(args.folder / 'geometry.npz') as arrays:
        geometry = {k: arrays[k].copy() for k in arrays.files}
    i = args.candidate_index
    if not 0 <= i < len(geometry['task']) or geometry['dispatch_time_s'][i] < args.root_time:
        raise ValueError('Candidate is outside the root hard-legal continuation universe')
    configs = {'C0': (.5, 0., 0, 0.), 'C1': (.5, .25, 60, 0.),
               'C2': (1., .25, 60, 0.), 'C3': (1., .25, 60, .01)}
    collateral, future, soon, threshold = configs[args.candidate]
    config = StopConfig(collateral=collateral, future_risk=future, tour_soon_s=soon, threshold=threshold)
    score = scalar_candidate(bundle, geometry, i, config)
    target = bundle['tasks'][score['task']]
    with np.load(folder / 'analytic-continuations.npz') as arrays:
        effect = {k: arrays[k][i].tolist() for k in arrays.files}
    public = {'root_time_s': args.root_time, 'candidate': args.candidate, 'geometry_index': i,
              'public_score': score, 'target_public_channels_and_timestamps': target,
              'public_affected_tasks': [bundle['tasks'][v['task']] for v in score['task_costs']],
              'excluded_known_alarm': target['already_alarm'],
              'excluded_missing_signal': score['target_signal'] is None,
              'excluded_zero_signal': score['target_signal'] is not None and score['target_signal'] <= 0,
              'excluded_threshold': score['net_score'] is not None and score['net_score'] <= config.threshold,
              'excluded_near_Tour_veto': score['tour_slack_veto']}
    # Evaluator outcomes are deliberately a sibling, not part of public input.
    result = {'public_candidate': public, 'offline_evaluator_only': effect,
              'scope': 'Old Tour-continuation one-check candidate. Labels use future truth and cannot be deployed or fed into a model as public history.'}
    serialized = json.dumps(json_finite(result), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(serialized)
    print(json.dumps({'output': str(args.output), 'geometry_index': i, 'target': score['task'],
                      'root_time': args.root_time, 'dispatch_time': score['dispatch_time_s']}))


if __name__ == '__main__':
    main()
