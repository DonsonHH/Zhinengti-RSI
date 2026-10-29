"""Read one B4-B0 public sample separately from its offline targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from patrol_rsi.opportunity_data import ACTION_FEATURE_NAMES, FEATURE_NAMES, HORIZONS


def load_public(condition, root_time_s, task, horizon_s):
    if root_time_s % 5 or not 5 <= root_time_s <= 1795 or not 0 <= task < 176 or horizon_s not in HORIZONS:
        raise ValueError('Original grid, task and frozen horizon required')
    with np.load(condition / 'public-features.npz') as data:
        root, h = int(root_time_s / 5) - 1, HORIZONS.index(horizon_s)
        base, action = data['base'][root, task], data['completion'][root, h, task]
        if base.shape != (len(FEATURE_NAMES),) or action.shape != (len(ACTION_FEATURE_NAMES),):
            raise ValueError('Feature schema differs')
        if base[0] != root_time_s or action[1] != root_time_s + horizon_s:
            raise ValueError('Feature time identity differs')
        return {'feature_names': list(FEATURE_NAMES + ACTION_FEATURE_NAMES),
                'values': np.concatenate((base, action)).tolist()}


def load_targets(condition, root_time_s, task, horizon_s):
    root, h = int(root_time_s / 5) - 1, HORIZONS.index(horizon_s)
    with np.load(condition / 'targets-evaluator-only.npz') as data:
        return {key: int(data[key][root, h, task]) for key in data.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=Path, required=True)
    parser.add_argument('--root-time', type=int, required=True)
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--horizon', type=int, choices=HORIZONS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    public = load_public(args.condition, args.root_time, args.task, args.horizon)
    targets = load_targets(args.condition, args.root_time, args.task, args.horizon)
    result = {'public_features_only': public, 'offline_targets_only': targets,
        'invalid_target_code': -1, 'note': 'B_reference is conditional confirmability; B_action also requires legal qualified completion. C is a separate sidecar reference, not a feature.'}
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    print(json.dumps({'output': str(args.output), 'A': targets['A'], 'B_action': targets['B_action']}))


if __name__ == '__main__':
    main()
