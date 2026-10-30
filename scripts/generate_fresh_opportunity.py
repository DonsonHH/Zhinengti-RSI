"""Execute exactly one frozen seed group, private final labels, fixed Tour only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from patrol_rsi.causal_observation import ObservedTask, capture_public
from patrol_rsi.coverage_plan import exact_task_gaps
from patrol_rsi.fresh_opportunity import (MANIFEST_SHA, PROTOCOL_SHA, ROOTS, WEIGHTS_SHA,
    condition_strata, electrical_diagnostic, frozen_inputs, need, read, sha, write)
from patrol_rsi.improver_task import PatrolProgram
from patrol_rsi.information_audit import current_legal_geometry, pack_geometry
from patrol_rsi.m11 import frozen_predictor
from patrol_rsi.opportunity_data import (FeatureHistory, HORIZONS, FEATURE_NAMES, ACTION_FEATURE_NAMES,
    action_features, completion_lookup, contract, evaluator_labels, legal_completions, sample_targets)
from patrol_rsi.public_dispatch import snapshot, choose


def gz_json(path, value):
    path.write_bytes(gzip.compress((json.dumps(value, allow_nan=False) + '\n').encode(), mtime=0))


def one_condition(item, predictor, group, geometry_common):
    started = time.perf_counter()
    spec = item['specification']
    output = group / item['id']
    output.mkdir(exist_ok=False)
    os.chmod(output, 0o700)
    env = ObservedTask(spec, PatrolProgram(), predictor)
    extractor = FeatureHistory()
    roots, bases, geometry_rows, actions = [], [], [], []
    # All359 packets are retained. The59 sampled roots still contain all
    # intervening qualified measurements in the last-three-history encoding.
    with gzip.open(output / 'public-states.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for _ in range(359):
            bundle = capture_public(env)
            base = extractor.append(bundle)
            stream.write(json.dumps(bundle, allow_nan=False) + '\n')
            geometry_rows.extend(current_legal_geometry(bundle))
            if env.clock in ROOTS:
                roots.append(bundle); bases.append(base)
            groups = snapshot(env, lambda f: np.zeros(len(f)))
            choices, logs = choose(groups, 'tour')
            actions.append({'time_s': env.clock, 'choices': {str(k): v for k, v in choices.items()}})
            env.step(choices)
            env.workflow_decisions.extend(logs)
    maxima, _ = exact_task_gaps(env.tasks, env.events)
    need(not np.any(maxima > env.deadlines + 1e-9) and not env.detours_used.any(), 'Protected unchanged Tour gap')
    need([v['time_s'] for v in roots] == list(ROOTS), 'All frozen roots')
    geometry = pack_geometry(geometry_rows)
    if geometry_common is not None:
        need(set(geometry) == set(geometry_common) and all(np.array_equal(v, geometry_common[k])
            for k, v in geometry.items()), 'Public Tour geometry changed across conditions')
    else:
        np.savez_compressed(group / 'geometry.npz', **geometry)
    record = {'specification': spec, 'method': 'protected_Tour', 'action_log': actions,
        'inspection_events': env.events, 'assignments': env.assignments,
        'trace_fingerprint': env.trace.fingerprint, 'estimator_calls': env.estimator_calls,
        'exact_gap_violations': 0, 'optional_checks': 0,
        'actor_scores': 'Unused zeros; public_dispatch.choose(method=tour) always takes protected next task.'}
    gz_json(output / 'tour-record.json.gz', record)
    key = item['physical_key']
    process_path = group / 'processes' / (key + '.npz')
    metadata_path = group / 'processes' / (key + '.json')
    if not process_path.exists():
        np.savez_compressed(process_path, **env.trace.arrays)
        write(metadata_path, env.trace.metadata)
    else:
        need(read(metadata_path)['numeric_digest'] == env.trace.fingerprint, 'Noise paired physical drift')
    labels, events = evaluator_labels(env.trace.arrays, spec['seed'])
    np.savez_compressed(output / 'labels-only.npz', **labels)
    write(output / 'events-evaluator-only.json', events)
    completions = completion_lookup(geometry, record)
    targets = {k: [] for k in ('A', 'B_reference', 'B_action', 'action_geometry_index', 'legal_action_count')}
    endpoint_features = []
    for bundle, base in zip(roots, bases):
        local_targets = {key: [] for key in targets}; local_actions = []
        now = bundle['time_s']
        for horizon in HORIZONS:
            selected, count = legal_completions(now, horizon, completions, bundle)
            local_actions.append(action_features(base, now, horizon, selected, count))
            for key, values in zip(('A', 'B_reference', 'B_action'), sample_targets(labels, now, horizon, selected)):
                local_targets[key].append(values)
            local_targets['action_geometry_index'].append([v['geometry_index'] if v is not None else -2 for v in selected])
            local_targets['legal_action_count'].append(count)
        endpoint_features.append(local_actions)
        for key in targets:
            targets[key].append(local_targets[key])
    base = np.asarray(bases, dtype=np.float32)
    completion = np.asarray(endpoint_features, dtype=np.float32)
    targets = {k: np.asarray(v, dtype=np.int8 if k in ('A', 'B_reference', 'B_action') else np.int16) for k, v in targets.items()}
    np.savez_compressed(output / 'public-features.npz', base=base, completion=completion)
    np.savez_compressed(output / 'targets-evaluator-only.npz', **targets)
    strata = condition_strata(base, targets, labels, events)
    identity = dict(item, numeric_process_fingerprint=env.trace.fingerprint,
        source_process_path=str(process_path.relative_to(group)), source_process_sha256=sha(process_path),
        source_Tour_sha256=sha(output / 'tour-record.json.gz'),
        source_public_history_sha256=sha(output / 'public-states.jsonl.gz'),
        source_geometry_sha256=sha(group / 'geometry.npz'),
        base_feature_names=list(FEATURE_NAMES), action_feature_names=list(ACTION_FEATURE_NAMES),
        protocol_sha256=PROTOCOL_SHA, manifest_sha256=MANIFEST_SHA,
        feature_roots=list(ROOTS), final_test_sealed=item['split'] == 'final_test',
        C_labels='Not computed; separate B4-C task, no utility or Oracle in features.')
    write(output / 'source-identity-not-features.json', identity)
    summary = dict(item, numeric_process_fingerprint=env.trace.fingerprint,
        strata=strata, event_definitions=events, dataset_samples=int(targets['A'].size),
        all_history_frames=extractor.frames, sampled_roots=len(ROOTS),
        feature_audit={'timestamp_available_readings_checked': extractor.timestamp_readings_checked,
            'future_timestamp_violations': 0, 'all_359_history_packets_validated': True,
            'all_intervening_qualified_events_kept': True, 'root_features_only': True},
        exact_gap_violations=0, optional_checks=0, CVAE_updates=0, predictor_fits=0, PPO_updates=0, LLM_calls=0)
    if spec['scenario'] == 'dc_contact_hotspot':
        summary['electrical_diagnostic'] = electrical_diagnostic(env.trace.arrays, spec, events, strata)
    write(output / 'summary.json', summary)
    files = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in output.iterdir() if p.is_file()}
    write(output / 'files.json', files)
    print('FRESH_CONDITION_DONE', item['id'], round(time.perf_counter() - started, 3), flush=True)
    return summary, geometry


def main():
    parser = argparse.ArgumentParser()
    for name in ('protocol', 'manifest', 'output', 'estimator-weights', 'estimator-manifest', 'gate'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--group-index', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    os.umask(0o077)
    protocol, manifest = frozen_inputs(args.protocol, args.manifest)
    gate = read(args.gate)
    need(gate['success'] and gate['tests'] >= 237 and not(gate['skipped'] or gate['failures'] or gate['errors']), 'Actual Torch gate')
    need(sha(args.estimator_weights) == WEIGHTS_SHA, 'Original CVAE checkpoint SHA')
    groups = [(split, seed) for split, seeds in protocol['new_seed_groups'].items() for seed in seeds]
    need(args.group_index in range(64), 'Exactly frozen seed group')
    split, seed = groups[args.group_index]
    selected = [v for v in manifest if v['split'] == split and v['group_seed'] == seed]
    need(len(selected) == 20, 'Exactly20 observation conditions in this seed group')
    group = args.output / split / f'seed-{seed}'
    group.mkdir(parents=True, exist_ok=False)
    (group / 'processes').mkdir()
    write(group / 'run-start.json', {'group_index': args.group_index, 'split': split, 'seed': seed,
        'job_id': os.environ.get('SLURM_JOB_ID'), 'array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
        'protocol_SHA': sha(args.protocol), 'manifest_SHA': sha(args.manifest),
        'weights_SHA': sha(args.estimator_weights), 'weights_manifest_SHA': sha(args.estimator_manifest),
        'torch': torch.__version__, 'numpy': np.__version__, 'python': sys.version,
        'conditions': [v['id'] for v in selected], 'generation_only': True, 'start_unix': time.time()})
    data_contract = dict(contract(), root_grid_s=30, roots=list(ROOTS),
        history_clock_s=5, parent_schema='b4b0-opportunity-v1', stage='B4-B1a')
    write(group / 'data-contract.json', data_contract)
    predictor = frozen_predictor(args.estimator_weights, args.estimator_manifest)
    results, geometry = [], None
    begin = time.perf_counter()
    try:
        for item in selected:
            summary, geometry = one_condition(item, predictor, group, geometry)
            results.append(summary)
        need(sha(args.estimator_weights) == WEIGHTS_SHA, 'Frozen CVAE unchanged after generation')
        write(group / 'complete.json', {'group_index': args.group_index, 'split': split, 'seed': seed,
            'completed_condition_ids': [v['id'] for v in results], 'conditions': 20,
            'wall_s': time.perf_counter() - begin, 'CVAE_unchanged': True,
            'predictor_fits': 0, 'PPO_updates': 0, 'LLM_calls': 0})
    except BaseException as exc:
        write(group / 'failure.json', {'type': type(exc).__name__, 'message': str(exc),
            'completed_condition_ids': [v['id'] for v in results], 'traceback': traceback.format_exc(),
            'distribution_changed': False, 'seed_replacements': 0})
        raise
    print('FRESH_GROUP_DONE', args.group_index, split, seed, flush=True)


if __name__ == '__main__':
    main()
