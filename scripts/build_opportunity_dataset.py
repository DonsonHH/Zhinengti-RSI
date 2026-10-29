"""B4-B0 old-data audit: separate public features, A/B truth and C pointers."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from patrol_rsi.causal_observation import CHANNELS
from patrol_rsi.m9 import arrays_digest
from patrol_rsi.opportunity_data import (FEATURE_NAMES, HORIZONS, KINDS, FeatureHistory,
    action_features, completion_lookup, contract, evaluator_labels, legal_completions, sample_targets)


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(value, message):
    if not value:
        raise RuntimeError(message)


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def one_condition(source_root, b4a_root, folder, index, output, expected):
    started = time.perf_counter()
    source = source_root / folder
    prior = b4a_root / folder / f'condition-{index:02}'
    for name in ('summary.json', 'public-states.jsonl.gz', 'analytic-continuations.npz'):
        path = prior / name
        need(sha(path) == expected[str(path)], 'Old B4-A factor bytes changed')
    summary = read(prior / 'summary.json')
    spec = summary['condition']
    case_index = read(source / 'case-index.json')
    tour_row = next(v for v in case_index if v['alias'] == 'tour' and v['specification'] == spec)
    tour_path = source / tour_row['path']
    anchored_tour = next(v for v in summary['source_cases'] if v['case'] == tour_row['case'])
    need(sha(tour_path) == tour_row['sha256'] == anchored_tour['source_sha256'],
         'Original Tour record differs from independently hash-bound B4-A source')
    record = json.loads(gzip.decompress(tour_path.read_bytes()))
    process = read(source / 'process-index.json')[tour_row['process_key']]
    need(sha(source / process['path']) == process['sha256'], 'Original physical arrays changed')
    with np.load(source / process['path']) as arrays:
        truth = {k: arrays[k].copy() for k in arrays.files}
    need(arrays_digest(truth) == record['trace_fingerprint'] == process['numeric_digest'],
         'Original physical numeric fingerprint differs from anchored Tour record')
    geometry_path = b4a_root / folder / 'geometry.npz'
    need(sha(geometry_path) == expected[str(geometry_path)], 'Original legal geometry changed')
    with np.load(geometry_path) as arrays:
        geometry = {k: arrays[k].copy() for k in arrays.files}
    completions = completion_lookup(geometry, record)
    labels, events = evaluator_labels(truth, spec['seed'])
    output.mkdir(exist_ok=False)
    np.savez_compressed(output / 'labels-only.npz', **labels)
    write(output / 'events-evaluator-only.json', events)
    extractor = FeatureHistory()
    bases, actions, a_rows, ref_rows, b_rows = [], [], [], [], []
    ids, candidates = [], []
    rows = []
    with gzip.open(prior / 'public-states.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            bundle = json.loads(line)
            base = extractor.append(bundle)
            bases.append(base)
            root = bundle['time_s']
            endpoint = []
            aa, rr, bb, ci, counts = [], [], [], [], []
            for horizon in HORIZONS:
                selected, count = legal_completions(root, horizon, completions, bundle)
                endpoint.append(action_features(base, root, horizon, selected, count))
                a, reference, b = sample_targets(labels, root, horizon, selected)
                aa.append(a); rr.append(reference); bb.append(b)
                ci.append([v['geometry_index'] if v is not None else -2 for v in selected])
                counts.append(count)
            actions.append(endpoint); a_rows.append(aa); ref_rows.append(rr); b_rows.append(bb)
            ids.append(ci); candidates.append(counts)
    need(extractor.frames == 359, 'All old public roots must be present')
    bases = np.asarray(bases, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    a_rows, ref_rows, b_rows = (np.asarray(v, dtype=np.int8) for v in (a_rows, ref_rows, b_rows))
    ids, candidates = (np.asarray(v, dtype=np.int16) for v in (ids, candidates))
    # Public feature matrices cannot carry source metadata, evaluator outcomes
    # or C utility values; all identifiers are in a separate file below.
    np.savez_compressed(output / 'public-features.npz', base=bases, completion=actions)
    np.savez_compressed(output / 'targets-evaluator-only.npz', A=a_rows, B_reference=ref_rows,
                        B_action=b_rows, action_geometry_index=ids, legal_action_count=candidates)
    kind_ids = np.r_[np.zeros(128, dtype=int), np.ones(16, dtype=int), np.full(16, 2), np.full(16, 3)]
    final_times = np.asarray(list(range(5, 1800, 5)))[:, None] + np.asarray(HORIZONS)[None, :]
    ticks = np.clip((final_times / 5).astype(int) - 1, 0, 359)
    for kind_index, kind in enumerate(KINDS):
        task_mask = kind_ids == kind_index
        for h, horizon in enumerate(HORIZONS):
            a, b, reference = a_rows[:, h, task_mask], b_rows[:, h, task_mask], ref_rows[:, h, task_mask]
            event = labels['event_id'][ticks[:, h]][:, task_mask]
            retained = labels['retained_event_id'][ticks[:, h]][:, task_mask]
            def event_list(values, mask):
                return sorted(int(v) for v in np.unique(values[mask]) if v > 0)
            own_events = [v for v in events if v['kind'] == kind]
            availability = {}
            for channel in CHANNELS:
                available = bases[:, task_mask, FEATURE_NAMES.index(channel + '.available')] > 0
                availability[channel] = {'B_valid_available': int((available & (b >= 0)).sum()),
                    'B_valid_missing': int((~available & (b >= 0)).sum())}
            rows.append({'kind': kind, 'horizon_s': horizon, 'total_samples': int(a.size),
                'A_valid': int((a >= 0).sum()), 'A_positive': int((a == 1).sum()), 'A_negative': int((a == 0).sum()),
                'B_reference_valid': int((reference >= 0).sum()), 'B_reference_positive': int((reference == 1).sum()),
                'B_action_valid': int((b >= 0).sum()), 'B_action_positive': int((b == 1).sum()),
                'B_action_negative': int((b == 0).sum()), 'B_action_invalid': int((b < 0).sum()),
                'A_positive_event_ids': event_list(event, a == 1),
                'B_positive_event_ids': event_list(event, b == 1),
                'B_positive_retained_event_ids': event_list(retained, b == 1),
                'availability_in_valid_B_samples': availability,
                'physical_event_count': len(own_events),
                'retained_event_count': sum(v['retained_at_least_15s'] for v in own_events)})
    audit = {'frames': extractor.frames, 'root_task_features': extractor.frames * 176,
        'timestamp_available_readings_checked': extractor.timestamp_readings_checked,
        'future_timestamp_violations': 0, 'closed_feature_schema_passed': True,
        'channel_availability_by_kind': {kind: {channel: {'available': int(extractor.available_channels[k, c]),
            'total': int(extractor.total_channels[k, c]), 'missing': int(extractor.total_channels[k, c] - extractor.available_channels[k, c])}
            for c, channel in enumerate(CHANNELS)} for k, kind in enumerate(KINDS)},
        'missing_padding_has_explicit_availability_masks': True,
        'feature_files_have_no_seed_scenario_event_truth_or_oracle_fields': True}
    write(output / 'feature-causality-audit.json', audit)
    source_id = {'condition': spec, 'original_numeric_process_fingerprint': process['numeric_digest'],
        'source_Tour_sha256': tour_row['sha256'], 'source_process_sha256': process['sha256'],
        'source_B4A_history_sha256': sha(prior / 'public-states.jsonl.gz'),
        'original_process_key': tour_row['process_key'], 'base_feature_names': list(FEATURE_NAMES),
        'C_value_label_reference_only': {'path': str(prior / 'analytic-continuations.npz'),
                                       'sha256': sha(prior / 'analytic-continuations.npz'),
                                       'join': 'action_geometry_index>=0; Tour -1 and unavailable -2 have no extra-CHECK C label'},
        'feature_dtypes': 'float32', 'label_dtypes': 'int8 -1 invalid,0 negative,1 positive',
        'training_split': 'diagnostic_old_only', 'two_noise_versions_share_physical_events': True}
    write(output / 'source-identity-not-features.json', source_id)
    result = {'condition': spec, 'numeric_process_fingerprint': process['numeric_digest'],
        'event_definitions': events, 'strata': rows, 'feature_audit': audit,
        'dataset_samples': int(a_rows.size), 'public_feature_bytes': (output / 'public-features.npz').stat().st_size,
        'whole_condition_bytes': sum(p.stat().st_size for p in output.iterdir() if p.is_file()),
        'wall_s': time.perf_counter() - started}
    write(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('source-root', 'b4a-root', 'source-inventory', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--folder', required=True)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    inventory = {row['path']: row['sha256'] for row in read(args.source_inventory)['files']}
    indexes = [0, 14] if args.pilot else list(range(16))
    write(args.output / 'data-contract.json', contract())
    summaries = []
    for index in indexes:
        result = one_condition(args.source_root, args.b4a_root, args.folder, index,
                               args.output / f'condition-{index:02}', inventory)
        summaries.append(result)
        print('B4B0_DATA_CONDITION_DONE', args.folder, index, result['dataset_samples'], result['public_feature_bytes'], flush=True)
    write(args.output / 'summary.json', {'folder': args.folder, 'conditions': len(summaries),
        'total_samples': sum(v['dataset_samples'] for v in summaries),
        'new_model_training': 0, 'new_physical_instances': 0,
        'all_sources_bound_to_B4A': True, 'source_conditions': [v['condition'] for v in summaries]})
    print('B4B0_DATA_DONE', args.folder, len(summaries), flush=True)


if __name__ == '__main__':
    main()
