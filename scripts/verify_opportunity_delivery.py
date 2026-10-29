"""Portable B0 source, all A/B labels, process/event counts and protocol audit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

from patrol_rsi.causal_observation import CHANNELS
from patrol_rsi.opportunity_data import FEATURE_NAMES, HORIZONS, KINDS, contract


COUNT_FIELDS = ('total_samples', 'A_valid', 'A_positive', 'A_negative',
    'B_reference_valid', 'B_reference_positive', 'B_action_valid',
    'B_action_positive', 'B_action_negative', 'B_action_invalid',
    'physical_event_count', 'retained_event_count')


def read(path):
    if not path.exists():
        path = path.with_name(path.name + '.gz')
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def need(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace.resolve()
    base = root / 'research/2026-10-30-m12c1b4b0-data'
    ev = base / 'evidence'
    old = root / 'research/2026-10-29-m12c1b4a-information/evidence'
    original = root / 'research/2026-10-29-m12c1b32-formal/evidence'
    protected = read(ev / 'protected-baseline.json')
    exceptions = protected['intentional_entry_document_exceptions']
    for name, row in protected['files'].items():
        p = root / name
        if name in exceptions:
            p = ev / ('baseline-' + p.name)
        raw = p.read_bytes()
        need(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'Original baseline drift: ' + name)
    manifests = sorted([read(p) for p in base.glob('*-manifest.json') if 'created_utc' in read(p)], key=lambda v: v['created_utc'])
    latest = {}
    for manifest in manifests:
        p = base / 'source' / manifest['archive']
        need(digest(p) == manifest['sha256'], 'Frozen source archive bytes')
        with tarfile.open(p) as tar:
            members = {v.name: v for v in tar.getmembers() if v.isfile()}
            need(set(members) == set(manifest['files']), 'Exact source archive members')
            for name, row in manifest['files'].items():
                raw = tar.extractfile(members[name]).read()
                need(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'Frozen source member bytes')
                latest[name] = row
    for name, row in latest.items():
        need(digest(root / name) == row['sha256'], 'Current source not frozen: ' + name)
    transfer, wrappers = read(ev / 'transfer-manifest.json'), read(ev / 'wrappers.json')
    for name, row in transfer.items():
        raw = (ev / name).read_bytes()
        need(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'Transfer bytes: ' + name)
    for name, row in read(ev / 'origin-manifest.json').items():
        raw = (ev / name).read_bytes()
        raw = gzip.decompress(raw) if name in wrappers else raw
        need(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'Lossless original bytes: ' + name)
    report = read(ev / 'report/coverage.json')
    conditions = read(ev / 'report/condition-summaries.json')
    expected_counts = defaultdict(Counter)
    expected_sets = defaultdict(lambda: defaultdict(set))
    expected_channels = defaultdict(lambda: defaultdict(Counter))
    kind_ids = np.r_[np.zeros(128, dtype=int), np.ones(16, dtype=int), np.full(16, 2), np.full(16, 3)]
    roots = np.arange(5, 1800, 5)
    with np.load(ev / 'full/development-0/condition-00/public-features.npz') as data:
        template = data['base'][:, :, [FEATURE_NAMES.index(c + '.available') for c in CHANNELS]].astype(bool)
    named = defaultdict(list)
    all_conditions = set()
    timestamp_count = samples = events_checked = 0
    for item in conditions:
        folder, number = item['folder'], item['condition_index']
        need((folder, number) not in all_conditions, 'Repeated source condition')
        all_conditions.add((folder, number))
        path = ev / 'full' / folder / f'condition-{number:02}'
        summary = read(path / 'summary.json')
        need({k: v for k, v in item.items() if k not in ('folder', 'condition_index')} == summary, 'Each condition summary matches its source')
        identity = read(path / 'source-identity-not-features.json')
        prior = read(old / 'formal' / folder / f'condition-{number:02}/summary.json')
        need(identity['condition'] == prior['condition'] == summary['condition'], 'B4-A condition identity')
        source_spec = identity['condition']; fp = identity['original_numeric_process_fingerprint']
        need(identity['base_feature_names'] == list(FEATURE_NAMES) and identity['training_split'] == 'diagnostic_old_only', 'Closed causal features and old-data status')
        tour = next(v for v in prior['source_cases'] if v['case'].endswith('--tour'))
        need(identity['source_Tour_sha256'] == tour['source_sha256'], 'B4-A anchored original Tour SHA')
        process = read(original / folder / 'process-index.json')[identity['original_process_key']]
        need(fp == process['numeric_digest'] and identity['source_process_sha256'] == process['sha256'], 'Original physical process identity')
        c_file = old / 'formal' / folder / f'condition-{number:02}/analytic-continuations.npz'
        need(identity['C_value_label_reference_only']['sha256'] == digest(c_file), 'Independent unchanged C-label sidecar')
        need(identity['source_B4A_history_sha256'] == next(v['sha256'] for v in read(old / 'raw-storage.json')['files']
            if v['path'].endswith('/' + folder + f'/condition-{number:02}/public-states.jsonl.gz')), 'Original public history SHA')
        with np.load(path / 'labels-only.npz') as data:
            labels = {k: data[k].copy() for k in data.files}
        with np.load(path / 'targets-evaluator-only.npz') as data:
            targets = {k: data[k].copy() for k in data.files}
        need(set(labels) == {'A_physical', 'B_reference', 'event_id', 'retained_event_id'}, 'No C utility mixed into A/B label table')
        need(set(targets) == {'A', 'B_reference', 'B_action', 'action_geometry_index', 'legal_action_count'}, 'Closed sample targets')
        need(all(v.shape == (360, 176) for v in labels.values()) and all(v.shape == (359, 3, 176) for v in targets.values()), 'All label dimensions')
        need(not np.any(labels['B_reference'] & ~labels['A_physical']), 'Qualified confirmation requires physical anomaly')
        events = read(path / 'events-evaluator-only.json')
        bank = {v['id']: v for v in events}
        reconstructed_event = np.zeros((360, 176), dtype=np.int32)
        reconstructed_retained = np.zeros_like(reconstructed_event)
        for event in events:
            first, last, task = event['first_tick'], event['last_exclusive_tick'], event['task']
            need(labels['A_physical'][first:last, task].all() and (first == 0 or not labels['A_physical'][first - 1, task]) and
                 (last == 360 or not labels['A_physical'][last, task]), 'Independent maximal event boundaries')
            need(event['retained_at_least_15s'] == (last - first >= 3), 'Physical positive run versus retained event')
            need(event['confirmable_positive_bins'] == int(labels['B_reference'][first:last, task].sum()), 'Event measured-confirmable bins')
            reconstructed_event[first:last, task] = event['id']
            if last - first >= 3:
                reconstructed_retained[first:last, task] = event['id']
        need(np.array_equal(reconstructed_event, labels['event_id']) and np.array_equal(reconstructed_retained, labels['retained_event_id']), 'Every raw/retained event membership')
        need(np.array_equal(labels['event_id'] > 0, labels['A_physical']), 'All physical positive bins have events')
        geometry_path = old / 'formal' / folder / 'geometry.npz'
        with np.load(geometry_path) as data:
            geometry = {k: data[k].copy() for k in data.files}
        if (path / 'public-features.npz').exists():
            with np.load(path / 'public-features.npz') as data:
                need(set(data.files) == {'base', 'completion'}, 'Public file cannot contain future labels')
                public = data['base']; action = data['completion']
                need(np.array_equal(public[:, :, [FEATURE_NAMES.index(c + '.available') for c in CHANNELS]].astype(bool), template),
                     'Known Tour/telemetry availability template differs')
                need(np.isfinite(public).all() and np.isfinite(action).all(), 'Finite delivered public matrices')
        audit = summary['feature_audit']
        need(audit['future_timestamp_violations'] == 0 and audit['closed_feature_schema_passed'], 'Full original feature causality audit')
        timestamp_count += audit['timestamp_available_readings_checked']
        for k, kind in enumerate(KINDS):
            task_mask = kind_ids == k
            for c, channel in enumerate(CHANNELS):
                value = audit['channel_availability_by_kind'][kind][channel]
                available = int(template[:, task_mask, c].sum())
                need(value == {'available': available, 'total': 359 * int(task_mask.sum()),
                    'missing': 359 * int(task_mask.sum()) - available}, 'All-root per-profession sensor availability')
            for h, horizon in enumerate(HORIZONS):
                end = roots + horizon; in_mission = end <= 1800
                tick = np.minimum(359, end // 5 - 1)
                expected_a = np.where(in_mission[:, None], labels['A_physical'][tick], -1).astype(np.int8)
                expected_ref = np.where(in_mission[:, None], labels['B_reference'][tick], -1).astype(np.int8)
                need(np.array_equal(targets['A'][:, h], expected_a) and np.array_equal(targets['B_reference'][:, h], expected_ref), 'All future-endpoint A/B reference labels')
                legal = targets['legal_action_count'][:, h] > 0
                ids = targets['action_geometry_index'][:, h]
                need(np.array_equal(legal, ids >= -1) and not legal[~in_mission].any(), 'Missing completion versus true negative')
                need(np.array_equal(targets['B_action'][:, h], np.where(legal, expected_ref, -1)), 'All B action labels preserve validity')
                optional = ids >= 0
                need((ids[optional] < len(geometry['task'])).all(), 'Legal C join range')
                rr, tt = np.nonzero(optional); gi = ids[optional]
                need(np.array_equal(geometry['task'][gi], tt) and np.array_equal(geometry['finish_s'][gi], end[rr]) and
                     (geometry['dispatch_time_s'][gi] >= roots[rr]).all(), 'C join references exact legal future action, not model feature')
                a = targets['A'][:, h, task_mask]; reference = targets['B_reference'][:, h, task_mask]
                b = targets['B_action'][:, h, task_mask]
                ev_id = labels['event_id'][tick][:, task_mask]
                retained_id = labels['retained_event_id'][tick][:, task_mask]
                own_events = [v for v in events if v['kind'] == kind]
                event_ids = sorted(int(v) for v in np.unique(ev_id[b == 1]) if v > 0)
                retained_ids = sorted(int(v) for v in np.unique(retained_id[b == 1]) if v > 0)
                counts = {'total_samples': int(a.size), 'A_valid': int((a >= 0).sum()), 'A_positive': int((a == 1).sum()),
                    'A_negative': int((a == 0).sum()), 'B_reference_valid': int((reference >= 0).sum()),
                    'B_reference_positive': int((reference == 1).sum()), 'B_action_valid': int((b >= 0).sum()),
                    'B_action_positive': int((b == 1).sum()), 'B_action_negative': int((b == 0).sum()),
                    'B_action_invalid': int((b < 0).sum()), 'physical_event_count': len(own_events),
                    'retained_event_count': sum(v['retained_at_least_15s'] for v in own_events)}
                source_row = next(v for v in summary['strata'] if v['kind'] == kind and v['horizon_s'] == horizon)
                need(all(source_row[key] == value for key, value in counts.items()) and source_row['B_positive_event_ids'] == event_ids and
                     source_row['B_positive_retained_event_ids'] == retained_ids, 'Every profession/horizon positive/negative and unique-event count')
                sensor_counts = {}
                for c, channel in enumerate(CHANNELS):
                    available = template[:, task_mask, c]
                    sensor_counts[channel] = {'B_valid_available': int((available & (b >= 0)).sum()),
                                              'B_valid_missing': int((~available & (b >= 0)).sum())}
                need(sensor_counts == source_row['availability_in_valid_B_samples'], 'Every valid-B sensor/missingness count')
                keys = [('profession_horizon', kind + '/' + str(horizon)),
                        ('numeric_process', fp + '/' + kind + '/' + str(horizon))]
                keys += [(name, str(source_spec[name]) + '/' + kind + '/' + str(horizon))
                         for name in ('seed', 'scenario', 'fault_start_s', 'noise_c')]
                def event_key(i):
                    e = bank[i]
                    return (fp, e['task'], e['first_tick'], e['last_exclusive_tick'])
                for dimension, key in keys:
                    group = (dimension, key)
                    expected_counts[group].update(counts)
                    sets = expected_sets[group]
                    sets['seeds'].add(source_spec['seed']); sets['numeric'].add(fp)
                    if counts['B_action_positive']:
                        sets['positive_seeds'].add(source_spec['seed']); sets['positive_numeric'].add(fp)
                    sets['physical_events'].update(event_key(v['id']) for v in own_events)
                    sets['positive_events'].update(event_key(i) for i in event_ids)
                    sets['retained_positive_events'].update(event_key(i) for i in retained_ids)
                    for channel, value in sensor_counts.items():
                        expected_channels[group][channel].update(value)
        physical_key = (source_spec['seed'], source_spec['scenario'], source_spec['duration_s'], source_spec['fault_start_s'])
        named[physical_key].append((source_spec['noise_c'], fp, digest(path / 'labels-only.npz'), digest(path / 'targets-evaluator-only.npz')))
        samples += int(targets['A'].size); events_checked += len(events)
        need(read(ev / 'full' / folder / 'data-contract.json') == contract(), 'Dataset feature/target contract')
    need(len(all_conditions) == 128 and len(named) == 64 and samples == report['total_sample_rows'] == 24262656, 'Full old-data inventory')
    for pair in named.values():
        need(len(pair) == 2 and {v[0] for v in pair} == {.2, 1.} and len({v[1:] for v in pair}) == 1,
             'Both noise versions share physical and A/B labels, not independent events')
    for group, count in expected_counts.items():
        dimension, key = group; value = report['coverage'][dimension][key]; sets = expected_sets[group]
        need(all(value[k] == v for k, v in count.items()), 'Every aggregate count stratum')
        need(value['horizon_s'] == int(key.rsplit('/', 1)[-1]), 'Horizon metadata is not a sum')
        need(value['seed_groups'] == sorted(sets['seeds']) and value['numeric_processes'] == len(sets['numeric']), 'Independent physical grouping')
        need(value['B_positive_seed_groups'] == sorted(sets['positive_seeds']) and value['B_positive_numeric_processes'] == len(sets['positive_numeric']), 'Independent positive process coverage')
        need(value['distinct_physical_events'] == len(sets['physical_events']) and value['distinct_B_positive_events'] == len(sets['positive_events']) and
             value['distinct_B_positive_retained_events'] == len(sets['retained_positive_events']), 'No repeated timestamp/noise event inflation')
        need(value['available_sensor_counts_in_valid_B'] == {k: dict(v) for k, v in expected_channels[group].items()}, 'All sensor availability strata')
    independent = read(ev / 'report/independent-verification.json')
    need(independent['status'] == 'passed' and independent['conditions'] == 128 and independent['counts']['samples'] == samples and
         independent['counts']['available_readings'] == timestamp_count == report['causal_available_readings_checked'], 'Full remote causality/label verification receipt')
    need(independent['counts']['public_base_values_verified'] == 128 * 359 * 176 * 85 and
         independent['counts']['public_action_values_verified'] == 128 * 359 * 3 * 176 * 13, 'All public feature columns independently verified')
    protocol_path = base / 'B4-B1冻结协议.json'; protocol = read(protocol_path)
    freeze = read(ev / 'B4-B1-design-freeze-final.json')
    need(digest(protocol_path) == freeze['sha256'] and digest(base / freeze['fresh_manifest']) == freeze['fresh_manifest_sha256'], 'Frozen B4-B1 protocol and new-process manifest')
    groups = protocol['new_seed_groups']; all_new = [v for values in groups.values() for v in values]
    need(len(set(all_new)) == len(all_new) == 64 and not (set(all_new) & set(report['seed_groups'])), 'Disjoint fresh seed groups, no old test reuse')
    need(not read(ev / 'new-seed-collision-review.json')['collisions'] and freeze['fresh_data_generated'] is False and freeze['new_predictor_training'] == 0, 'No unverified fresh-data GO')
    need(protocol['candidate_budget']['optional_small_temporal']['enabled'] is False and
         protocol['candidate_budget']['optional_small_temporal']['training_budget'] == 0, 'Optional temporal model not silently enabled')
    gate = read(ev / 'final/gate/full-tests.json')
    need(gate['tests'] >= 237 and gate['success'] and not(gate['skipped'] or gate['failures'] or gate['errors']), 'Real Torch complete gate')
    from review_opportunity_resources import review
    costs = review(ev / 'slurm-final.psv')
    need(costs == read(ev / 'slurm-cost-review.json'), 'Independent resource accounting')
    receipt = {'portable_verification': 'passed', 'protected_old_files': len(protected['files']) - len(exceptions),
        'old_entry_snapshots_preserved': len(exceptions), 'source_archives': len(manifests),
        'transferred_files': len(transfer), 'conditions': len(all_conditions), 'sample_rows': samples,
        'named_processes': len(named), 'numeric_processes': report['numeric_process_fingerprints'],
        'all_noise_pair_labels_identical': True, 'all_profession_horizon_scenario_process_strata_verified': True,
        'timestamp_readings_checked': timestamp_count, 'event_occurrences_including_noise': events_checked,
        'C_labels_kept_independent': True, 'fresh_protocol_and_seed_manifest_frozen': True,
        'actual_Torch_tests': gate['tests'], 'allocated_CPU_minutes': costs['allocated_CPU_minutes'],
        'training_gate': 'NO-GO',
        'scope': 'All delivered A/B matrices, event/group/count/availability strata and byte sources verified. All128 full public matrices were independently checked remotely; two full public matrices delivered locally. No predictor performance claim.'}
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
