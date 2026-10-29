"""Portable byte, completeness, stratification and event-label checks."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import tarfile

import numpy as np


def read(path):
    if not path.exists():
        path = path.with_name(path.name + '.gz')
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def need(value, message):
    if not value:
        raise RuntimeError(message)


def same(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(v, b[k]) for k, v in a.items())
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return isinstance(b, (float, int)) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-7)
    return a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace.resolve()
    base = root / 'research/2026-10-29-m12c1b4a-information'
    ev = base / 'evidence'
    old = root / 'research/2026-10-29-m12c1b32-formal/evidence'
    protected = read(ev / 'protected-baseline.json')
    exceptions = set(protected['intentional_documentation_change_exceptions'])
    for name, row in protected['files'].items():
        if name in exceptions:
            continue
        data = (root / name).read_bytes()
        need(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], 'Old tracked bytes drift: ' + name)
    need(len((ev / 'README-at-8ce2049.md').read_bytes()) == protected['files']['README.md']['bytes'], 'Old README snapshot size')
    need(hashlib.sha256((ev / 'README-at-8ce2049.md').read_bytes()).hexdigest() == protected['files']['README.md']['sha256'], 'Old README snapshot bytes')
    latest = {}
    manifests = sorted([read(p) for p in base.glob('*-manifest.json') if 'created_utc' in read(p)], key=lambda v: v['created_utc'])
    for manifest in manifests:
        path = base / 'source' / manifest['archive']
        need(hashlib.sha256(path.read_bytes()).hexdigest() == manifest['sha256'], 'Source archive SHA')
        with tarfile.open(path) as tar:
            members = {v.name: v for v in tar.getmembers() if v.isfile()}
            need(members.keys() == manifest['files'].keys(), 'Exact archived source member set')
            for name, row in manifest['files'].items():
                data = tar.extractfile(members[name]).read()
                need(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], 'Source member SHA')
                latest[name] = row
    for name, row in latest.items():
        need(hashlib.sha256((root / name).read_bytes()).hexdigest() == row['sha256'], 'Current source not frozen: ' + name)
    transfer = read(ev / 'transfer-manifest.json')
    for name, row in transfer.items():
        data = (ev / name).read_bytes()
        need(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], 'Transferred original differs: ' + name)
    wrappers = read(ev / 'wrappers.json')
    for name, row in read(ev / 'origin-manifest.json').items():
        data = (ev / name).read_bytes()
        if name in wrappers:
            data = gzip.decompress(data)
        need(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], 'Lossless original binding: ' + name)
    report = read(ev / 'report/aggregate.json')
    conditions = read(ev / 'report/all-condition-summaries.json')
    errors = defaultdict(Counter)
    layers = defaultdict(Counter)
    fusion, cache, source_counts = Counter(), Counter(), Counter()
    source_ids = []
    candidate_pairs = actions = states = sha_frames = original_branches = 0
    frames = []
    with gzip.open(ev / 'report/all-revised-frames.jsonl.gz', 'rt', encoding='utf-8') as stream:
        frames = [json.loads(line) for line in stream]
    independent = read(ev / 'report/independent-verification.json')
    for folder, bindings in read(ev / 'original-source-bindings.json').items():
        for name, digest in bindings.items():
            path = old / folder / name
            raw = path.read_bytes() if path.exists() else gzip.decompress(path.with_name(name + '.gz').read_bytes())
            need(hashlib.sha256(raw).hexdigest() == digest, 'Original split/index/model-input binding')
    for condition in conditions:
        folder, index = condition['folder'], condition['condition_index']
        own = read(ev / 'formal' / folder / f'condition-{index:02}' / 'summary.json')
        need(same(own, {k: v for k, v in condition.items() if k not in ('folder', 'condition_index')}), 'Each condition report matches raw factor summary')
        original_index = read(old / folder / 'case-index.json')
        by_name = {v['case']: v for v in original_index}
        for source in own['source_cases']:
            need(source['case'] in by_name and source['source_sha256'] == by_name[source['case']]['sha256'] and source['source_bytes'] == by_name[source['case']]['bytes'], 'Every formal original case byte binding')
            source_ids.append(source['case'])
        process = read(old / folder / 'process-index.json')
        # The independent process gate was run on the archived arrays. Local
        # metadata is independently bound to the earlier frozen phase index.
        need(own['condition'] in [v['specification'] for v in original_index if v['alias'] == 'tour'], 'Original condition specification')
        for key, value in own['error_statistics'].items():
            errors[key].update(value)
        for alias, value in own['actual_revised_decision_layers'].items():
            layers[alias].update(value)
        fusion.update(own['thermal_fusion_unique_and_tied_positive_counts'])
        cache.update(own['thermal_raw_and_assimilated_cache_provenance'])
        source_counts.update(own['all_candidate_winner_source_counts'])
        with np.load(ev / 'formal' / folder / 'geometry.npz') as data:
            geometry = {k: data[k].copy() for k in data.files}
        need(len(geometry['task']) == 8282, 'Complete hard-legal geometry count')
        frame_count = sum(int((geometry['dispatch_time_s'] >= t).sum()) for t in range(5, 1800, 5))
        need(frame_count == own['factorized_candidate_root_pairs'], 'All factorized root/candidate pairs counted')
        windows = next(v['anomaly_intervals'] for v in original_index if v['case'] == own['source_cases'][0]['case'])
        with np.load(ev / 'formal' / folder / f'condition-{index:02}' / 'analytic-continuations.npz') as a:
            first = a['first_confirmation_s']; detected = first >= 0
            old_detected = np.asarray([v['detected'] for v in windows], dtype=bool)
            delta = (~detected).sum(1) - int((~old_detected).sum())
            need(np.array_equal(delta, a['missed_delta']), 'Independent saved-event miss sum')
            if windows:
                weights = np.asarray([v['weight'] for v in windows])
                onset = np.asarray([v['onset_s'] for v in windows]); end = np.asarray([v['end_s'] for v in windows])
                delay = np.where(detected, first - onset, end - onset)
                old_delay = np.asarray([v['delay_or_censor_s'] for v in windows])
                wait = ((delay - old_delay) * weights).sum(1) / weights.sum()
            else:
                wait = np.full(8282, np.nan)
            need(np.allclose(wait, a['weighted_wait_delta_s'], rtol=1e-10, atol=1e-8, equal_nan=True), 'Independent event-weighted delay sum')
            helpful = (delta < 0) | ((delta == 0) & (wait < -1e-9))
            harmful = (delta > 0) | ((delta == 0) & (wait > 1e-9))
            need(np.array_equal(helpful, a['helpful']) and np.array_equal(harmful, a['harmful']), 'Every saved helpful/harmful candidate label')
            need(int(helpful.sum()) == own['oracle']['net_helpful_candidates'] and int(harmful.sum()) == own['oracle']['net_harmful_candidates'], 'Condition helpful/harmful totals')
        candidate_pairs += frame_count
        actions += 8282; states += own['common_Tour_history_frames']
        sha_frames += own['original_public_SHA_frames_verified']
        original_branches += own['oracle']['executed_original_branches_verified']
    need(len(conditions) == 128 and len(set(source_ids)) == len(source_ids) == 768, 'All 128 original conditions/768 cases')
    need(report['factorized_candidate_root_pairs'] == candidate_pairs and report['hard_legal_actions_across_conditions'] == actions, 'Full candidate denominator')
    need(report['old_public_history_frames_reconstructed'] == states and report['original_public_SHA_frames_verified'] == sha_frames, 'All historical frame counts')
    need(report['actual_original_branches_verified'] == original_branches, 'All actual check branch validation')
    need(report['actual_revised_layers'] == {k: dict(v) for k, v in layers.items()}, 'All candidate decision layers')
    need(report['thermal_fusion_unique_and_tied_positive_counts'] == dict(fusion) and report['thermal_raw_and_assimilated_cache_provenance'] == dict(cache), 'Fusion ties and cache counts')
    for key, row in errors.items():
        actual = report['error_statistics'][key]
        need(all(same(actual[k], v) for k, v in row.items()), 'All professional/horizon/source error sums')
        need(same(actual['MAE'], row['sum_absolute_error'] / row['n']) and same(actual['mean_error'], row['sum_error'] / row['n']), 'All source forecast error ratios')
    for field, strata in report['actual_revised_strata'].items():
        for value, summary in strata.items():
            own = [v for v in frames if str(v['specification'][field]) == value]
            need(summary['layers'] == dict(Counter(v['layer'] for v in own)), 'All phase/seed/scenario/start/noise layers')
    need(independent['independent_verification'] == 'passed' and independent['counts']['analytic_actions'] == actions and independent['counts']['public_frames'] == states, 'Full remote independent event/history verifier receipt')
    gate = read(ev / 'final-gate/full-tests.json')
    need(gate['success'] and gate['tests'] >= 230 and not(gate['skipped'] or gate['failures'] or gate['errors']), 'Final real Torch suite')
    costs = read(ev / 'slurm-cost-review.json')
    from review_information_resources import review
    need(same(costs, review(ev / 'slurm-final.psv')), 'Independent complete CPU/Slurm resource arithmetic')
    need(costs['all_frozen_resource_caps_verified'], 'Final resource accounting/caps')
    receipt = {'portable_verification': 'passed', 'protected_old_files': len(protected['files']) - len(exceptions),
               'preserved_old_README_snapshot': True, 'source_archives': len(manifests), 'transferred_original_files': len(transfer),
               'original_cases_bound': len(source_ids), 'old_conditions': len(conditions), 'public_history_frames': states,
               'original_public_SHA_frames': sha_frames, 'hard_legal_actions': actions, 'candidate_root_pairs': candidate_pairs,
               'original_executed_branch_records': original_branches, 'final_actual_Torch_tests': gate['tests'],
               'all_event_label_and_statistics_sums_recomputed': True, 'all_strata_and_fusion_counts_verified': True,
               'allocated_CPU_minutes': costs['allocated_CPU_minutes'],
               'scope': 'Local complete compact-factor/source/statistics verification. All 128 full histories and direct future measurements were independently checked remotely; local full-history originals retained for two fixed examples.'}
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
