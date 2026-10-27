"""Local M8 byte, accounting, weight and factor-statistic verification."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def near(left, right):
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


def accepts(row, subset):
    parts = subset.split('/')
    return subset == 'all' or (row['family'] == parts[0] and (len(parts) == 1 or row['profile'] == parts[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--research', type=Path, required=True)
    parser.add_argument('--private-checkpoints', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = args.research / 'evidence'
    sources = {}
    for name in ('source_manifest_v1.json', 'report_source_manifest_v1.json', 'package_source_manifest_v1.json'):
        sources.update(json.loads((args.research / name).read_text())['files'])
    for name, fingerprint in sources.items():
        require(sha((args.workspace / name).read_bytes().replace(b'\r\n', b'\n')) == fingerprint, f'Frozen source mismatch: {name}')
    transfer = json.loads((evidence / 'transfer-files.json').read_text())
    adjustments = evidence / 'text-normalization.json'
    normalized = json.loads(adjustments.read_text())['files'] if adjustments.exists() else {}
    for name, info in transfer.items():
        raw = (evidence / name).read_bytes()
        if name in normalized:
            row = normalized[name]
            require(row['original_sha256'] == info['sha256'] and row['original_bytes'] == info['bytes'], 'Original SVG source fingerprint')
            require(name.endswith('.svg') and row['xml_token_content_identical'], 'Only verified SVG whitespace changes allowed')
            require(sha(raw) == row['normalized_sha256'] and len(raw) == row['normalized_bytes'], 'Normalized SVG fingerprint')
        else:
            require(len(raw) == info['bytes'] and sha(raw) == info['sha256'], f'Transferred file mismatch: {name}')
    with (evidence / 'slurm-outcomes-final.psv').open() as handle:
        jobs = {row['JobID']: row for row in csv.DictReader(handle, delimiter='|')}
    handle_ids = json.loads((evidence / 'job-identities.json').read_text())
    required = [str(handle_ids[key]) for key in ('review', 'smoke', 'planner', 'report', 'package')]
    required += [f"{handle_ids['preflight']}_{index}" for index in range(2)]
    required += [f"{handle_ids['main_array']}_{index}" for index in range(12)]
    for job in required:
        require(jobs[job]['State'] == 'COMPLETED' and jobs[job]['ExitCode'] == '0:0', f'Job incomplete: {job}')
    test_log = (evidence / f"jobs/smoke-{handle_ids['smoke']}.err").read_text()
    require('Ran 82 tests' in test_log and test_log.rstrip().endswith('OK') and 'skipped=' not in test_log, 'Full 82-test log')
    require(json.loads((evidence / 'development-gate.json').read_text())['passed'], 'Development gate')
    weights = json.loads((evidence / 'checkpoint-fingerprints.json').read_text())
    for name, info in weights.items():
        raw = (args.private_checkpoints / name / 'best-policy.pt').read_bytes()
        require(len(raw) == info['bytes'] and sha(raw) == info['sha256'], f'Actual local weight fingerprint {name}')
    audit = json.loads((evidence / 'full-chain-audit.json').read_text())
    require(audit['from_scratch_training_tasks'] == 12 and audit['method_condition_records_replayed'] == 2880, 'Full replay scope')
    require(json.loads((evidence / 'actual-weight-audit.json').read_text())['actual_selected_weights_read'] == 12, 'Actual selected-weight audit')
    groups = {}
    count = 0
    for directory in sorted(evidence.iterdir()):
        if not directory.is_dir() or not (directory / 'summary.json.gz').exists():
            continue
        stored = (directory / 'summary.json.gz').read_bytes()
        raw = gzip.decompress(stored)
        fingerprint = json.loads((directory / 'summary-fingerprint.json').read_text())
        require(sha(stored) == fingerprint['stored_sha256'] and len(stored) == fingerprint['stored_bytes'], 'Stored gzip fingerprint')
        require(sha(raw) == fingerprint['uncompressed_sha256'] and len(raw) == fingerprint['uncompressed_bytes'], 'Original JSON fingerprint')
        summary = json.loads(raw)
        if summary.get('smoke') or summary.get('preflight'):
            require(not summary['test'] and summary['fresh_checkpoint_trace_replay_matches'], 'Preflight final-test isolation')
            continue
        if summary['planners']:
            for method, group in summary['test'].items():
                groups[f'planner/{method}'] = {0: group['cases']}
                count += len(group['cases'])
        else:
            require(summary['episodes'] == 1200 and summary['gradient_updates'] == 3600 and summary['parameters'] == 22594, 'Matched budget')
            require(summary['fresh_checkpoint_trace_replay_matches'] and not summary['test_used_for_selection'] and not summary['rsi'], 'Reload/selection contract')
            for method, group in summary['test'].items():
                groups.setdefault(f"{summary['variant']}/{method}", {})[summary['seed']] = group['cases']
                count += len(group['cases'])
        for group in summary['test'].values():
            require(len(group['cases']) == 96, '96 conditions')
            for metric, value in group['metrics'].items():
                require(near(statistics.mean(row[metric] for row in group['cases']), value), 'Method means')
    require(count == 2880 and len(groups) == 14, 'Complete groups and records')
    report = json.loads((evidence / 'report/aggregate.json').read_text())
    for subset, table in report['tables'].items():
        for name, metric_rows in table.items():
            for metric, entry in metric_rows.items():
                if not isinstance(entry, dict):
                    continue
                values = [statistics.mean(row[metric] for row in groups[name][seed] if accepts(row, subset)) for seed in sorted(groups[name])]
                require(len(values) == len(entry['per_training_seed']) and all(near(x, y) for x, y in zip(values, entry['per_training_seed'])), 'Stratified seed values')
                require(near(statistics.mean(values), entry['mean']), 'Stratified mean')
                require(near(statistics.stdev(values), entry['sample_sd_across_training_seeds']) if len(values) > 1 else entry['sample_sd_across_training_seeds'] is None, 'Stratified sample SD')
    pairs = [('mixed_minus_rack_local', 'rack_local', 'mixed_local'), ('mixed_minus_rack_relation', 'rack_relation', 'mixed_relation'),
             ('relation_minus_local_rack', 'rack_local', 'rack_relation'), ('relation_minus_local_mixed', 'mixed_local', 'mixed_relation')]
    for effect, before, after in pairs:
        for subset, metrics in report['paired_factor_effects'][effect].items():
            for metric, entry in metrics.items():
                values = [statistics.mean(right[metric] - left[metric] for left, right in zip(groups[f'{before}/learned'][seed], groups[f'{after}/learned'][seed])
                                           if accepts(left, subset)) for seed in range(3)]
                require(all(near(x, y) for x, y in zip(values, entry['per_training_seed'])), 'Paired factor values')
                require(near(statistics.mean(values), entry['mean']) and near(statistics.stdev(values), entry['sample_sd_across_training_seeds']), 'Factor mean/SD')
    for subset, metrics in report['interaction'].items():
        for metric, entry in metrics.items():
            values = [report['paired_factor_effects']['relation_minus_local_mixed'][subset][metric]['per_training_seed'][seed]
                      - report['paired_factor_effects']['relation_minus_local_rack'][subset][metric]['per_training_seed'][seed] for seed in range(3)]
            require(all(near(x, y) for x, y in zip(values, entry['per_training_seed'])), 'Interaction values')
            require(near(statistics.mean(values), entry['mean']) and near(statistics.stdev(values), entry['sample_sd_across_training_seeds']), 'Interaction mean/SD')
    result = {'bytes_and_statistics_verified': True, 'frozen_source_files': len(sources), 'transferred_files': len(transfer),
              'required_completed_jobs': len(required), 'private_weight_files_locally_verified': len(weights),
              'selected_weight_files_in_independent_CPU_audit': 12, 'full_tests': 82, 'test_records': count,
              'all_stratified_means_sds_and_paired_factor_interactions_recomputed': True,
              'scope': 'Complements full physical replay, actual-weight CPU replay and separate PNG visual review'}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
