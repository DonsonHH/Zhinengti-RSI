"""Local byte, ledger and descriptive-statistic verification of M7 delivery."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def near(left, right):
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--research', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = args.research / 'evidence'
    source_files = {}
    for name in ('source_manifest_v1.json', 'report_source_manifest_v1.json', 'package_source_manifest_v1.json'):
        source_files.update(json.loads((args.research / name).read_text())['files'])
    for name, value in source_files.items():
        require(sha((args.workspace / name).read_bytes().replace(b'\r\n', b'\n')) == value, f'Current source mismatch: {name}')
    transferred = json.loads((evidence / 'transfer-files.json').read_text())
    normalization_path = evidence / 'text-normalization.json'
    normalized = json.loads(normalization_path.read_text())['files'] if normalization_path.exists() else {}
    for name, info in transferred.items():
        raw = (evidence / name).read_bytes()
        if name in normalized:
            adjustment = normalized[name]
            require(adjustment['original_sha256'] == info['sha256'] and adjustment['original_bytes'] == info['bytes'], 'Normalization source differs from transfer')
            require(sha(raw) == adjustment['normalized_sha256'] and len(raw) == adjustment['normalized_bytes'], 'Normalized bytes differ')
            require(adjustment['xml_token_content_identical'] and name.endswith('.svg'), 'Only verified SVG whitespace normalization allowed')
        else:
            require(len(raw) == info['bytes'] and sha(raw) == info['sha256'], f'Transfer mismatch: {name}')
    audit = json.loads((evidence / 'full-chain-audit.json').read_text())
    require(audit['deployment_tasks'] == 27 and audit['method_condition_records_replayed'] == 2592, 'Full replay scope')
    groups = {}
    summaries = list(evidence.glob('*-seed-*/summary.json.gz'))
    require(len(summaries) == 27, 'Expected 27 deployment outputs')
    count = 0
    for path in summaries:
        raw = path.read_bytes()
        original = gzip.decompress(raw)
        fingerprints = json.loads((path.parent / 'summary-fingerprint.json').read_text())
        require(sha(raw) == fingerprints['stored_sha256'] and sha(original) == fingerprints['uncompressed_sha256'], 'Gzip byte fingerprints')
        require(len(original) == fingerprints['uncompressed_bytes'], 'Gzip byte length')
        summary = json.loads(original)
        require(not summary['smoke'] and not summary['training'] and not summary['checkpoint_reselection'] and not summary['rsi'], 'Deployment contract')
        require(len(summary['groups']) == 1, 'One setting per task')
        name, group = next(iter(summary['groups'].items()))
        require(len(group['cases']) == 96, 'All 96 conditions')
        for metric, mean in group['metrics'].items():
            require(near(statistics.mean(row[metric] for row in group['cases']), mean), f'Group mean differs: {metric}')
        groups.setdefault(name, {})[summary['seed']] = group['cases']
        count += len(group['cases'])
    require(count == 2592 and len(groups) == 13, 'Complete records/settings')
    report = json.loads((evidence / 'report/aggregate.json').read_text())
    for subset, table in report['tables'].items():
        for name, metric_rows in table.items():
            for metric, data in metric_rows.items():
                if not isinstance(data, dict):
                    continue
                values = []
                for seed in sorted(groups[name]):
                    cases = [row for row in groups[name][seed] if accepts(row, subset)]
                    values.append(statistics.mean(row[metric] for row in cases))
                require(len(values) == len(data['per_checkpoint_seed']) and all(near(x, y) for x, y in zip(values, data['per_checkpoint_seed'])), 'Stratified values differ')
                require(near(statistics.mean(values), data['mean']), 'Stratified mean differs')
                require(near(statistics.stdev(values), data['sample_sd_across_checkpoint_seeds']) if len(values) > 1 else data['sample_sd_across_checkpoint_seeds'] is None, 'Stratified SD differs')
    pairs = [('VAE_trained_tool_on_minus_off', 'joint_vae_tool_off', 'joint_vae'),
             ('noVAE_trained_tool_on_minus_off', 'no_vae', 'no_vae_tool_on'),
             ('joint_trained_mask_on_minus_off', 'joint_vae_mask_off', 'joint_vae'),
             ('independent_trained_mask_on_minus_off', 'independent_targets', 'independent_mask_on')]
    for effect, off, on in pairs:
        for subset, metrics in report['paired_deployment_effects'][effect].items():
            for metric, entry in metrics.items():
                values = [statistics.mean(right[metric] - left[metric]
                                           for left, right in zip(groups[off][seed], groups[on][seed]) if accepts(left, subset)) for seed in range(3)]
                require(all(near(x, y) for x, y in zip(values, entry['per_checkpoint_seed'])), 'Deployment crossover values differ')
                require(near(statistics.mean(values), entry['mean']) and near(statistics.stdev(values), entry['sample_sd_across_checkpoint_seeds']), 'Deployment crossover mean/SD differs')
    for index, witness in enumerate(report['condition_feasibility_witnesses']):
        expected = [f'{name}/seed-{seed}' for name in report['tables']['all'] for seed in sorted(groups[name])
                    if groups[name][seed][index]['target_coverage'] == 1 and groups[name][seed][index]['deadline_violation'] == 0]
        require(expected == witness['zero_violation_full_coverage_witnesses'], 'Feasibility witnesses differ')
    log = (evidence / 'jobs/smoke-129855.err').read_text()
    require('Ran 74 tests' in log and log.rstrip().endswith('OK') and 'skipped=' not in log, 'Full 74-test GPU log')
    result = {'delivery_bytes_and_statistics_verified': True, 'source_files': len(source_files),
              'transferred_files': len(transferred), 'deployment_tasks': 27, 'records': count,
              'full_test_count': 74, 'all_stratified_means_sample_sds_and_crossovers_recomputed': True,
              'svg_files_with_declared_whitespace_normalization': len(normalized),
              'feasibility_witnesses_checked': 96,
              'scope': 'Does not replace Slurm terminal verification, full physical replay or PNG visual inspection'}
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps(result, indent=2))


def accepts(row, subset):
    if subset == 'all':
        return True
    parts = subset.split('/')
    return row['family'] == parts[0] and (len(parts) == 1 or row['profile'] == parts[1])


if __name__ == '__main__':
    main()
