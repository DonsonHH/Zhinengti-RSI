"""Verify delivered M6 files; this does not replace GPU or trajectory audits."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

from m6_evidence_io import read_json


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def close(left, right):
    return math.isclose(left, right, rel_tol=1e-11, abs_tol=1e-12)


def rank(score):
    return tuple(round(float(value), 6) for value in
                 (1 - score['target_coverage'], score['deadline_violation'], score['weighted_idleness']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--research', type=Path, required=True)
    parser.add_argument('--private-checkpoints', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = args.research / 'evidence'
    expected = {}
    roles = ('source_manifest_v2_atomic.json', 'final_review_source_manifest_v2.json',
             'package_source_manifest_v2.json')
    for name in roles:
        expected.update(read_json(args.research / name)['files'])
    for name, fingerprint in expected.items():
        data = (args.workspace / name).read_bytes().replace(b'\r\n', b'\n')
        require(sha(data) == fingerprint, f'Current source mismatch: {name}')
    storage = read_json(evidence / 'evidence-storage-manifest.json')['summaries']
    for row in storage:
        stored = (evidence / row['delivered']).read_bytes()
        raw = gzip.decompress(stored) if row['storage'] == 'lossless_gzip' else stored
        require(len(stored) == row['stored_bytes'] and sha(stored) == row['stored_sha256'], 'Stored bytes mismatch')
        require(len(raw) == row['original_bytes'] and sha(raw) == row['uncompressed_sha256'], 'Original bytes mismatch')
    require(len(storage) == 9, 'Expected nine lossless summaries')
    with (evidence / 'slurm-outcomes-final.psv').open() as handle:
        jobs = {row['JobID']: row for row in csv.DictReader(handle, delimiter='|')}
    required = ['129810', '129811', '129812', '129814', '129822', '129824', '129837', '129839', '129850']
    required += [f'129813_{index}' for index in range(9)]
    for job in required:
        require(jobs[job]['State'] == 'COMPLETED' and jobs[job]['ExitCode'] == '0:0', f'Job incomplete: {job}')
    require(jobs['129830']['State'] == 'FAILED' and jobs['129830']['ExitCode'] == '1:0', 'Packaging failure ledger changed')
    test_log = (evidence / 'jobs/smoke-129837.err').read_text()
    require('Ran 66 tests' in test_log and test_log.rstrip().endswith('OK') and 'skipped=' not in test_log, 'Full test log mismatch')
    preflight = read_json(evidence / 'preflight-129812/summary.json')
    require(preflight['preflight_improved_vs_initial'] and not preflight['test'], 'Preflight gate failed')
    require(preflight['episodes'] == 256 and preflight['selected_episode'] == 192, 'Preflight changed')
    audit = read_json(evidence / 'post-storage-atomic-audit.json')
    require(audit == read_json(evidence / 'full-chain-audit-v2.json') == read_json(evidence / 'full-chain-audit.json'), 'Replay audit differs')
    require(audit['test_method_case_records'] == 2160 and audit['distinct_case_and_target_choice_sequences_replayed'] == 786, 'Replay scope changed')
    require(read_json(evidence / 'selected-weight-audit.json') == read_json(evidence / 'selected-weight-audit-v2.json'), 'Weight audit differs')
    weights = read_json(evidence / 'checkpoint-fingerprints.json')
    for name, info in weights.items():
        path = args.private_checkpoints / Path(name).parent.name / 'best-policy.pt'
        raw = path.read_bytes()
        require(len(raw) == info['bytes'] and sha(raw) == info['sha256'], f'Local weight mismatch: {name}')
    summaries = {}
    for variant in ('joint_vae', 'no_vae', 'independent_targets'):
        summaries[variant] = []
        for seed in range(3):
            directory = evidence / f'{variant}-seed-{seed}'
            run = read_json(directory / 'summary.json')
            summaries[variant].append(run)
            require(run['episodes'] == 1200 and run['gradient_updates'] == 3600, 'Training budget changed')
            require(run['parameters'] == 17666 and run['initial_parameter_sha256'] != run['selected_parameter_sha256'], 'Parameters did not update')
            require(not run['test_used_for_selection'] and run['fresh_checkpoint_trace_replay_matches'], 'Selection/reload evidence failed')
            require(run['behavior_logp_max_error'] <= 2e-5 and not run['rsi_enabled'], 'Likelihood or RSI contract changed')
            curve = [json.loads(line) for line in (directory / 'learning.jsonl').read_text().splitlines()]
            best = min(curve, key=lambda row: rank(row['validation']))
            require(best['episode'] == run['selected_episode'] and best['validation'] == run['best_validation'], 'Validation selection differs')
            require(curve[-1]['episode'] == 1200 and len(curve) == 12, 'Incomplete learning curve')
            for method, result in run['test'].items():
                require(len(result['cases']) == 48, f'Incomplete conditions: {variant}/{method}')
                for metric, value in result['metrics'].items():
                    require(close(statistics.mean(row[metric] for row in result['cases']), value), f'Case aggregation mismatch: {metric}')
    for seed in range(3):
        require(len({runs[seed]['initial_parameter_sha256'] for runs in summaries.values()}) == 1, 'Initial seeds not paired')
    aggregate = read_json(evidence / 'report/aggregate.json')
    for subset, groups in aggregate['table'].items():
        for group, metrics in groups.items():
            variant, method = group.split('/')
            for metric, entry in metrics.items():
                values = []
                for run in summaries[variant]:
                    rows = run['test'][method]['cases']
                    if subset == 'paired_noise_difference':
                        clean = {(row['layout_seed'], row['profile']): row for row in rows if row['noise'] == 0}
                        values.append(statistics.mean(row[metric] - clean[row['layout_seed'], row['profile']][metric] for row in rows if row['noise'] > 0))
                    elif subset == 'paired_profile_difference':
                        balanced = {(row['layout_seed'], row['noise']): row for row in rows if row['profile'] == 'balanced'}
                        values.append(statistics.mean(row[metric] - balanced[row['layout_seed'], row['noise']][metric] for row in rows if row['profile'] == 'tight'))
                    else:
                        values.append(statistics.mean(row[metric] for row in rows if subset == 'all' or row['profile'] == subset))
                require(all(close(left, right) for left, right in zip(values, entry['per_training_seed'])), f'Aggregate values differ: {subset}/{group}/{metric}')
                require(close(statistics.mean(values), entry['mean']) and close(statistics.stdev(values), entry['sample_sd_across_training_seeds']), 'Mean/SD differs')
    report = {'artifact_checks_passed': True, 'replaced_gpu_or_trajectory_audit': False,
              'current_source_files_verified_with_role_overrides': len(expected),
              'lossless_summary_byte_pairs_verified': len(storage),
              'required_successful_slurm_jobs_verified': len(required),
              'preserved_initial_packaging_failure': '129830',
              'local_weight_files_sha256_verified': len(weights), 'full_tests_in_gpu_log': 66,
              'test_records_in_referenced_complete_replay': 2160,
              'all_aggregate_subsets_means_and_sample_sds_recomputed': True,
              'roles_in_order_of_override': list(roles)}
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
