"""Portable archive/transfer/budget/pairing/aggregate verification for M12-B."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import tarfile


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def close(a, b):
    return a is None and b is None or a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--research', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root, base = args.workspace, args.research
    ev = base / 'evidence'
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    manifests = sorted(base.glob('*source_manifest_v*.json'))
    latest = {}
    for path in manifests:
        prefix, version = path.stem.rsplit('_v', 1)
        latest[prefix] = max(latest.get(prefix, 0), int(version))
    current = 0
    for path in manifests:
        value = read(path)
        archive = base / 'source' / path.name.replace('_manifest_', '-').replace('_', '-').replace('.json', '.tar.gz')
        need(sha(archive) == value['archive_sha256'], 'Archive bytes ' + archive.name)
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            need({m.name for m in members} == set(value['files']), 'Archive members')
            for m in members:
                need(hashlib.sha256(tar.extractfile(m).read()).hexdigest() == value['files'][m.name], 'Archived file bytes')
        prefix, version = path.stem.rsplit('_v', 1)
        if int(version) == latest[prefix]:
            for rel, h in value['files'].items():
                need(hashlib.sha256((root / rel).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == h, 'Current source drift ' + rel)
                current += 1
    transfer = read(ev / 'transfer-manifest.json')
    for rel, r in transfer.items():
        path = ev / rel
        need(path.stat().st_size == r['bytes'] and sha(path) == r['sha256'], 'Original transferred file ' + rel)
    with (ev / 'slurm-final.psv').open() as f:
        jobs = {r['JobID']: r for r in csv.DictReader(f, delimiter='|')}
    ids = read(ev / 'job-identities.json')
    checked_jobs = []
    for key, value in ids.items():
        targets = [f'{value}_{i}' for i in range(3)] if key.endswith('_array') else [str(value)]
        for job in targets:
            need(jobs[job]['State'] == 'COMPLETED' and jobs[job]['ExitCode'] == '0:0', 'Terminal job ' + job)
            checked_jobs.append(job)
    log = (ev / f"jobs/smoke-v3-{ids['smoke_v3']}.err").read_text()
    need('Ran 152 tests' in log and '\nOK\n' in log and 'skipped' not in log, '152complete tests no skips')
    gate = read(ev / 'development-gate.json')
    need(gate['passed'] and not gate['formal_test_read'] and gate['raw_workflow_and_protocol_responses_valid'], 'Development gate')
    smoke_audit = read(ev / 'smoke-audit.json')
    need(smoke_audit['all_actual_cases_replayed'] == 8 and smoke_audit['interface_model_calls'] == 5, 'Actual interface replay')
    compact = json.loads(gzip.decompress((ev / 'report/compact-metrics.json.gz').read_bytes()))
    aggregate = read(ev / 'report/aggregate.json')
    total_cases, total_calls, test_refs, task_slots = 0, 0, 0, 0
    keys_across_trials, physical_keys = set(), set()
    for trial in range(3):
        folder = ev / f'trial-{trial}'
        summary, index = read(folder / 'summary.json'), read(folder / 'case-index.json')
        audit = read(ev / f'audit-{trial}.json')
        need(not summary['smoke'] and summary['actual_model_calls'] == audit['actual_model_calls'] == 39, 'Formal call count')
        need(summary['unique_cases'] == len(index) == audit['all_actual_cases_replayed'], 'Whole-case replay scope')
        need(audit['task_candidate_slots'] == 50 and audit['all_meta_arms_reset_and_paired']
            and audit['only_protocol_transferred_to_fresh_tasks'] and audit['all_test_selections_frozen_before_test_execution'], 'Meta/transfer boundaries')
        need(audit['maximum_exact_gap_ratio'] <= 1 and audit['quota_cooldown_and_exact_initial_terminal_gaps_verified'], 'Exact coverage')
        rows = [r for r in compact if r['trial'] == trial]
        lookup = {r['case']: r for r in rows}
        need(len(lookup) == len(index), 'Compact scope')
        for row in index:
            need(all(lookup[row['case']][k] == value for k, value in row.items()), 'Compact original values')
            need(row['metrics']['tasks_violating_exact_gap'] == 0, 'Protected case overdue')
        keys_across_trials.update(lookup)
        physical_keys.update(r['process_key'] for r in index)
        groups, frozen = read(folder / 'test-groups.json'), read(folder / 'selections-frozen.json')
        need(len(groups) == 11 and all(len(v) == 32 for v in groups.values()), '32paired observation conditions per group')
        need(sum(map(len, groups.values())) == 352, 'Group reference count')
        for name, keys in groups.items():
            values = [lookup[k] for k in keys]
            need(len(set(keys)) == 32 and all(r['specification']['phase'] == 'test' for r in values), 'No duplicated or non-test group cases')
            need({r['specification']['seed'] for r in values} == {6600, 6601}, 'New test seeds')
            need({r['specification']['noise_c'] for r in values} == {.2, 1.}, 'Paired noise conditions')
            pairs = Counter((r['specification']['seed'], r['specification']['scenario'], r['specification']['duration_s'],
                r['specification']['fault_start_s']) for r in values)
            need(len(pairs) == 16 and set(pairs.values()) == {2}, 'Same process under two noises')
            result = aggregate['trials'][str(trial)][name]
            segments = sum(r['metrics']['anomaly_segments'] for r in values)
            misses = sum(r['metrics']['missed_segments'] for r in values)
            weight = sum(r['event_weight_sum'] for r in values)
            need(result['anomaly_segments'] == segments and result['missed_segments'] == misses, 'Pooled event counts')
            need(close(result['miss_rate'], misses / segments if segments else None), 'Pooled miss rate')
            need(close(result['weighted_censored_delay_s'], sum(r['weighted_delay_sum'] for r in values) / weight if weight else None), 'Pooled censor weights')
            for target, source in [('mean_extra_checks', 'selected_detours'), ('mean_estimator_calls', 'estimator_calls'),
                ('mean_travel_m', 'travel_m'), ('mean_service_s', 'service_s')]:
                need(close(result[target], sum(r['metrics'][source] for r in values) / 32), 'Reported cost average')
            if name in frozen:
                need(all(r['version_id'] == frozen[name]['version_id'] for r in values), 'Frozen selected program actually tested')
        journal = [json.loads(s) for s in (folder / 'journal.jsonl').read_text().splitlines()]
        freeze = next(r for r in journal if r['event'] == 'all_selections_frozen')
        need(freeze['sha256'] == sha(folder / 'selections-frozen.json'), 'Selection freeze unchanged')
        need(all(r['sequence'] > freeze['sequence'] for r in journal if r['event'] == 'case_completed' and r['phase'] == 'test'), 'No test execution before selection freeze')
        updates = [read(p) for p in sorted((folder / 'meta').glob('*.json'))]
        need(len(updates) == 4, 'Two rounds per meta method')
        for update in updates:
            old, new = update['arms']['incumbent']['selected'], update['arms']['candidate']['selected']
            need(old['task_start']['version_id'] == new['task_start']['version_id'] and
                old['history_start_length'] == new['history_start_length'] == 0, 'Meta arms reset task start and memory')
            expected = next(r for r in aggregate['meta_updates'] if r['trial'] == trial and r['method'] == update['method'] and r['round'] == update['round'])
            need(expected['accepted'] == update['accepted'] and expected['retained_id'] == update['retained_id'], 'Reported meta decisions')
        proposals = [read(p) for p in (folder / 'proposals').glob('*.json')]
        need(len(proposals) == 50, 'No extra task proposal slots')
        for method, cost in aggregate['cost_by_trial'][str(trial)].items():
            chosen = [r for r in proposals if r['tag'] == method or r['tag'].startswith(method + '-')]
            generations = [r['generation'] for r in chosen if 'generation' in r]
            generations += [r['generation'] for r in updates if r['method'] == method and 'generation' in r]
            need(cost['model_calls'] == len(generations) == (0 if method == 'random_search' else 13), 'Total preparation plus task model calls')
            need(cost['task_candidate_slots'] == len(chosen), 'Candidate slot cost')
            for field in ('input_tokens', 'output_tokens', 'generation_seconds'):
                need(close(cost[field], sum(r[field] for r in generations)), 'Actual generation cost')
            need(all(r['input_tokens'] <= 6144 and r['output_tokens'] <= 768 for r in generations), 'All token caps')
        total_cases += len(index);total_calls += 39;task_slots += 50;test_refs += 352
    need(aggregate['actual_closed_loop_records'] == total_cases and aggregate['distinct_case_signatures_across_trials'] == len(keys_across_trials), 'Actual versus reused case counts')
    need(aggregate['distinct_physical_processes_across_trials'] == len(physical_keys) == 28, 'Physical process scope')
    need(total_calls == 117 and task_slots == 150 and test_refs == 1056, 'Whole study envelopes')
    for name, group in aggregate['main_groups'].items():
        for target, source in [('mean_miss_rate', 'miss_rate'), ('mean_weighted_censored_delay_s', 'weighted_censored_delay_s'), ('mean_extra_checks', 'mean_extra_checks')]:
            need(close(group[target], sum(aggregate['trials'][str(t)][name][source] for t in range(3)) / 3), 'Trial average')
    for row in read(ev / 'report/demo-selection.json'):
        path = ev / 'demo' / (row['case'] + '.json.gz')
        original = read(ev / f"trial-{row['trial']}/case-index.json")
        indexed = next(r for r in original if r['case'] == row['case'])
        need(sha(path) == indexed['sha256'], 'Exact full demo record')
        record = json.loads(gzip.decompress(path.read_bytes()))
        for gap in record['exact_task_gaps']:
            times = [0.] + sorted(e['time_s'] for e in record['inspection_events'] if e['task'] == gap['task']) + [1800.]
            need(max(b - a for a, b in zip(times, times[1:])) == gap['maximum_gap_including_initial_and_terminal_s'], 'Exact demo event gaps')
    figures = []
    for path in sorted((ev / 'report').glob('*.png')):
        raw = path.read_bytes()
        need(raw[:8] == b'\x89PNG\r\n\x1a\n', 'PNG format')
        dimensions = struct.unpack('>II', raw[16:24])
        need(dimensions[0] >= 1500 and dimensions[1] >= 700, 'Readable figure resolution')
        figures.append({'file': path.name, 'pixels': list(dimensions), 'sha256': sha(path)})
    need(len(figures) == 3, 'Three declared figures')
    result = {'source_archives_verified': len(manifests), 'current_source_members_verified': current,
        'original_transferred_files': len(transfer), 'successful_jobs_and_array_items': checked_jobs,
        'full_tests_no_skips': 152, 'actual_closed_loop_records': total_cases,
        'distinct_case_signatures': len(keys_across_trials), 'distinct_physical_processes': len(physical_keys),
        'formal_model_calls': total_calls, 'task_candidate_slots': task_slots, 'test_group_case_references': test_refs,
        'all_pooled_statistics_and_generation_costs_recomputed': True,
        'meta_reset_transfer_and_selection_records_verified': True, 'figures': figures,
        'scope': 'Portable original-file/source/statistic verification; whole actual case replay in independent Slurm audits'}
    with args.output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
