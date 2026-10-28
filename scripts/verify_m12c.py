"""Portable C0 archive, original-file, pairing and full-statistic checks."""
import argparse
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
        m = read(path)
        archive = base / 'source' / path.name.replace('_manifest_', '-').replace('_', '-').replace('.json', '.tar.gz')
        need(sha(archive) == m['archive_sha256'], 'Archive fingerprint')
        with tarfile.open(archive) as tar:
            members = [item for item in tar.getmembers() if item.isfile()]
            need({item.name for item in members} == set(m['files']), 'Archive contents')
            for item in members:
                need(hashlib.sha256(tar.extractfile(item).read()).hexdigest() == m['files'][item.name], 'Archived source bytes')
        prefix, version = path.stem.rsplit('_v', 1)
        if int(version) == latest[prefix]:
            for rel, h in m['files'].items():
                need(hashlib.sha256((root / rel).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == h, 'Current source drift ' + rel)
                current += 1
    transfer = read(ev / 'transfer-manifest.json')
    for rel, value in transfer.items():
        path = ev / rel
        need(path.stat().st_size == value['bytes'] and sha(path) == value['sha256'], 'Original transferred file ' + rel)
    with (ev / 'slurm-final.psv').open() as f:
        ledger = {r['JobID']: r for r in csv.DictReader(f, delimiter='|')}
    ids, jobs = read(ev / 'job-identities.json'), []
    for key, job in ids.items():
        targets = [f'{job}_{i}' for i in range(3)] if key.endswith('_array') else [str(job)]
        for target in targets:
            need(ledger[target]['State'] == 'COMPLETED' and ledger[target]['ExitCode'] == '0:0', 'Own job terminal state')
            jobs.append(target)
    log = (ev / f"jobs/smoke-{ids['smoke']}.err").read_text()
    need('Ran 160 tests' in log and '\nOK\n' in log and 'skipped' not in log, 'Complete160tests')
    gate, smoke = read(ev / 'development-gate.json'), read(ev / 'smoke-audit.json')
    need(gate['passed'] and not gate['formal_test_read'] and gate['all_raw_proposals_valid'], 'Development gate')
    need(smoke['actual_closed_loops_replayed'] == 10 and smoke['all_event_provenance_and_redaction_checks_passed'], 'Real interface replay')
    aggregate = read(ev / 'report/aggregate.json')
    compact = json.loads(gzip.decompress((ev / 'report/compact-metrics.json.gz').read_bytes()))
    actual_count, signatures, physical_keys, formal_calls = 0, set(), set(), 0
    trial_updates = []
    for t in range(3):
        folder = ev / f'trial-{t}'
        summary, index, audit = read(folder / 'summary.json'), read(folder / 'case-index.json'), read(ev / f'audit-{t}.json')
        need(summary['actual_model_calls'] == audit['actual_model_calls'] == 15, 'Frozen model-call budget')
        need(summary['unique_cases'] == len(index) == audit['actual_closed_loops_replayed'], 'Complete actual replay scope')
        need(audit['task_candidate_slots'] == 24 and audit['all_event_provenance_and_redaction_checks_passed'] and
            audit['all_information_prompts_and_acceptance_chains_reconstructed'] and audit['all_starts_reset_and_test_selected_before_execution'], 'Actual evidence and selection checks')
        need(audit['maximum_exact_gap_ratio'] <= 1., 'Protected coverage')
        rows = [r for r in compact if r['trial'] == t]
        lookup = {r['case']: r for r in rows}
        need(len(rows) == len(lookup) == len(index), 'No duplicate compact rows within a trial')
        for r in index:
            need(all(lookup[r['case']][k] == value for k, value in r.items()), 'Compact original values')
            need(r['metrics']['tasks_violating_exact_gap'] == 0, 'Exact revisit failure')
        signatures.update(lookup);physical_keys.update(r['process_key'] for r in rows)
        groups, frozen = read(folder / 'test-groups.json'), read(folder / 'selections-frozen.json')
        need(len(groups) == 11 and all(len(v) == 32 for v in groups.values()), 'Paired test group scope')
        common = None
        for name, keys in groups.items():
            values = [lookup[k] for k in keys]
            conditions = {(r['specification']['seed'], r['specification']['scenario'], r['specification']['duration_s'],
                r['specification']['fault_start_s'], r['specification']['noise_c']) for r in values}
            common = conditions if common is None else common
            need(len(set(keys)) == len(conditions) == 32 and conditions == common, 'Common physical/noise conditions')
            need({r['specification']['seed'] for r in values} == {7400, 7401}, 'New test seeds')
            need({r['specification']['noise_c'] for r in values} == {.2, 1.}, 'Paired noises')
            result = aggregate['trials'][str(t)][name]
            count = sum(r['metrics']['anomaly_segments'] for r in values)
            miss = sum(r['metrics']['missed_segments'] for r in values)
            weight = sum(r['event_weight_sum'] for r in values)
            need(result['anomaly_segments'] == count and result['missed_segments'] == miss, 'Pooled event denominator')
            need(close(result['miss_rate'], miss / count if count else None), 'Pooled miss rate')
            need(close(result['weighted_censored_delay_s'], sum(r['weighted_delay_sum'] for r in values) / weight if weight else None), 'Censored weights')
            for target, field in [('mean_extra_checks', 'selected_detours'), ('mean_estimator_calls', 'estimator_calls'),
                    ('mean_travel_m', 'travel_m'), ('mean_service_s', 'service_s')]:
                need(close(result[target], sum(r['metrics'][field] for r in values) / 32), 'Actual resource costs')
            if name in frozen:
                need(all(r['version_id'] == frozen[name]['version_id'] for r in values), 'Selected program actually executed')
        deployment = read(folder / 'deployment-interventions.json')['rule_signals_inactive']
        expected = dict(deployment['original_program'])
        expected['rules'] = [r for r in expected['rules'] if r['signal'] == 'age']
        need(expected == deployment['effective_program'], 'Explicit deployed rule-signal intervention')
        need(all(lookup[k]['workflow'] == expected for k in groups['events3/rule_signals_inactive']), 'Actual inactive-rule deployment')
        journal = [json.loads(s) for s in (folder / 'journal.jsonl').read_text().splitlines()]
        freeze = next(r for r in journal if r['event'] == 'all_selections_frozen')
        need(freeze['sha256'] == sha(folder / 'selections-frozen.json'), 'Frozen selection bytes')
        need(all(r['sequence'] > freeze['sequence'] for r in journal if r['event'] == 'case_completed' and r['phase'] == 'test'), 'No test execution before freezing')
        proposals = [read(p) for p in (folder / 'proposals').glob('*.json')]
        need(len(proposals) == 24, 'No added candidate slots')
        for method, slots in [('summary', 9), ('events', 3), ('redacted', 3), ('random', 9)]:
            values = [r for r in proposals if r['method'] == method]
            cost = aggregate['cost_by_trial'][str(t)][method]
            need(len(values) == cost['candidate_slots'] == slots, 'Method candidate budget')
            generated = [r['generation'] for r in values if 'generation' in r]
            need(len(generated) == cost['model_calls'] == (0 if method == 'random' else slots), 'Actual model replies')
            need(cost['accepted'] == sum(r['accepted'] for r in values) and cost['duplicates'] == sum(r['reason'] == 'duplicate' for r in values), 'Proposal outcomes')
            for k in ('input_tokens', 'output_tokens', 'generation_seconds'):
                need(close(cost[k], sum(g[k] for g in generated)), 'Generation costs')
            for generation in generated:
                need(generation['input_tokens'] <= 6144 and generation['output_tokens'] <= 768, 'Token caps')
                prompt = generation['prompt']
                if method == 'summary':
                    need('event_evidence' not in prompt, 'Summary without raw event cards')
                else:
                    evidence = prompt['event_evidence']
                    need(evidence['selected_source_events'] <= 8 and max(evidence['category_counts'].values()) <= 2, 'Event budget')
                    if method == 'redacted':
                        need(all(set(r) == {'category', 'details'} for r in evidence['events']), 'No identifying/time/value detail in redacted cards')
        actual_count += len(index);formal_calls += 15
        trial_updates.append({'trial': t, 'cases': len(index), 'processes': summary['unique_processes']})
    need(aggregate['actual_closed_loop_records'] == actual_count and aggregate['distinct_case_signatures'] == len(signatures), 'Actual versus deduplicated records')
    need(aggregate['distinct_physical_processes'] == len(physical_keys) == 24, 'Physical process scope')
    need(formal_calls == 45 and aggregate['test_group_case_references'] == 1056, 'Total study scope')
    for name, value in aggregate['main_groups'].items():
        for target, field in [('mean_miss_rate', 'miss_rate'), ('mean_weighted_censored_delay_s', 'weighted_censored_delay_s'), ('mean_extra_checks', 'mean_extra_checks')]:
            need(close(value[target], sum(aggregate['trials'][str(t)][name][field] for t in range(3)) / 3), 'Trial mean')
    for demo in read(ev / 'report/demo-selection.json'):
        path = ev / 'demo' / (demo['case'] + '.json.gz')
        row = next(r for r in read(ev / f"trial-{demo['trial']}/case-index.json") if r['case'] == demo['case'])
        need(sha(path) == row['sha256'], 'Exact full example bytes')
        record = json.loads(gzip.decompress(path.read_bytes()))
        need(record['workflow'] == demo['workflow'], 'Predeclared example program')
        for gap in record['exact_task_gaps']:
            times = [0.] + sorted(e['time_s'] for e in record['inspection_events'] if e['task'] == gap['task']) + [1800.]
            need(max(b - a for a, b in zip(times, times[1:])) == gap['maximum_gap_including_initial_and_terminal_s'], 'Exact example event gaps')
    evidence = read(ev / 'report/predeclared-event-input.json')
    proposal = read(ev / 'trial-0/proposals/events-0.json')
    need(evidence == proposal['generation']['prompt']['event_evidence'], 'Unselected predeclared event figure input')
    figures = []
    for path in sorted((ev / 'report').glob('*.png')):
        raw = path.read_bytes()
        need(raw[:8] == b'\x89PNG\r\n\x1a\n', 'PNG format')
        size = struct.unpack('>II', raw[16:24])
        need(size[0] >= 1500 and size[1] >= 700, 'Readable figure resolution')
        figures.append({'file': path.name, 'pixels': list(size), 'sha256': sha(path)})
    need(len(figures) == 3, 'Three figures')
    result = {'source_archives_verified': len(manifests), 'current_source_members_verified': current,
        'original_transferred_files': len(transfer), 'successful_jobs_and_array_items': jobs,
        'complete_tests_no_skips': 160, 'actual_closed_loop_records': actual_count, 'distinct_case_signatures': len(signatures),
        'distinct_physical_processes': len(physical_keys), 'formal_model_calls': 45, 'candidate_slots': 72,
        'test_group_case_references': 1056, 'trial_scopes': trial_updates,
        'all_group_statistics_generation_costs_and_selection_records_verified': True,
        'event_controls_and_predeclared_figure_source_verified': True, 'figures': figures,
        'scope': 'Current source/original-file and aggregate checks; actual complete replay and event provenance are independently checked on Slurm'}
    with args.output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
