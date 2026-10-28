"""Local M12 frozen-source, transfer, budgets, grouping and demo verification."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import tarfile


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    archived, current = 0, 0
    manifests = sorted(base.glob('*source_manifest_v*.json'))
    latest = {}
    for path in manifests:
        prefix, version = path.name.rsplit('_v', 1)
        latest[prefix] = max(latest.get(prefix, 0), int(version.removesuffix('.json')))
    for manifest_path in manifests:
        manifest = read(manifest_path)
        archive_name = manifest_path.name.replace('_manifest_', '-').replace('_', '-').replace('.json', '.tar.gz')
        archive = base / 'source' / archive_name
        require(sha(archive) == manifest['archive_sha256'], 'Source archive fingerprint ' + archive_name)
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            require({m.name for m in members} == set(manifest['files']), 'Archive members')
            for member in members:
                require(hashlib.sha256(tar.extractfile(member).read()).hexdigest() == manifest['files'][member.name], 'Archive member bytes')
        archived += 1
        prefix, version = manifest_path.name.rsplit('_v', 1)
        if int(version.removesuffix('.json')) == latest[prefix]:
            for rel, value in manifest['files'].items():
                require(hashlib.sha256((root / rel).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == value, 'Current source drift ' + rel)
                current += 1
    require(archived >= 5, 'Three experiment archives plus analysis and package')
    transfer = read(ev / 'transfer-manifest.json')
    for rel, record in transfer.items():
        path = ev / rel
        require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], 'Transferred original file ' + rel)
    with (ev / 'slurm-final.psv').open() as f:
        jobs = {r['JobID']: r for r in csv.DictReader(f, delimiter='|')}
    ids = read(ev / 'job-identities.json')
    for label, job in ids.items():
        expected = ('FAILED', '1:0') if label in ('smoke_v1', 'superseded_audit_report') else ('COMPLETED', '0:0')
        require((jobs[str(job)]['State'], jobs[str(job)]['ExitCode']) == expected, 'Job terminal state ' + label)
    log = (ev / f"jobs/smoke-v3-{ids['smoke_v3']}.err").read_text()
    require('Ran 143 tests' in log and '\nOK\n' in log and 'skipped' not in log, 'Complete tests no skips')
    gate = read(ev / 'development-gate-v3.json')
    require(gate['passed'] and not gate['pilot_test_read'] and gate['raw_proposal_valid'], 'Gate before pilot')
    summary, audit = read(ev / 'summary.json'), read(ev / 'full-audit.json')
    index, groups = read(ev / 'case-index.json'), read(ev / 'test-groups.json')
    compact = json.loads(gzip.decompress((ev / 'report/compact-metrics.json.gz').read_bytes()))
    lookup = {r['case']: r for r in compact}
    require(len(lookup) == len(index) == summary['unique_cases'] == audit['all_unique_cases_replayed'], 'Full actual scope')
    for row in index:
        require(all(lookup[row['case']][key] == value for key, value in row.items()), 'Compact exact values')
        require(row['metrics']['tasks_violating_exact_gap'] == 0, 'Protected case failure')
    require(not summary['new_neural_training'] and not summary['improver_updated'], 'No unsupported training or RSI claim')
    require(audit['actual_pilot_model_calls'] == summary['model_calls'] == 18 and audit['candidate_slots_checked'] == 27, 'Frozen generation budgets')
    require(audit['all_quota_cooldown_and_public_outcomes_recomputed'] and audit['public_only_prompt_chains_reconstructed'] and audit['selection_before_all_test_execution_verified'], 'Actual semantic checks')
    require(audit['maximum_exact_gap_ratio'] <= 1, 'Full exact coverage')
    for phase, seeds in [('development', {6201, 6202}), ('test', {6300, 6301})]:
        require({r['specification']['seed'] for r in index if r['specification']['phase'] == phase} == seeds, 'Data split')
    journal = [json.loads(s) for s in (ev / 'journal.jsonl').read_text().splitlines()]
    freeze = next(r for r in journal if r['event'] == 'all_selections_frozen')
    require(freeze['sha256'] == sha(ev / 'selections-frozen.json'), 'Freeze fingerprint')
    require(all(r['sequence'] > freeze['sequence'] for r in journal if r['event'] == 'case_completed' and r['phase'] == 'test'), 'Test cannot precede freeze')
    aggregate = read(ev / 'report/aggregate.json')
    for name, values in aggregate['all_groups'].items():
        keys = groups[name]
        require(len(keys) == 16, 'Common test conditions')
        rows = [lookup[k] for k in keys]
        require(all(r['specification']['phase'] == 'test' for r in rows), 'Group references only test')
        total = sum(r['metrics']['anomaly_segments'] for r in rows)
        missed = sum(r['metrics']['missed_segments'] for r in rows)
        weight = sum(r['event_weight_sum'] for r in rows)
        require(values['anomaly_segments'] == total and values['missed_segments'] == missed, 'Event counts')
        require(close(values['pooled_miss_rate'], missed / total if total else None), 'Pooled misses')
        require(close(values['weighted_censored_delay_s'], sum(r['weighted_delay_sum'] for r in rows) / weight if weight else None), 'Censored weights')
        counts = sum((Counter(r['behavior_counts']) for r in rows), Counter())
        require(dict(counts) == values['behavior_counts'], 'Dispatch attribution')
        require(close(values['mean_extra_checks'], statistics.mean(r['metrics']['selected_detours'] for r in rows)), 'Extra checking')
        require(close(values['mean_estimator_calls'], statistics.mean(r['metrics']['estimator_calls'] for r in rows)), 'Real tool calls')
    require(len(groups) == 29 and sum(len(v) for v in groups.values()) == 464, 'Declared group reference scope')
    require(len({k for v in groups.values() for k in v}) == aggregate['test_unique_case_records'], 'Deduplicated test evidence')
    proposals = [read(p) for p in (ev / 'proposals').glob('*.json')]
    require(len(proposals) == 27, 'No extra candidate attempts')
    for method, cost in aggregate['proposal_costs'].items():
        rows = [r for r in proposals if r['method'] == method]
        require(len(rows) == cost['candidate_slots'] == 9, 'Same candidate slots')
        require(sum(r['accepted'] for r in rows) == cost['accepted'], 'Accepted proposal count')
        generated = [r['generation'] for r in rows if 'generation' in r]
        require(len(generated) == cost['model_calls'], 'Real model responses')
        for key in ('input_tokens', 'output_tokens', 'generation_seconds'):
            require(close(sum(r[key] for r in generated), cost[key]), 'Actual generation cost')
    demos = read(ev / 'report/demo-selection.json')
    for group, row in demos.items():
        path = ev / 'demo' / (row['case'] + '.json.gz')
        record = json.loads(gzip.decompress(path.read_bytes()))
        require(sha(path) == lookup[row['case']]['sha256'], 'Exact full demo bytes')
        require(row['case'] in groups[group] and record['workflow'] == row['workflow'], 'Predeclared demo program')
        for gap in record['exact_task_gaps']:
            times = [0.] + sorted(e['time_s'] for e in record['inspection_events'] if e['task'] == gap['task']) + [1800.]
            maximum = max(b - a for a, b in zip(times, times[1:]))
            require(maximum == gap['maximum_gap_including_initial_and_terminal_s'], 'Exact demo completion gaps')
            require(maximum <= record['workflow_contract']['deadlines_s'][gap['task']], 'Demo revisit requirement')
    figures = []
    for path in sorted((ev / 'report').glob('*.png')):
        raw = path.read_bytes()
        require(raw[:8] == b'\x89PNG\r\n\x1a\n', 'PNG format')
        dimensions = struct.unpack('>II', raw[16:24])
        require(dimensions[0] >= 1500 and dimensions[1] >= 700, 'Figure dimensions')
        figures.append({'file': path.name, 'pixels': list(dimensions), 'sha256': sha(path)})
    require(len(figures) == 3, 'Three figures')
    repair = read(ev / 'figure-repair.json')
    require(jobs[str(repair['job_id'])]['State'] == 'COMPLETED' and jobs[str(repair['job_id'])]['ExitCode'] == '0:0', 'Figure repair terminal state')
    require(repair['source_aggregate_sha256'] == sha(ev / 'report/aggregate.json'), 'Unchanged data in repaired figure')
    for name, value in repair['files'].items():
        path = ev / 'report-corrected' / name
        require(path.stat().st_size == value['bytes'] and sha(path) == value['sha256'], 'Corrected plot fingerprint')
    final_figures = [r for r in figures if r['file'] != 'm12-attribution.png']
    repaired = ev / 'report-corrected/m12-attribution.png'
    raw = repaired.read_bytes()
    require(raw[:8] == b'\x89PNG\r\n\x1a\n', 'Corrected PNG format')
    dimensions = struct.unpack('>II', raw[16:24])
    require(dimensions[0] >= 1500 and dimensions[1] >= 700, 'Corrected figure resolution')
    final_figures.append({'file': 'report-corrected/m12-attribution.png', 'pixels': list(dimensions), 'sha256': sha(repaired)})
    result = {'source_archives_verified': archived, 'current_source_members_verified': current,
        'original_transferred_files': len(transfer), 'unique_cases': len(index), 'full_tests_no_skips': 143,
        'test_group_references': 464, 'test_unique_cases': aggregate['test_unique_case_records'],
        'whole_case_replay_receipt_and_all_statistics_checked': True, 'model_calls': 18,
        'candidate_slots': 27, 'preserved_failed_smoke_attempt': ids['smoke_v1'],
        'preserved_invalid_smoke_proposal': ids['smoke_v2'], 'original_figures': figures,
        'final_report_figures': final_figures, 'layout_repair_verified_without_data_change': True,
        'scope': 'Artifact/current source verification; actual full replay is the independent Slurm audit'}
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
