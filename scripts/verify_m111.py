"""Portable local receipt, grouping, exact-gap and capability verification."""
import argparse
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


def close(a, b):
    return a is None and b is None or a is not None and b is not None and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--research', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    base, root = args.research, args.workspace
    ev = base / 'evidence'
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    for name, filename in [('source_manifest_v1.json', 'source-v1.tar.gz'),
                            ('analysis_source_manifest_v1.json', 'analysis-source-v1.tar.gz'),
                            ('package_source_manifest_v1.json', 'package-source-v1.tar.gz')]:
        manifest = read(base / name)
        archive = base / 'source' / filename
        require(sha(archive) == manifest['archive_sha256'], 'Source archive bytes')
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            require({m.name for m in members} == set(manifest['files']), 'Source archive members')
            for member in members:
                raw = tar.extractfile(member).read()
                require(hashlib.sha256(raw).hexdigest() == manifest['files'][member.name], 'Archived source content')
        for rel, value in manifest['files'].items():
            require(hashlib.sha256((root / rel).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == value, 'Current source drift')
    transfer = read(ev / 'transfer-manifest.json')
    for rel, value in transfer.items():
        path = ev / rel
        require(path.stat().st_size == value['bytes'] and sha(path) == value['sha256'], 'Transferred file ' + rel)
    ids = read(ev / 'job-identities.json')
    with (ev / 'slurm-final.psv').open() as f:
        jobs = {r['JobID']: r for r in csv.DictReader(f, delimiter='|')}
    for name, job in ids.items():
        if isinstance(job, int):
            require(jobs[str(job)]['State'] == 'COMPLETED' and jobs[str(job)]['ExitCode'] == '0:0', 'Terminal job')
    log = (ev / f"jobs/development-{ids['development']}.err").read_text()
    require('Ran 132 tests' in log and log.rstrip().endswith('OK') and 'skipped' not in log, 'Full tests')
    require(read(ev / 'm11-targeted-review.json')['all_four_declared_demo_cases_replayed'], 'M11 targeted actual replay')
    require(read(ev / 'pilot-summary.json')['new_policy_training'] is False, 'No new training')
    index = read(ev / 'case-index.json')
    compact = json.loads(gzip.decompress((ev / 'report/compact-metrics.json.gz').read_bytes()))
    require(len(index) == len(compact) == 128 and len({r['case'] for r in index}) == 128, 'Pilot scope')
    lookup = {r['case']: r for r in index}
    for row in compact:
        for key in lookup[row['case']]:
            require(row[key] == lookup[row['case']][key], 'Compact/index values')
    aggregate = read(ev / 'report/aggregate.json')
    for estimator, methods in aggregate['groups'].items():
        for method, summary in methods.items():
            rows = [r for r in compact if r['specification']['estimator'] == estimator and r['specification']['selector'] == method]
            require(len(rows) == summary['cases'] == 16, 'Paired fixture count')
            count = sum(r['metrics']['anomaly_segments'] for r in rows)
            missed = sum(r['metrics']['missed_segments'] for r in rows)
            weight = sum(r['event_weight_sum'] for r in rows)
            require(count == summary['segments'] and missed == summary['missed_segments'], 'Pooled event counts')
            require(close(missed / count if count else None, summary['pooled_miss_rate']), 'Pooled miss rate')
            delay = sum(r['weighted_delay_sum'] for r in rows) / weight if weight else None
            require(close(delay, summary['weighted_censored_delay_s']), 'Censored weights')
            require(close(max(r['metrics']['maximum_exact_gap_ratio'] for r in rows), summary['maximum_exact_gap_ratio']), 'Worst exact gap')
            require(sum(r['metrics']['tasks_violating_exact_gap'] > 0 for r in rows) == summary['cases_with_exact_gap_violation'], 'Violating cases')
            for target, source in [('mean_coverage', 'final_professional_coverage'),
                ('mean_overdue_fraction', 'mean_overdue_fraction'), ('mean_extra_inspections', 'selected_detours')]:
                require(close(statistics.mean(r['metrics'][source] for r in rows), summary[target]), 'Case metric average')
            if method != 'risk_free':
                require(summary['cases_with_exact_gap_violation'] == 0 and summary['maximum_exact_gap_ratio'] <= 1, 'Protected group violation')
    require(aggregate['remote_service_evidence_identical_across_all_selectors'], 'Common power evidence')
    for process, record in aggregate['remote_service_by_process'].items():
        counts = {kind: sum(e['kind'] == kind for e in record['events']) for kind in ('new_measured_alert', 'persistent_recheck', 'recovery_confirmation')}
        for row in index:
            if row['process_key'] == process:
                require(row['service_evidence_event_counts'] == counts, 'Power lifecycle counts')
        for event in record['events']:
            require(not event['root_cause_diagnosed'] and not event['voltage_or_current_measured'], 'Do not infer unavailable channels')
    audit = read(ev / 'full-audit.json')
    require(audit['all_pilot_cases_replayed'] == 128 and audit['unique_process_arrays_replayed'] == 16, 'Whole pilot audit')
    require(audit['protected_cases'] == 96 and audit['maximum_protected_exact_gap_ratio'] <= 1, 'Coverage audit')
    require(audit['all_exact_completion_gaps_including_initial_and_terminal_recomputed'] and audit['all_public_power_evidence_lifecycles_independently_rebuilt'], 'Actual event semantics')
    require(audit['untrained_scorer_seed_and_no_updates_verified'], 'Untrained comparator')
    demos = list((ev / 'demo').glob('*.json.gz'))
    require(len(demos) == 4, 'Four predeclared demo cases')
    deadlines = {'thermal': 300, 'electrical': 600, 'cooling': 600, 'control': 600}
    for path in demos:
        record = json.loads(gzip.decompress(path.read_bytes()))
        row = next(r for r in index if r['specification'] == record['specification'])
        require(sha(path) == row['sha256'], 'Exact demo bytes')
        for gap in record['exact_task_gaps']:
            times = [0.] + sorted(e['time_s'] for e in record['inspection_events'] if e['task'] == gap['task']) + [1800.]
            maximum = max(b - a for a, b in zip(times, times[1:]))
            require(close(maximum, gap['maximum_gap_including_initial_and_terminal_s']), 'Exact demo gap')
            if record['specification']['selector'] != 'risk_free':
                require(maximum <= deadlines[gap['task_id'].split(':')[0]], 'Protected demo')
        for selected in record['selected_certificates']:
            if selected['protection_enabled']:
                cert = selected['certificate']
                require(cert['ok'] and all(t <= d + 1e-9 for t, d in zip(cert['fallback_completion_s'], cert['fallback_due_s'])), 'Full fallback not candidate-only')
    figures = []
    for path in sorted((ev / 'report').glob('*.png')):
        raw = path.read_bytes()
        require(raw[:8] == b'\x89PNG\r\n\x1a\n', 'PNG format')
        width, height = struct.unpack('>II', raw[16:24])
        require(width > 1000 and height > 650, 'Figure legibility dimensions')
        figures.append({'file': path.name, 'pixels': [width, height], 'sha256': sha(path)})
    require(len(figures) == 3, 'Three figures')
    result = {'three_source_archives_and_current_members_verified': True, 'original_transferred_files': len(transfer),
        'five_jobs_completed_successfully': True, 'full_tests_no_skips': 132,
        'pilot_cases': 128, 'protected_cases': 96, 'all_groups_recomputed_from_case_data': True,
        'four_full_demo_gap_and_fallback_checks': True, 'new_policy_training': False, 'figures': figures}
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
