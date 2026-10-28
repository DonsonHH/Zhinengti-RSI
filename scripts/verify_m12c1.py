"""Portable original/source checks and independent opportunity aggregation."""
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
        manifest = read(path)
        archive = base / 'source' / path.name.replace('_manifest_', '-').replace('_', '-').replace('.json', '.tar.gz')
        need(sha(archive) == manifest['archive_sha256'], 'Source archive fingerprint')
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            need({m.name for m in members} == set(manifest['files']), 'Source archive members')
            for member in members:
                need(hashlib.sha256(tar.extractfile(member).read()).hexdigest() == manifest['files'][member.name], 'Archived source bytes')
        prefix, version = path.stem.rsplit('_v', 1)
        if int(version) == latest[prefix]:
            for rel, h in manifest['files'].items():
                need(hashlib.sha256((root / rel).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == h, 'Current source drift ' + rel)
                current += 1
    transfer = read(ev / 'transfer-manifest.json')
    for rel, record in transfer.items():
        path = ev / rel
        need(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], 'Original transferred file ' + rel)
    with (ev / 'slurm-final.psv').open() as f:
        ledger = {r['JobID']: r for r in csv.DictReader(f, delimiter='|')}
    ids = read(ev / 'job-identities.json')
    for label, job in ids.items():
        expected = ('FAILED', '1:0') if label == 'superseded_smoke_audit' else ('COMPLETED', '0:0')
        need((ledger[str(job)]['State'], ledger[str(job)]['ExitCode']) == expected, 'Own job terminal state ' + label)
    log = (ev / f"jobs/smoke-{ids['smoke']}.err").read_text()
    need('Ran 167 tests' in log and '\nOK\n' in log and 'skipped' not in log, '167complete tests without skips')
    gate = read(ev / 'development-gate.json')
    need(gate['passed'] and not gate['formal_data_read'] and gate['all_smoke_cases_replayed'] == 12, 'Smoke audit gate before formal execution')
    index, summary, audit = read(ev / 'case-index.json'), read(ev / 'summary.json'), read(ev / 'formal-audit.json')
    compact = json.loads(gzip.decompress((ev / 'report/compact-metrics.json.gz').read_bytes()))
    need(compact == index, 'All compact case/probe records are exact original values')
    main_rows = [r for r in index if r['counterfactual_source'] is None]
    branch_rows = [r for r in index if r['counterfactual_source'] is not None]
    need(len(main_rows) == summary['main_cases'] == audit['main_cases_replayed'] == 192, 'Predeclared main192cases')
    need(len(index) == summary['all_actual_cases'] == audit['all_actual_cases_replayed'], 'Complete actual replay scope')
    need(len(branch_rows) == summary['evaluator_branches'] == audit['evaluator_one_detour_branches_replayed'], 'Actual counterfactual branch scope')
    need(summary['physical_processes'] == audit['physical_arrays_verified'] == 16, 'Physical process scope')
    need(summary['LLM_calls'] == 0 and not summary['new_training'] and summary['all_weights_unchanged'], 'No unrequested model calls or training')
    need(audit['all_finite_probe_counts_first_witnesses_and_cues_independently_rebuilt'] and audit['maximum_exact_gap_ratio'] <= 1, 'Independent complete local probe and exact coverage')
    need(audit['all_public_legal_candidates_independently_rebuilt'] > audit['actual_dispatches'], 'Unselected candidates checked as well as chosen actions')
    lookup = {r['case']: r for r in index}
    need(len(lookup) == len(index), 'Unique actual case IDs')
    aggregate = read(ev / 'report/aggregate.json')
    expected_branches = {}
    for estimator in ('nominal', 'history'):
        reference = None
        for method in ('tour', 'cue_rule', 'actor'):
            rows = [r for r in main_rows if r['specification']['estimator'] == estimator and r['specification']['controller'] == method]
            need(len(rows) == 32, 'Common main condition count')
            conditions = {(r['specification']['seed'], r['specification']['scenario'], r['specification']['duration_s'],
                r['specification']['fault_start_s'], r['specification']['noise_c']) for r in rows}
            reference = conditions if reference is None else reference
            need(len(conditions) == 32 and reference == conditions, 'Common paired processes/noise')
            need({r['specification']['seed'] for r in rows} == {7700, 7701} and
                {r['specification']['noise_c'] for r in rows} == {.2, 1.}, 'Declared seeds and noise')
            result = aggregate['groups'][estimator][method]
            total = sum(r['metrics']['anomaly_segments'] for r in rows);missed = sum(r['metrics']['missed_segments'] for r in rows)
            weight = sum(r['event_weight_sum'] for r in rows)
            need(result['anomaly_segments'] == total and result['missed_segments'] == missed, 'Pooled event counts')
            need(close(result['pooled_miss_rate'], missed / total if total else None), 'Pooled miss rate')
            need(close(result['weighted_censored_delay_s'], sum(r['weighted_delay_sum'] for r in rows) / weight if weight else None), 'Censored weights')
            for target, field in [('mean_extra_checks', 'selected_detours'), ('mean_travel_m', 'travel_m'),
                ('mean_service_s', 'service_s'), ('mean_estimator_calls', 'estimator_calls')]:
                need(close(result[target], sum(r['metrics'][field] for r in rows) / 32), 'Actual resource mean')
            need(all(r['metrics']['tasks_violating_exact_gap'] == 0 for r in rows), 'Protected main task overdue')
        baselines = [r for r in main_rows if r['specification']['controller'] == 'tour' and r['specification']['estimator'] == estimator]
        missed_segments = [s for r in baselines for s in r['evaluator_probe']['segments'] if not s['baseline_detected']]
        categories = Counter(s['opportunity_class'] for s in missed_segments)
        result = aggregate['opportunity'][estimator]
        need(result['missed_segment_records'] == len(missed_segments) and result['classes'] == dict(categories), 'Local witness classes recomputed')
        need(result['missed_with_any_prespecified_cue_witness'] == sum(r['any_cue_witness_count'] > 0 for r in missed_segments), 'Any cue versus unconfirmed cue distinguished')
        for baseline in baselines:
            for segment in baseline['evaluator_probe']['segments']:
                if segment['baseline_detected'] or segment['first_witness'] is None:
                    continue
                witness = segment['first_witness']
                case = baseline['case'] + '--branch--' + str(segment['segment_index'])
                expected_branches[case] = {'baseline_case': baseline['case'], 'segment_index': segment['segment_index'],
                    'intervention': {'dispatch_time_s': witness['dispatch_time_s'], 'agent': witness['agent'], 'task': segment['task']}}
    branches = read(ev / 'branch-index.json')
    need({r['case'] for r in branch_rows} == set(expected_branches) == {r['case'] for r in branches}, 'Exactly first-witness branches and no outcome-driven extras')
    for branch in branches:
        need(all(branch[k] == value for k, value in expected_branches[branch['case']].items()), 'Exact first witness source')
        old, new = branch['baseline_intervals'], branch['branch_intervals']
        need(len(old) == len(new), 'Same physical anomaly catalogue')
        recovered = sum(not a['detected'] and b['detected'] for a, b in zip(old, new))
        lost = sum(a['detected'] and not b['detected'] for a, b in zip(old, new))
        need(branch['target_detected'] and branch['total_missed_delta'] == lost - recovered, 'Actual target recovery and collateral misses')
        need(lookup[branch['case']]['metrics']['tasks_violating_exact_gap'] == 0, 'Protected counterfactual overdue')
    for estimator in ('nominal', 'history'):
        rows = [b for b in branches if lookup[b['baseline_case']]['specification']['estimator'] == estimator]
        result = aggregate['opportunity'][estimator]
        need(result['actual_one_detour_branches'] == len(rows), 'Separate branch count')
        expected = []
        for b in rows:
            recovered = sum(not a['detected'] and c['detected'] for a, c in zip(b['baseline_intervals'], b['branch_intervals']))
            lost = sum(a['detected'] and not c['detected'] for a, c in zip(b['baseline_intervals'], b['branch_intervals']))
            expected.append({'case': b['case'], 'baseline_case': b['baseline_case'], 'target_segment_index': b['segment_index'],
                'recovered_segments': recovered, 'lost_segments': lost, 'total_missed_delta': b['total_missed_delta'],
                'travel_delta_m': b['travel_delta_m'], 'service_delta_s': b['service_delta_s']})
        need(result['branch_tradeoffs'] == expected, 'Full branch tradeoff table')
        need(result['branches_with_collateral_miss'] == sum(r['lost_segments'] > 0 for r in expected), 'Collateral misses')
        need(result['branches_net_fewer_missed'] == sum(r['total_missed_delta'] < 0 for r in expected), 'Net branch effects')
    need(aggregate['main_cases'] == 192 and aggregate['all_actual_cases'] == len(index) and aggregate['evaluator_branches'] == len(branches), 'Aggregate scope')
    demo = read(ev / 'report/demo-selection.json')
    need(demo['baseline_case'] == (branches[0]['baseline_case'] if branches else next(r['case'] for r in main_rows if r['specification']['controller'] == 'tour')), 'Fixed example selection order')
    cases = set(demo['control_cases'].values())
    if demo['branch_case']:
        need(demo['branch_case'] == branches[0]['case'], 'First saved branch example')
        cases.add(demo['branch_case'])
    for case in cases:
        path = ev / 'demo' / (case + '.json.gz')
        need(sha(path) == lookup[case]['sha256'], 'Exact declared raw demo bytes')
        record = json.loads(gzip.decompress(path.read_bytes()))
        for gap in record['exact_task_gaps']:
            times = [0.] + sorted(e['time_s'] for e in record['inspection_events'] if e['task'] == gap['task']) + [1800.]
            maximum = max(b - a for a, b in zip(times, times[1:]))
            need(maximum == gap['maximum_gap_including_initial_and_terminal_s'] and maximum <= record['workflow_contract']['deadlines_s'][gap['task']], 'Exact initial/interior/terminal demo gaps')
    figures = []
    for path in sorted((ev / 'report').glob('*.png')):
        raw = path.read_bytes();need(raw[:8] == b'\x89PNG\r\n\x1a\n', 'PNG format')
        pixels = struct.unpack('>II', raw[16:24]);need(pixels[0] >= 1500 and pixels[1] >= 700, 'Readable figure resolution')
        figures.append({'file': path.name, 'pixels': list(pixels), 'sha256': sha(path)})
    need(len(figures) == 3, 'Three figures')
    result = {'source_archives_verified': len(manifests), 'current_source_members_verified': current,
        'original_transferred_files': len(transfer), 'complete_tests_no_skips': 167,
        'main_cases': 192, 'independent_one_detour_branches': len(branches), 'all_actual_cases': len(index),
        'physical_processes': 16, 'all_case_statistics_probe_classes_and_branch_tradeoffs_recomputed': True,
        'all_candidate_masks_and_first_witness_scope_checked_in_full_replay': True,
        'original_failed_comparator_audit_preserved': ids['superseded_smoke_audit'], 'figures': figures,
        'scope': 'Portable original-file/source/statistic verification; independent whole-state legal enumeration and actual branch replay in Slurm audit'}
    with args.output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, indent=2);f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
