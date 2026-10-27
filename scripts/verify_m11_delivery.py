"""Local M11 source, terminal state, transport, groups and demo verification."""
from __future__ import annotations

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

import numpy as np


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    return a is None and b is None or a is not None and b is not None and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--research', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, base = args.workspace, args.research
    ev = base / 'evidence'
    archives = {'source_manifest_v1.json': 'source-v1.tar.gz',
                'source_manifest_v2.json': 'source-v2.tar.gz',
                'analysis_source_manifest_v1.json': 'analysis-source-v1.tar.gz',
                'package_source_manifest_v1.json': 'package-source-v1.tar.gz'}
    for name, filename in archives.items():
        manifest = json.loads((base / name).read_text(encoding='utf-8'))
        archive = base / 'source' / filename
        require(sha(archive) == manifest['archive_sha256'], 'Source archive fingerprint')
        with tarfile.open(archive) as tar:
            files = [m for m in tar.getmembers() if m.isfile()]
            require({m.name for m in files} == set(manifest['files']), 'Source members')
            for member in files:
                raw = tar.extractfile(member).read()
                require(hashlib.sha256(raw).hexdigest() == manifest['files'][member.name], 'Source member fingerprint')
        if name != 'source_manifest_v1.json':
            for rel, fingerprint in manifest['files'].items():
                raw = (root / rel).read_bytes().replace(b'\r\n', b'\n')
                require(hashlib.sha256(raw).hexdigest() == fingerprint, 'Current source drift: ' + rel)
    transfer = json.loads((ev / 'transfer-manifest.json').read_text())
    for rel, record in transfer.items():
        path = ev / rel
        require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], 'Transferred evidence: ' + rel)
    ids = json.loads((ev / 'job-identities.json').read_text())
    with (ev / 'slurm-final.psv').open() as handle:
        jobs = {row['JobID']: row for row in csv.DictReader(handle, delimiter='|')}
    checked = []
    for key, value in ids.items():
        if not isinstance(value, int):
            continue
        names = [str(value) + '_' + str(i) for i in range(6)] if key in ('train_array', 'train_array_v2') else [str(value)]
        for name in names:
            require(jobs[name]['State'] == 'COMPLETED' and jobs[name]['ExitCode'] == '0:0', 'Nonterminal or unsuccessful job ' + name)
            checked.append(name)
    log = (ev / f"jobs/smoke-v2-{ids['smoke_v2']}.err").read_text()
    require('Ran 122 tests' in log and log.rstrip().endswith('OK') and 'skipped' not in log, 'Full 122 tests')
    require(json.loads((ev / 'm10-full-review.json').read_text()) ==
            json.loads((root / 'research/2026-10-27-m10-estimation/evidence/full-audit.json').read_text()), 'M10 fresh audit equality')
    capacity = json.loads((ev / 'capacity-v2/feasibility.json').read_text())
    require(len(capacity) == 6, 'Capacity combinations')
    for row in capacity:
        for role, value in row['roles'].items():
            service = value['tasks'] * (10 if role == 'thermal' else 15 if role == 'control' else 20)
            require(close(value['service_only_utilization'], service / value['agents'] / value['reference_revisit_s']), 'Service load units')
            require(value['service_only_minimum_agents'] == math.ceil(service / value['reference_revisit_s']), 'Service bound')
            actual = all(v['closed_quantized_cycle_s'] <= value['reference_revisit_s'] for v in value['constructed_cycles'])
            require(actual == value['all_constructed_cycles_within_revisit'], 'Cycle certificate flag')
            if row['level'] == 'provisioned':
                require(actual, 'Configured tier has no constructed-tour feasibility certificate')
        if row['level'] == 'scarce':
            require(not row['roles']['thermal']['service_only_necessary_condition'], 'Scarce thermal load must remain infeasible')
    index = json.loads(gzip.decompress((ev / 'case-index.json.gz').read_bytes()))
    require(len(index) == 1136 and len({r['case'] for r in index}) == 1136, 'All formal cases')
    compact = json.loads(gzip.decompress((ev / 'report/compact-case-metrics.json.gz').read_bytes()))
    require(len(compact) == 1136 and {r['case'] for r in compact} == {r['case'] for r in index}, 'Compact cases')
    source = {r['case']: r for r in index}
    for row in compact:
        for key in source[row['case']]:
            require(row[key] == source[row['case']][key], 'Compact/index consistency')
    aggregate = json.loads((ev / 'report/aggregate.json').read_text())
    for group, modes in aggregate['groups'].items():
        for mode, methods in modes.items():
            for method, summary in methods.items():
                subset = [r for r in compact if r['specification']['group'] == group and
                          r['specification']['estimator'] == mode and r['specification']['method'] == method]
                for seed, result in summary['per_training_seed'].items():
                    number = None if seed == 'rule' else int(seed)
                    items = [r for r in subset if r['specification']['training_seed'] == number]
                    require(len(items) == result['cases'], 'Per-seed group size')
                    segments = sum(r['metrics']['anomaly_segments'] for r in items)
                    missed = sum(r['metrics']['missed_segments'] for r in items)
                    weights = sum(r['event_weight_sum'] for r in items)
                    require(segments == result['anomaly_segments'] and missed == result['missed_segments'], 'Pooled event counts')
                    require(close(missed / segments if segments else None, result['pooled_miss_rate']), 'Pooled miss rate')
                    delay = sum(r['weighted_delay_sum'] for r in items) / weights if weights else None
                    require(close(delay, result['weighted_censored_delay_s']), 'Pooled censored delay')
                    mapping = {'mean_overdue_fraction': 'mean_overdue_fraction',
                               'mean_final_coverage': 'final_professional_coverage',
                               'mean_episode_maximum_age_s': 'maximum_inspection_age_s',
                               'mean_travel_m': 'travel_m', 'mean_service_s': 'service_s',
                               'mean_decision_CPU_s': 'decision_CPU_seconds',
                               'mean_estimator_CPU_s': 'history_estimator_inference_s',
                               'mean_guard_forced_choices': 'guard_forced_choices'}
                    for key, metric in mapping.items():
                        require(close(statistics.mean(r['metrics'][metric] for r in items), result[key]), 'Group mean: ' + key)
                    require(max(r['metrics']['maximum_inspection_age_s'] for r in items) == result['maximum_age_s_across_cases'], 'Worst waiting')
                    for role in ('thermal', 'electrical', 'cooling', 'control'):
                        total = sum(r['by_role'][role]['segments'] for r in items)
                        found = sum(r['by_role'][role]['confirmed'] for r in items)
                        require(result['by_role'][role]['segments'] == total and result['by_role'][role]['confirmed'] == found, 'Role event counts')
                        require(close(1 - found / total if total else None, result['by_role'][role]['pooled_miss_rate']), 'Role miss rate')
                        require(close(statistics.mean(r['by_role'][role]['mean_overdue_fraction'] for r in items),
                                      result['by_role'][role]['mean_overdue_fraction']), 'Role overdue')
                        require(close(statistics.mean(r['by_role'][role]['final_coverage'] for r in items),
                                      result['by_role'][role]['mean_final_coverage']), 'Role coverage')
                per_seed = list(summary['per_training_seed'].values())
                for key, value in summary['mean'].items():
                    if key == 'by_role':
                        for role, values in value.items():
                            for metric, average in values.items():
                                samples = [r[key][role][metric] for r in per_seed if r[key][role][metric] is not None]
                                require(close(statistics.mean(samples) if samples else None, average), 'Training-seed role average')
                    else:
                        samples = [r[key] for r in per_seed if r[key] is not None]
                        require(close(statistics.mean(samples) if samples else None, value), 'Training-seed average')
                        require(close(statistics.stdev(samples) if len(samples) > 1 else None, summary['training_seed_SD'][key]), 'Training-seed SD')
    for group, modes in aggregate['paired_guard_minus_free'].items():
        for mode, metrics in modes.items():
            methods = aggregate['groups'][group][mode]
            guarded, free = methods['ppo_guard']['per_training_seed'], methods['ppo_free']['per_training_seed']
            for metric, record in metrics.items():
                values = [guarded[s][metric] - free[s][metric] for s in guarded if guarded[s][metric] is not None and free[s][metric] is not None]
                require(values == record['per_training_seed'], 'Guard paired effects')
                require(close(statistics.mean(values) if values else None, record['mean']), 'Guard paired mean')
                require(close(statistics.stdev(values) if len(values) > 1 else None, record['sample_SD']), 'Guard paired SD')
    audit = json.loads((ev / 'full-audit.json').read_text())
    require(audit['all_test_episodes_replayed'] == 1136 and audit['actual_selected_policy_weights_read'] == 6, 'Full actual policy audit')
    require(audit['graph_paths_roles_zones_reservations_and_service_clocks_verified'] and audit['all_ages_overdue_and_censoring_recomputed'], 'Full task audit')
    for path in sorted((ev / 'demo').glob('*.json.gz')):
        record = json.loads(gzip.decompress(path.read_bytes()))
        row = next(r for r in index if r['specification'] == record['specification'])
        require(sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Exact full demo case')
        for agent in record['team']['per_agent']:
            require(close(agent['travel_m'] + agent['service_s'] + agent['idle_s'], 1795), 'Demo full time budget')
    require(len(list((ev / 'demo').glob('*.json.gz'))) == 4, 'Four declared demo policies')
    figures = []
    for path in sorted((ev / 'report').glob('*.png')):
        raw = path.read_bytes()
        require(raw[:8] == b'\x89PNG\r\n\x1a\n', 'PNG signature')
        width, height = struct.unpack('>II', raw[16:24])
        require(width > 1000 and height > 650, 'Figure resolution')
        figures.append({'file': path.name, 'pixels': [width, height], 'sha256': sha(path)})
    require(len(figures) == 4, 'Four figures')
    result = {'four_source_archives_verified': True, 'current_sources_match_final_run': True,
              'transferred_files': len(transfer), 'successful_terminal_jobs_including_superseded': checked,
              'full_tests_no_skips': 122, 'full_test_episodes': 1136,
              'all_groups_training_seed_SDs_and_paired_guard_effects_recomputed': True,
              'all_resource_certificates_checked': True, 'exact_demo_cases_retrieved': 4, 'figures': figures,
              'scope': 'Local source, transport, terminal-state, grouping and demo verification; full closed-loop policy replay is in independent Slurm audit'}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
