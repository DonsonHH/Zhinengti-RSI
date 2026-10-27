"""Local current-source, transport, grouping and demonstration verification."""
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--research', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    base, root = args.research, args.workspace
    evidence = base / 'evidence'
    names = {'source_manifest_v1.json': 'source-v1.tar.gz',
             'source_manifest_v2.json': 'source-v2.tar.gz',
             'analysis_source_manifest_v1.json': 'analysis-source-v1.tar.gz',
             'package_source_manifest_v1.json': 'package-source-v1.tar.gz',
             'package_source_manifest_v2.json': 'package-source-v2.tar.gz'}
    for manifest_name, archive_name in names.items():
        manifest = json.loads((base / manifest_name).read_text(encoding='utf-8'))
        archive = base / 'source' / archive_name
        require(sha(archive) == manifest['archive_sha256'], 'Frozen source archive')
        with tarfile.open(archive) as tar:
            members = [v for v in tar.getmembers() if v.isfile()]
            require({m.name for m in members} == set(manifest['files']), 'Frozen source members')
            for member in members:
                require(not member.name.startswith('/') and '..' not in Path(member.name).parts, 'Archive path')
                raw = tar.extractfile(member).read()
                require(hashlib.sha256(raw).hexdigest() == manifest['files'][member.name], 'Frozen source member bytes')
        if manifest_name not in ('source_manifest_v1.json', 'package_source_manifest_v1.json'):
            for name, value in manifest['files'].items():
                require(hashlib.sha256((root / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == value,
                        'Current run source changed: ' + name)
    transfer = json.loads((evidence / 'transfer-manifest.json').read_text())
    for name, value in transfer.items():
        path = evidence / name
        require(path.stat().st_size == value['bytes'] and sha(path) == value['sha256'], 'Transport bytes: ' + name)
    indices = []
    for name, count in [('training-data-index.json.gz', 280), ('test-data-index.json.gz', 318)]:
        rows = json.loads(gzip.decompress((evidence / name).read_bytes()))
        require(len(rows) == count, 'Dataset case count')
        for row in rows:
            for event in row['effective_inspection_events']:
                require(event['access'] == 'external_sensing' and event['calibrated'], 'Qualified calibrated event')
                require(event['role'] == {'thermal': 'thermal', 'electrical': 'electrical',
                                         'cooling': 'cooling', 'control': 'control'}[event['kind']], 'Event qualification')
                require(0 < event['time_s'] <= 1800 and event['service_s'] > 0, 'Event timing')
        indices.extend(rows)
    metrics = json.loads((evidence / 'metrics.json').read_text())
    require(len(metrics) == 3498, 'Metric coverage')
    aggregate = json.loads((evidence / 'report/aggregate.json').read_text())
    for group, methods in aggregate['groups'].items():
        split, condition, *extra = group.split('/')
        subset = [r for r in metrics if r['split'] == split and r['condition'] == condition
                  and (not extra or '+' in r['scenario'])]
        for method, summary in methods.items():
            means = []
            for seed, record in summary['per_training_seed'].items():
                items = [r for r in subset if r['method'] == method and
                         r['training_seed'] == (None if seed == 'nonlearned' else int(seed))]
                require(len(items) == record['cases'], 'Per-seed group coverage')
                for key, value in record['metrics'].items():
                    if key.startswith('pooled_'):
                        continue
                    if key == 'incremental_CPU_ms_per_frame':
                        calculated = 1000 * statistics.mean(r['mean_incremental_CPU_seconds'] for r in items)
                    elif key in ('tp', 'fp', 'fn', 'hot_count'):
                        calculated = sum(r['metrics'][key] for r in items)
                    else:
                        values = [r['metrics'][key] for r in items if r['metrics'][key] is not None]
                        calculated = statistics.mean(values) if values else None
                    require(calculated is None and value is None or
                            calculated is not None and value is not None and math.isclose(calculated, value, rel_tol=1e-10, abs_tol=1e-10),
                            'Independent aggregate: ' + key)
                values = record['metrics']
                for key, denominator in [('pooled_precision', values['tp'] + values['fp']),
                                         ('pooled_recall', values['tp'] + values['fn'])]:
                    expected = values['tp'] / denominator if denominator else None
                    require(values[key] == expected, 'Pooled hotspot metric')
                means.append(values['rmse_c'])
            require(math.isclose(statistics.mean(means), summary['mean_case_RMSE_c'], rel_tol=1e-12), 'Training-seed mean')
            if len(means) == 3:
                require(math.isclose(statistics.stdev(means), summary['training_seed_SD_of_mean_RMSE_c'], rel_tol=1e-12), 'Training-seed SD')
    for name, value in aggregate['paired_history_effect'].items():
        group = aggregate['groups'][name]
        differences = [group['cvae_history']['per_training_seed'][str(s)]['metrics']['rmse_c'] -
                       group['cvae_current']['per_training_seed'][str(s)]['metrics']['rmse_c'] for s in (0, 1, 2)]
        require(differences == value['history_minus_current_per_training_seed'], 'Paired history differences')
        require(statistics.mean(differences) == value['mean'], 'Paired mean')
    gw = [r for r in indices if r['split'] == 'gw']
    require(len(gw) == 12 and max(r['peak_source_input_MW'] for r in gw) == aggregate['gw_power']['peak_source_input_MW'], 'Actual GW input')
    require(all(r['IT_nameplate_MW'] == 800 and r['global_supply_budget_MW'] == 1000 and not r['budget_limiter_implemented'] for r in gw), 'GW meanings')
    audit = json.loads((evidence / 'full-audit.json').read_text())
    require(audit['all_dataset_cases_regenerated'] == 598 and audit['independent_method_condition_metrics'] == 3498, 'Independent full audit')
    require(audit['independent_physical_balances']['steps'] == 215280 and audit['selected_checkpoints_read'] == 6, 'Full physics and actual weights')
    proof = json.loads((evidence / 'version-proof.json').read_text())
    require(proof['cases_compared'] == 318 and proof['all_six_CVAE_outputs_identical_after_GP_revision'], 'GP-only revision')
    with (evidence / 'slurm-final.psv').open() as handle:
        jobs = {row['JobID']: row for row in csv.DictReader(handle, delimiter='|')}
    ids = json.loads((evidence / 'job-identities.json').read_text())
    for name, value in ids.items():
        if isinstance(value, int):
            if name == 'package_failed':
                require(jobs[str(value)]['State'] == 'FAILED' and jobs[str(value)]['ExitCode'] == '1:0', 'Preserved first packaging failure')
            else:
                require(jobs[str(value)]['State'] == 'COMPLETED' and jobs[str(value)]['ExitCode'] == '0:0', 'Terminal job')
    log = (evidence / f"jobs/evaluate-v2-{ids['evaluate_v2']}.err").read_text()
    require('Ran 107 tests' in log and log.rstrip().endswith('OK') and 'skipped' not in log, 'Full tests')
    figures = []
    for path in sorted((evidence / 'report').glob('*.png')):
        raw = path.read_bytes()
        require(raw[:8] == b'\x89PNG\r\n\x1a\n', 'PNG signature')
        width, height = struct.unpack('>II', raw[16:24])
        require(width > 1000 and height > 650, 'Readable static figure dimensions')
        figures.append({'file': path.name, 'pixels': [width, height], 'sha256': sha(path)})
    require(len(figures) == 4, 'Four figures')
    result = {'current_sources_and_all_five_archives_verified': True,
              'transferred_files': len(transfer), 'full_tests_no_skips': 107,
              'data_cases': 598, 'metric_records': 3498, 'all_group_statistics_and_pairings_recomputed': True,
              'all_seven_required_terminal_jobs_successful': True,
              'first_packaging_failure_preserved_and_resolved': True, 'figures': figures,
              'scope': 'Full numeric/selected-weight replay is in independent Slurm audit; local verification covers transport, groups, source and final job evidence'}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
