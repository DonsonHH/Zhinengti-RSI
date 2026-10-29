"""Read-only resource and illustrative forecast checks from archived evidence."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def need(value, message):
    if not value:
        raise RuntimeError(message)


def cpu_seconds(value):
    days, clock = value.split('-', 1) if '-' in value else ('0', value)
    fields = [float(v) for v in clock.split(':')]
    need(len(fields) in (2, 3), 'Unexpected Slurm CPU duration')
    if len(fields) == 2:
        fields.insert(0, 0.)
    return int(days) * 86400 + fields[0] * 3600 + fields[1] * 60 + fields[2]


def resource_ledger(path, protocol):
    rows = list(csv.DictReader(path.read_text(encoding='utf-8').splitlines(), delimiter='|'))
    jobs = [v for v in rows if '.' not in v['JobID']]
    expected = {'130522', '130526', '130530', '130545', '130556', '130557'}
    for job, count in ((130532, 2), (130534, 2), (130546, 2), (130548, 2),
                       (130549, 2), (130552, 4), (130553, 4)):
        expected.update(f'{job}_{i}' for i in range(count))
    need(len(jobs) == 24 and {v['JobID'] for v in jobs} == expected,
         'Complete task-only Slurm allocations, without double counting batch steps')
    failures = {'130526', '130534_0', '130534_1'}
    caps = protocol['resource_caps']
    events = []
    for job in jobs:
        failed = job['JobID'] in failures
        need((job['State'], job['ExitCode']) == ('FAILED', '1:0') if failed else
             (job['State'], job['ExitCode']) == ('COMPLETED', '0:0'), 'Unexpected job state')
        elapsed, cpus = int(job['ElapsedRaw']), int(job['AllocCPUS'])
        need(cpus == caps['cpus_per_task'] and elapsed <= caps['time_limit_each_min'] * 60,
             'Allocation exceeds frozen CPU/time cap')
        need(job['ReqMem'] == str(caps['memory_gb']) + 'G', 'Frozen memory request')
        need(int(job['CPUTimeRAW']) == elapsed * cpus, 'Allocated CPU accounting')
        need('gpu' not in job['AllocTRES'].lower(), 'Unexpected GPU allocation')
        start, end = (datetime.fromisoformat(job[k]) for k in ('Start', 'End'))
        need((end - start).total_seconds() == elapsed, 'Slurm elapsed mismatch')
        events.extend(((start, 1), (end, -1)))
    active = peak = 0
    for _, change in sorted(events, key=lambda v: (v[0], v[1])):
        active += change
        peak = max(peak, active)
    need(active == 0 and peak <= caps['concurrency'], 'Frozen concurrency cap')
    allocated = sum(int(v['CPUTimeRAW']) for v in jobs)
    need(allocated <= caps['maximum_allocated_CPU_minutes_including_one_full_replay'] * 60,
         'Total allocated CPU exceeds frozen cap')
    rss = []
    for v in rows:
        if not v['MaxRSS']:
            continue
        m = re.fullmatch(r'([0-9.]+)([KMGT]?)', v['MaxRSS'])
        need(m is not None, 'Unrecognized Slurm RSS unit')
        rss.append(float(m[1]) * {'': 1, 'K': 1024, 'M': 1024**2,
                                  'G': 1024**3, 'T': 1024**4}[m[2]])
    return {'allocations': jobs, 'allocation_count': len(jobs),
            'retained_failed_allocations': sorted(failures),
            'allocated_CPU_seconds': allocated, 'allocated_CPU_minutes': allocated / 60,
            'actual_TotalCPU_seconds': sum(cpu_seconds(v['TotalCPU']) for v in jobs),
            'maximum_RSS_MiB': max(rss) / 1024**2,
            'maximum_concurrent_allocations': peak, 'GPU_allocations': 0,
            'frozen_allocated_CPU_minute_cap': caps['maximum_allocated_CPU_minutes_including_one_full_replay'],
            'within_all_frozen_resource_caps': True,
            'accounting_scope': 'This C1-B3.2 execution only, includes gates, failed jobs, recovery, reports and packaging. Batch steps are not added twice. Allocated CPU time is not elapsed wall time.'}


def explain_examples(base):
    selection = json.loads((base / 'evidence/explanatory-selection.json').read_bytes())
    folder = base / 'evidence/explanatory-cases'
    for row in selection['files']:
        need(hashlib.sha256((folder / row['local_name']).read_bytes()).hexdigest() == row['sha256'],
             'Illustrative source SHA mismatch')
    result = {}
    for alias in ('development-positive', 'validation-waiting'):
        record = json.loads(gzip.decompress((folder / (alias + '.json.gz')).read_bytes()))
        with np.load(folder / (alias + '.npz')) as data:
            arrays = {k: data[k].copy() for k in data.files}
        points = []
        for decision in record['stopping_decisions']:
            future = decision.get('future_selected')
            if not future:
                continue
            j = future['task']
            tick = min(359, int(future['finish_s'] / 5) - 1)
            temperature = None
            if j < 128:
                temperature = float(arrays['rack_temperature_c'][tick, j])
                actual = max(temperature - 65., 0.) / 20
            elif j < 144:
                actual = max(float(arrays['contact_temperature_c'][tick, j - 128]) - 60., 0.) / 30
            elif j < 160:
                actual = max(1 - float(arrays['rack_flow_kg_s'][tick].reshape(16, 8).sum(1)[j - 144]) / 33.6, 0.)
            else:
                actual = float(abs((arrays['public_sensor_temperature_c'][tick] -
                                    arrays['rack_temperature_c'][tick]).reshape(16, 8).mean(1)[j - 160]) > 1)
            points.append({'decision_time_s': decision['time_s'], 'action': decision['action'],
                           'q_check': decision['q_check'], 'q_wait': decision['q_wait'],
                           'future_task': j, 'future_finish_s': future['finish_s'],
                           'future_target_source': future['target_source'],
                           'predicted_target_signal': future['target_signal'],
                           'reference_target_signal': actual, 'reference_temperature_c': temperature})
        errors = [v['predicted_target_signal'] - v['reference_target_signal'] for v in points]
        positives = [v for v in points if v['action'] == 'WAIT' and v['q_check'] > 0]
        result[alias] = {'formal_identity': record['formal_identity'],
                         'actual_checks': record['metrics']['selected_detours'],
                         'positive_current_value_wait_frames': len(positives),
                         'first_three_positive_wait_frames': positives[:3],
                         'actual_CHECK_scores': [v['selected'] for v in record['stopping_decisions'] if v.get('selected')],
                         'forecast_points': len(points),
                         'forecast_positive_reference_zero': sum(v['predicted_target_signal'] > 0 and v['reference_target_signal'] <= 0 for v in points),
                         'forecast_signal_MAE': sum(abs(v) for v in errors) / len(errors) if errors else None}
    return {'selection_scope': selection['scope'], 'examples': result,
            'scope': 'Post hoc examples only. Recomputes instantaneous reference signals from archived physical arrays; no policy rerun, new selection or estimator intervention. Forecast score and realized whole-mission utility have different units.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace.resolve()
    base = root / 'research/2026-10-29-m12c1b32-formal'
    protocol = json.loads((base / 'evidence/protocol-original.json').read_bytes())
    result = {'resources': resource_ledger(base / 'evidence/slurm-final.psv', protocol),
              'illustrative_forecast_checks': explain_examples(base)}
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result['resources'].items() if k != 'allocations'}))


if __name__ == '__main__':
    main()
