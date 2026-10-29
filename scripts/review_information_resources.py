"""Recompute bounded B4-A allocations, including gates and pilot revisions."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re


def need(value, message):
    if not value:
        raise RuntimeError(message)


def duration(value):
    days, value = value.split('-', 1) if '-' in value else ('0', value)
    fields = [float(v) for v in value.split(':')]
    if len(fields) == 2:
        fields.insert(0, 0.)
    need(len(fields) == 3, 'Unexpected Slurm duration format')
    return int(days) * 86400 + fields[0] * 3600 + fields[1] * 60 + fields[2]


def review(path):
    rows = list(csv.DictReader(path.read_text(encoding='utf-8').splitlines(), delimiter='|'))
    jobs = [v for v in rows if '.' not in v['JobID']]
    expected = {'130570', '130571', '130573', '130574', '130582', '130586'}
    expected.update(f'130576_{i}' for i in range(8))
    need(len(jobs) == 14 and {v['JobID'] for v in jobs} == expected, 'Complete B4-A allocation ledger')
    events = []
    for job in jobs:
        need(job['State'] == 'COMPLETED' and job['ExitCode'] == '0:0', 'Non-successful audit job')
        need(job['AllocCPUS'] == '2' and job['ReqMem'] == '8G' and 'gpu' not in job['AllocTRES'].lower(), 'Frozen CPU-only allocation')
        need(job['Partition'] == 'cu', 'Declared CPU partition')
        elapsed = int(job['ElapsedRaw'])
        limit = 600 if job['JobID'] in ('130570', '130573', '130586') else 900 if job['JobID'] in ('130571', '130574') else 2400
        need(elapsed <= limit and int(job['CPUTimeRAW']) == 2 * elapsed, 'Frozen time/CPU cap')
        start, end = (datetime.fromisoformat(job[k]) for k in ('Start', 'End'))
        need((end - start).total_seconds() == elapsed, 'Slurm wall clock mismatch')
        events.extend(((start, 1), (end, -1)))
    peak = active = 0
    for _, delta in sorted(events, key=lambda v: (v[0], v[1])):
        active += delta
        peak = max(peak, active)
    need(active == 0 and peak <= 2, 'Declared maximum concurrency')
    seconds = sum(int(v['CPUTimeRAW']) for v in jobs)
    need(seconds <= 960 * 60, 'Total allocation cap')
    memory = []
    for row in rows:
        if not row['MaxRSS']:
            continue
        match = re.fullmatch(r'([0-9.]+)([KMGT]?)', row['MaxRSS'])
        need(match is not None, 'Unexpected RSS unit')
        memory.append(float(match[1]) * {'': 1, 'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4}[match[2]])
    return {'allocations': jobs, 'allocation_count': len(jobs), 'allocated_CPU_minutes': seconds / 60,
            'actual_TotalCPU_seconds': sum(duration(v['TotalCPU']) for v in jobs),
            'maximum_RSS_MiB': max(memory) / 1024**2, 'maximum_concurrent_allocations': peak,
            'GPU_allocations': 0, 'all_frozen_resource_caps_verified': True,
            'scope': 'Only B4-A listed jobs, all revisions and pilots included; batch steps not added twice. Future B4-B research not started.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.ledger)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'allocations'}))


if __name__ == '__main__':
    main()
