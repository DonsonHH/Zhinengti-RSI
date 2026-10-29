"""Exact B4-B0 allocation ledger; every gate/pilot/review revision retained."""
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


def seconds(value):
    day, value = value.split('-', 1) if '-' in value else ('0', value)
    fields = [float(v) for v in value.split(':')]
    if len(fields) == 2:
        fields.insert(0, 0.)
    need(len(fields) == 3, 'Unexpected Slurm CPU duration')
    return int(day) * 86400 + fields[0] * 3600 + fields[1] * 60 + fields[2]


def review(path):
    rows = list(csv.DictReader(path.read_text(encoding='utf-8').splitlines(), delimiter='|'))
    jobs = [v for v in rows if '.' not in v['JobID']]
    expected = {'130589', '130590', '130591', '130592', '130595', '130596',
                '130602', '130607', '130608'} | {f'130597_{i}' for i in range(8)}
    need(len(jobs) == 17 and {v['JobID'] for v in jobs} == expected, 'All B4-B0 job allocations included once')
    events = []
    for job in jobs:
        need(job['State'] == 'COMPLETED' and job['ExitCode'] == '0:0', 'Nonterminal/failed data job')
        need(job['AllocCPUS'] == '2' and job['ReqMem'] == '8G' and job['Partition'] == 'cu', 'Frozen CPU/memory/partition')
        need('gpu' not in job['AllocTRES'].lower(), 'Unexpected GPU allocation')
        elapsed = int(job['ElapsedRaw'])
        limit = 600 if job['JobID'] in ('130589', '130591', '130595', '130607') else 1800
        need(elapsed <= limit and int(job['CPUTimeRAW']) == 2 * elapsed, 'Frozen time and allocated CPU accounting')
        start, end = (datetime.fromisoformat(job[key]) for key in ('Start', 'End'))
        need((end - start).total_seconds() == elapsed, 'Slurm elapsed accounting')
        events.extend(((start, 1), (end, -1)))
    active = peak = 0
    for _, change in sorted(events, key=lambda v: (v[0], v[1])):
        active += change
        peak = max(peak, active)
    need(active == 0 and peak <= 2, 'Frozen maximum concurrency')
    cpu = sum(int(v['CPUTimeRAW']) for v in jobs)
    need(cpu <= 960 * 60, 'B0 allocation cap exceeded')
    rss = []
    for row in rows:
        if row['MaxRSS']:
            match = re.fullmatch(r'([0-9.]+)([KMGT]?)', row['MaxRSS'])
            need(match is not None, 'Unknown MaxRSS unit')
            rss.append(float(match[1]) * {'': 1, 'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4}[match[2]])
    return {'allocations': jobs, 'allocation_count': len(jobs), 'allocated_CPU_minutes': cpu / 60,
            'actual_TotalCPU_seconds': sum(seconds(v['TotalCPU']) for v in jobs),
            'maximum_RSS_MiB': max(rss) / 1024**2, 'maximum_concurrent_allocations': peak,
            'GPU_allocations': 0, 'all_frozen_B0_resource_caps_verified': True,
            'scope': 'Only B4-B0 data preparation, all gates/pilots and superseded verifier/report included. Batch steps not double counted. B4-B1 generation/training not started.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.ledger)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'allocations'}))


if __name__ == '__main__':
    main()
