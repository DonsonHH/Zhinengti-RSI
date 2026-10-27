"""Post-hoc duration/role audit of every main/history/fixed-tour case."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    index = json.loads(args.index.read_text())
    selected = [r for r in index if r['specification']['group'] == 'main'
                and r['specification']['estimator'] == 'history'
                and r['specification']['method'] == 'cycle']
    if len(selected) != 36:
        raise RuntimeError('All36 baseline conditions required')
    by_role, by_duration, per_case = {}, {}, []
    def add(target, key, event):
        row = target.setdefault(key, {'segments': 0, 'missed': 0, 'weight': 0., 'weighted_censor_delay': 0.})
        row['segments'] += 1
        row['missed'] += not event['detected']
        row['weight'] += event['weight']
        row['weighted_censor_delay'] += event['weight'] * event['delay_or_censor_s']
    for row in selected:
        path = args.cases / Path(row['record']).name
        raw = path.read_bytes()
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise RuntimeError('Full baseline record changed')
        record = json.loads(gzip.decompress(raw))
        if record['specification'] != row['specification'] or record['metrics'] != row['metrics']:
            raise RuntimeError('Case/index mismatch')
        for event in record['anomaly_intervals']:
            duration = event['end_s'] - event['onset_s']
            bucket = '15_to_under60s' if duration < 60 else '60_to_under300s' if duration < 300 else '300s_or_more'
            add(by_role, event['kind'], event)
            add(by_duration, bucket, event)
        per_case.append({'case': row['case'], 'sha256': row['sha256'],
                         'segments': len(record['anomaly_intervals']),
                         'missed': sum(not e['detected'] for e in record['anomaly_intervals']),
                         'anomaly_intervals': record['anomaly_intervals']})
    result = {'selection': 'Every36 main/history/cycle condition, no case filtering by result',
              'post_hoc_reason': 'Explain nonzero fixed-tour misses after full test scores; no policy or threshold change',
              'by_role': by_role, 'by_segment_duration': by_duration, 'per_case_fingerprints': per_case,
              'scope': 'Duration bins are descriptive, not new event thresholds or causal proof of all algorithm failure mechanisms'}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'per_case_fingerprints'}, indent=2))


if __name__ == '__main__':
    main()
