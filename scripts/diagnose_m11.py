"""Predeclared resource, scenario and guard-use stratification for M11."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def pooling(rows):
    total = sum(row['metrics']['anomaly_segments'] for row in rows)
    missed = sum(row['metrics']['missed_segments'] for row in rows)
    weight = sum(row['event_weight_sum'] for row in rows)
    return {'episodes': len(rows), 'no_defined_anomaly_episodes': sum(r['metrics']['anomaly_segments'] == 0 for r in rows),
            'miss_rate': missed / total if total else None,
            'weighted_censored_delay_s': sum(r['weighted_delay_sum'] for r in rows) / weight if weight else None,
            'mean_overdue_fraction': statistics.mean(r['metrics']['mean_overdue_fraction'] for r in rows),
            'mean_coverage': statistics.mean(r['metrics']['final_professional_coverage'] for r in rows),
            'mean_guard_fraction_of_completed_tasks': statistics.mean(
                r['metrics']['guard_forced_choices'] / max(1, r['metrics']['completed_inspections']) for r in rows),
            'mean_decision_CPU_seconds': statistics.mean(r['metrics']['decision_CPU_seconds'] for r in rows),
            'mean_estimator_CPU_seconds': statistics.mean(r['metrics']['history_estimator_inference_s'] for r in rows)}


def group(rows):
    result = {}
    for mode in ('nominal', 'history'):
        result[mode] = {}
        methods = sorted({r['specification']['method'] for r in rows if r['specification']['estimator'] == mode})
        for method in methods:
            selected = [r for r in rows if r['specification']['estimator'] == mode and r['specification']['method'] == method]
            seeds = sorted({r['specification']['training_seed'] for r in selected}, key=lambda s: -1 if s is None else s)
            per_seed = {'rule' if seed is None else str(seed): pooling(
                [r for r in selected if r['specification']['training_seed'] == seed]) for seed in seeds}
            means = {}
            for key in next(iter(per_seed.values())):
                values = [v[key] for v in per_seed.values() if v[key] is not None]
                means[key] = statistics.mean(values) if values else None
            result[mode][method] = {'per_training_seed': per_seed, 'mean': means}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.evidence / 'report/compact-case-metrics.json.gz'
    rows = json.loads(gzip.decompress(source.read_bytes()))
    if len(rows) != 1136:
        raise RuntimeError('Complete final suite required')
    strata = {}
    for level in ('scarce', 'provisioned'):
        strata['gw/' + level] = group([r for r in rows if r['specification']['group'] == 'gw' and r['specification']['resources'] == level])
    for scenario in sorted({r['specification']['scenario'] for r in rows if r['specification']['group'] == 'main'}):
        strata['main/' + scenario] = group([r for r in rows if r['specification']['group'] == 'main' and r['specification']['scenario'] == scenario])
    process_effects = {}
    for row in rows:
        key = row['process_key']
        values = (row['metrics']['anomaly_segments'], row['metrics']['unserved_IT_kwh'])
        if key in process_effects and process_effects[key] != values:
            raise RuntimeError('Inspection unexpectedly changed the physical label/service consequences')
        process_effects[key] = values
    blind_consequences = [{'process_key': key, 'unserved_IT_kwh': values[1]} for key, values in process_effects.items() if values[0] == 0 and values[1] > 0]
    result = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'scope': 'Resource/scenario split specified before final score inspection; no new thresholds, policy or training',
              'strata': strata,
              'physical_anomaly_counts_and_service_loss_identical_across_methods': True,
              'unique_process_conditions': len(process_effects),
              'service_loss_without_defined_task_anomaly': blind_consequences,
              'interpretation': 'Electrical confirmation currently covers contact heat; a service gap without one of the defined task anomalies is outside the confirmation metric, not a proven successful power diagnosis'}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print('M11_RESOURCE_SCENARIO_DIAGNOSTICS_COMPLETE', len(strata), 'strata')


if __name__ == '__main__':
    main()
