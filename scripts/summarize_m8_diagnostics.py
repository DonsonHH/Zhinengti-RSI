"""M8 descriptive first-coverage windows and finite-case feasibility witnesses."""
import argparse
import gzip
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def describe(values):
    return {'per_training_seed': values, 'mean': statistics.mean(values),
            'sample_sd_across_training_seeds': statistics.stdev(values) if len(values) > 1 else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    variants = ('rack_local', 'rack_relation', 'mixed_local', 'mixed_relation')
    groups = {}
    summaries = {}
    for variant in variants:
        summaries[variant] = [read(args.results / f'{variant}-seed-{seed}/summary.json.gz') for seed in range(3)]
        for method in ('learned', 'untrained_network'):
            groups[f'{variant}/{method}'] = [summary['test'][method]['cases'] for summary in summaries[variant]]
    planners = read(args.results / 'common-planners/summary.json.gz')
    for method, group in planners['test'].items():
        groups[f'planner/{method}'] = [group['cases']]
    windows = {}
    for name, runs in groups.items():
        windows[name] = {
            'first24_prediction_mse': describe([statistics.mean(statistics.mean(row['prediction_mse_series'][:24]) for row in rows) for rows in runs]),
            'decision_fraction_after_all_targets_measured': describe([
                statistics.mean((96 - row['first_all_seen_step']) / 96 if not row['first_all_seen_censored'] else 0 for row in rows) for rows in runs]),
            'incomplete_coverage_conditions': sum(row['first_all_seen_censored'] for rows in runs for row in rows),
        }
    witnesses = []
    first_rows = groups['rack_local/learned'][0]
    for index, case in enumerate(first_rows):
        names = [f'{name}/seed-{seed}' for name, runs in groups.items() for seed, rows in enumerate(runs)
                 if rows[index]['target_coverage'] == 1 and rows[index]['deadline_violation'] == 0]
        witnesses.append({key: case[key] for key in ('family', 'layout_seed', 'profile', 'noise', 'startup_violation_lower_bound')} |
                         {'zero_violation_full_coverage_methods': names, 'recurrent_feasibility_unknown_without_witness': not bool(names)})
    directions = {}
    for name, before, after in [('relation_minus_local_rack', 'rack_local', 'rack_relation'),
                                ('relation_minus_local_mixed', 'mixed_local', 'mixed_relation'),
                                ('mixed_minus_rack_local', 'rack_local', 'mixed_local'),
                                ('mixed_minus_rack_relation', 'rack_relation', 'mixed_relation')]:
        values = [right['test']['learned']['metrics']['deadline_violation'] - left['test']['learned']['metrics']['deadline_violation']
                  for left, right in zip(summaries[before], summaries[after])]
        directions[name] = describe(values) | {'better_seeds': sum(value < 0 for value in values),
                                               'worse_seeds': sum(value > 0 for value in values),
                                               'equal_seeds': sum(value == 0 for value in values)}
    output = {'descriptive_only': True, 'training_seeds_per_group': 3, 'windows': windows,
              'condition_witnesses': witnesses, 'zero_violation_witness_conditions': sum(bool(row['zero_violation_full_coverage_methods']) for row in witnesses),
              'positive_startup_lower_bound_conditions': sum(row['startup_violation_lower_bound'] > 0 for row in witnesses),
              'max_startup_violation_lower_bound': max(row['startup_violation_lower_bound'] for row in witnesses),
              'paired_factor_directions_all_conditions': directions,
              'interpretation': 'Static field with permanent observed-value overwrite; post-full-coverage estimator output cannot alter target estimates, but prior action/training effects remain possible'}
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(output, handle, indent=2)
        handle.write('\n')
    print('M8_DIAGNOSTICS_WRITTEN', len(groups), 'groups;', len(witnesses), 'conditions')


if __name__ == '__main__':
    main()
