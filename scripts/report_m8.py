"""M8 matched training-distribution and actor-information results."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from audit_m8 import read
from patrol_rsi.m8 import VARIANTS
from patrol_rsi.m7 import PLANNER_SETTINGS


LABELS = {'rack_local': 'Rack / local actor', 'rack_relation': 'Rack / relational actor',
          'mixed_local': 'Mixed / local actor', 'mixed_relation': 'Mixed / relational actor',
          'risk_vae': 'Risk-age / VAE', 'risk_hold': 'Risk-age / hold',
          'matching_vae': 'Team matching / VAE', 'matching_hold': 'Team matching / hold',
          'slack': 'Slack rule', 'pair_cycle': 'Pair-cycle rule'}


def describe(values):
    return {'per_training_seed': values, 'mean': float(np.mean(values)),
            'sample_sd_across_training_seeds': float(np.std(values, ddof=1)) if len(values) > 1 else None}


def accepts(row, subset):
    if subset == 'all':
        return True
    parts = subset.split('/')
    return row['family'] == parts[0] and (len(parts) == 1 or row['profile'] == parts[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    groups, curves, costs = {}, {}, []
    for variant in VARIANTS:
        summaries = [read(args.results / f'{variant}-seed-{seed}') for seed in range(3)]
        for method in ('learned', 'untrained_network'):
            groups[f'{variant}/{method}'] = [summary['test'][method]['cases'] for summary in summaries]
        curves[variant] = []
        for seed, summary in enumerate(summaries):
            curve = [json.loads(line) for line in (args.results / f'{variant}-seed-{seed}/learning.jsonl').read_text().splitlines()]
            curves[variant].append(curve)
            costs.append({'variant': variant, 'seed': seed, 'selected_episode': summary['selected_episode'],
                          'training_plus_validation_seconds': curve[-1]['seconds'],
                          'post_selection_seconds': summary['seconds'] - curve[-1]['seconds'],
                          'parameters': summary['parameters'], 'gradient_updates': summary['gradient_updates'],
                          'behavior_logp_max_error': summary['behavior_logp_max_error']})
    planners = read(args.results / 'common-planners')
    for method in PLANNER_SETTINGS:
        groups[f'planner/{method}'] = [planners['test'][method]['cases']]
    metrics = list(planners['test']['pair_cycle']['metrics'])
    subsets = ['all', 'rack', 'baffle'] + [f'{family}/{profile}' for family in ('rack', 'baffle') for profile in ('balanced', 'tight', 'pressure')]
    tables = {}
    for subset in subsets:
        tables[subset] = {}
        for name, runs in groups.items():
            table = {metric: describe([float(np.mean([row[metric] for row in rows if accepts(row, subset)])) for rows in runs]) for metric in metrics}
            rows = [row for run in runs for row in run if accepts(row, subset)]
            table['worst_case_violation'] = max(row['deadline_violation'] for row in rows)
            table['worst_case_peak_age'] = max(row['peak_age'] for row in rows)
            table['incomplete_coverage_cases'] = sum(row['target_coverage'] < 1 for row in rows)
            tables[subset][name] = table
    effects = {}
    comparisons = [('mixed_minus_rack_local', 'rack_local', 'mixed_local'),
                   ('mixed_minus_rack_relation', 'rack_relation', 'mixed_relation'),
                   ('relation_minus_local_rack', 'rack_local', 'rack_relation'),
                   ('relation_minus_local_mixed', 'mixed_local', 'mixed_relation')]
    for effect, before, after in comparisons:
        effects[effect] = {subset: {metric: describe([float(np.mean([
            right[metric] - left[metric] for left, right in zip(left_rows, right_rows) if accepts(left, subset)]))
            for left_rows, right_rows in zip(groups[f'{before}/learned'], groups[f'{after}/learned'])])
            for metric in metrics} for subset in subsets}
    interaction = {subset: {metric: describe([
        effects['relation_minus_local_mixed'][subset][metric]['per_training_seed'][seed]
        - effects['relation_minus_local_rack'][subset][metric]['per_training_seed'][seed] for seed in range(3)])
        for metric in metrics} for subset in subsets}
    output = {'tables': tables, 'paired_factor_effects': effects, 'interaction': interaction,
              'learning_curves': curves, 'training_costs': costs, 'from_scratch_training_tasks': 12,
              'paired_training_seeds_per_group': 3, 'common_planner_task': 1, 'final_conditions_per_method_seed': 96,
              'total_method_condition_records': 2880, 'RSI': False,
              'scope': 'Matched actor-information input and training-distribution replacement; same full critic, topology and PPO; descriptive statistics, not significance test'}
    (args.output / 'aggregate.json').write_text(json.dumps(output, indent=2))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    for ax, variant in zip(axes.flat, VARIANTS):
        for seed, curve in enumerate(curves[variant]):
            ax.plot([row['episode'] for row in curve], [row['validation']['deadline_violation'] * 100 for row in curve], label=f'Seed {seed}')
        ax.set(title=LABELS[variant], xlabel='Training episode', ylabel='Validation violation (%)')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('M8 equal-budget training; same mixed-layout validation; validation-only checkpoint selection')
    fig.savefig(args.output / 'm8-learning.png', dpi=170)
    fig.savefig(args.output / 'm8-learning.svg')
    plt.close(fig)
    shown = [f'{variant}/learned' for variant in VARIANTS] + [f'planner/{method}' for method in PLANNER_SETTINGS]
    labels = [LABELS[name.split('/')[-1] if name.startswith('planner/') else name.split('/')[0]] for name in shown]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), layout='constrained')
    for ax, subset in zip(axes, ('rack', 'baffle')):
        values = [tables[subset][name]['deadline_violation']['mean'] * 100 for name in shown]
        errors = [(tables[subset][name]['deadline_violation']['sample_sd_across_training_seeds'] or 0) * 100 for name in shown]
        ax.barh(range(len(shown)), values, xerr=errors, capsize=2, color=['#487b9f'] * 4 + ['#8a9d79'] * 6)
        ax.set_yticks(range(len(shown)), labels, fontsize=9)
        ax.invert_yaxis()
        ax.set(title=subset, xlabel='Test deadline violation (%)')
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('M8 held-out maps; identical sensing, commitments, human navigation and cyclic avoidance')
    fig.savefig(args.output / 'm8-comparison.png', dpi=170)
    fig.savefig(args.output / 'm8-comparison.svg')
    plt.close(fig)
    columns = [f'{family}/{profile}' for family in ('rack', 'baffle') for profile in ('balanced', 'tight', 'pressure')]
    values = np.asarray([[tables[column][name]['deadline_violation']['mean'] * 100 for column in columns] for name in shown])
    fig, ax = plt.subplots(figsize=(11, 6.5), layout='constrained')
    im = ax.imshow(values, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(6), [value.replace('/', '\n') for value in columns])
    ax.set_yticks(range(len(shown)), labels)
    for row in range(len(shown)):
        for col in range(6):
            ax.text(col, row, f'{values[row, col]:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if values[row, col] > values.max() * .6 else '#202020')
    fig.colorbar(im, ax=ax, label='Test violation (%)')
    ax.set_title('M8 training distribution and actor information; new final maps; paired noise pooled')
    fig.savefig(args.output / 'm8-strata.png', dpi=170)
    fig.savefig(args.output / 'm8-strata.svg')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for ax, (effect, before, after) in zip(axes.flat, comparisons):
        for seed in range(3):
            values = [tables['baffle'][f'{variant}/learned']['deadline_violation']['per_training_seed'][seed] * 100 for variant in (before, after)]
            ax.plot([0, 1], values, marker='o', label=f'Seed {seed}')
        ax.set_xticks([0, 1], [LABELS[before].replace(' / ', '\n'), LABELS[after].replace(' / ', '\n')])
        ax.set(title=effect.replace('_', ' '), ylabel='Baffle test violation (%)')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('M8 paired training seeds; lower is better; common full critic and parameter count')
    fig.savefig(args.output / 'm8-paired.png', dpi=170)
    fig.savefig(args.output / 'm8-paired.svg')
    plt.close(fig)
    print('M8_REPORT', json.dumps({name: tables['all'][name]['deadline_violation']['mean'] for name in shown}), flush=True)


if __name__ == '__main__':
    main()
