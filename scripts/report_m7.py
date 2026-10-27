"""M7 descriptive transfer tables, paired deployment switches and figures."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from patrol_rsi.m6 import METRICS
from patrol_rsi.m7 import EXTRA_METRICS, LEARNED_SETTINGS, PLANNER_SETTINGS, contract


LABELS = {'joint_vae': 'Joint + VAE', 'joint_vae_tool_off': 'Joint / tool off',
          'no_vae': 'No-VAE trained', 'no_vae_tool_on': 'No-VAE / tool on',
          'independent_targets': 'Independent targets', 'joint_vae_mask_off': 'Joint / mask off',
          'independent_mask_on': 'Independent / mask on', 'risk_vae': 'Risk-age / VAE',
          'risk_hold': 'Risk-age / hold', 'matching_vae': 'Team matching / VAE',
          'matching_hold': 'Team matching / hold', 'slack': 'Slack rule', 'pair_cycle': 'Pair-cycle rule'}


def describe(values):
    return {'per_checkpoint_seed': values, 'mean': float(np.mean(values)),
            'sample_sd_across_checkpoint_seeds': float(np.std(values, ddof=1)) if len(values) > 1 else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    groups = {}
    jobs = []
    for name in LABELS:
        groups[name] = []
        for seed in range(3) if name in LEARNED_SETTINGS else range(1):
            path = args.results / f'{name}-seed-{seed}/summary.json.gz'
            summary = json.loads(gzip.decompress(path.read_bytes()))
            groups[name].append(summary['groups'][name]['cases'])
            jobs.append({'setting': name, 'seed': seed, 'job_id': summary['job_id'],
                         'evaluation_seconds': summary['seconds'], 'gpu': summary['gpu']})
    subsets = {'all': lambda row: True}
    for family in ('rack', 'baffle'):
        subsets[family] = lambda row, family=family: row['family'] == family
        for profile in ('balanced', 'tight', 'pressure'):
            subsets[f'{family}/{profile}'] = lambda row, family=family, profile=profile: row['family'] == family and row['profile'] == profile
    tables = {}
    for subset, accepts in subsets.items():
        tables[subset] = {}
        for name, runs in groups.items():
            metrics = {metric: describe([float(np.mean([row[metric] for row in rows if accepts(row)]))
                                         for rows in runs]) for metric in METRICS + EXTRA_METRICS}
            cases = [row for rows in runs for row in rows if accepts(row)]
            metrics['worst_case_violation'] = max(row['deadline_violation'] for row in cases)
            metrics['worst_case_peak_age'] = max(row['peak_age'] for row in cases)
            metrics['incomplete_coverage_cases'] = sum(row['target_coverage'] < 1 for row in cases)
            tables[subset][name] = metrics
    effects = {}
    for key, before, after in [
        ('VAE_trained_tool_on_minus_off', 'joint_vae_tool_off', 'joint_vae'),
        ('noVAE_trained_tool_on_minus_off', 'no_vae', 'no_vae_tool_on'),
        ('joint_trained_mask_on_minus_off', 'joint_vae_mask_off', 'joint_vae'),
        ('independent_trained_mask_on_minus_off', 'independent_targets', 'independent_mask_on'),
    ]:
        effects[key] = {subset: {metric: describe([
            float(np.mean([right[metric] - left[metric] for left, right in zip(left_rows, right_rows) if accepts(left)]))
            for left_rows, right_rows in zip(groups[before], groups[after])])
            for metric in METRICS + EXTRA_METRICS} for subset, accepts in subsets.items()}
    witnesses = []
    for index, case in enumerate(contract()['test']):
        methods = [f'{name}/seed-{seed}' for name, runs in groups.items() for seed, rows in enumerate(runs)
                   if rows[index]['target_coverage'] == 1 and rows[index]['deadline_violation'] == 0]
        sample = groups['joint_vae'][0][index]
        witnesses.append(case | {'startup_violation_lower_bound': sample['startup_violation_lower_bound'],
                                  'zero_violation_full_coverage_witnesses': methods,
                                  'recurrent_feasibility_unknown_without_witness': not bool(methods)})
    result = {'tables': tables, 'paired_deployment_effects': effects, 'jobs': jobs,
              'condition_feasibility_witnesses': witnesses, 'training_tasks': 0,
              'learned_deployment_tasks': 21, 'planner_tasks': 6, 'conditions': 96,
              'record_count': 2592, 'new_RSI_experiment': False,
              'statistics': 'Descriptive checkpoint-seed means and sample SD; final cases never used for selection. Planner SD is null, not a training-seed estimate.'}
    (args.output / 'aggregate.json').write_text(json.dumps(result, indent=2))
    names = list(LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5), layout='constrained')
    for ax, metric, title in zip(axes, ['deadline_violation', 'late_deadline_violation'], ['All patrol steps', 'Steps 33-96']):
        values = [100 * tables['all'][name][metric]['mean'] for name in names]
        errors = [100 * (tables['all'][name][metric]['sample_sd_across_checkpoint_seeds'] or 0) for name in names]
        ax.barh(range(len(names)), values, xerr=errors, capsize=2,
                color=['#467c9f' if name in LEARNED_SETTINGS else '#889d76' for name in names])
        ax.set_yticks(range(len(names)), [LABELS[name] for name in names], fontsize=9)
        ax.invert_yaxis()
        ax.set(xlabel='Deadline violation (%)', title=title)
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('M7 fixed M6 checkpoints; 96 new conditions; shared human low-level navigation')
    fig.savefig(args.output / 'm7-comparison.png', dpi=170)
    fig.savefig(args.output / 'm7-comparison.svg')
    plt.close(fig)
    columns = [f'{family}/{profile}' for family in ('rack', 'baffle') for profile in ('balanced', 'tight', 'pressure')]
    values = np.asarray([[100 * tables[column][name]['deadline_violation']['mean'] for column in columns] for name in names])
    fig, ax = plt.subplots(figsize=(11, 7.5), layout='constrained')
    im = ax.imshow(values, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(columns)), [column.replace('/', '\n') for column in columns])
    ax.set_yticks(range(len(names)), [LABELS[name] for name in names])
    for i in range(len(names)):
        for j in range(len(columns)):
            ax.text(j, i, f'{values[i, j]:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if values[i, j] > values.max() * .6 else '#202020')
    fig.colorbar(im, ax=ax, label='Deadline violation (%)')
    ax.set_title('M7 layout family and deadline pressure; checkpoint seeds averaged; paired noise pooled')
    fig.savefig(args.output / 'm7-strata.png', dpi=170)
    fig.savefig(args.output / 'm7-strata.svg')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    pairs = [('joint_vae_tool_off', 'joint_vae', 'VAE-trained: tool off / on'),
             ('no_vae', 'no_vae_tool_on', 'No-VAE-trained: tool off / on'),
             ('joint_vae_mask_off', 'joint_vae', 'Joint-trained: mask off / on'),
             ('independent_targets', 'independent_mask_on', 'Independent-trained: mask off / on')]
    for ax, (off, on, title) in zip(axes.flat, pairs):
        for seed in range(3):
            values = [tables['all'][name]['deadline_violation']['per_checkpoint_seed'][seed] * 100 for name in (off, on)]
            ax.plot([0, 1], values, marker='o', label=f'Checkpoint seed {seed}')
        ax.set_xticks([0, 1], ['Off', 'On'])
        ax.set(title=title, ylabel='Test violation (%)')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('M7 paired deployment switches; frozen weights; not retrained ablations')
    fig.savefig(args.output / 'm7-crossover.png', dpi=170)
    fig.savefig(args.output / 'm7-crossover.svg')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), layout='constrained')
    for ax, name in zip(axes.flat, ('joint_vae', 'independent_targets', 'matching_hold', 'pair_cycle')):
        row = next(row for row in groups[name][0] if row['family'] == 'baffle' and row['layout_seed'] == 31100
                   and row['profile'] == 'pressure' and row['noise'] == 0)
        age = np.asarray(row['target_ages']) / np.asarray(row['target_deadlines'])
        im = ax.imshow(age.T, origin='lower', aspect='auto', vmin=0, vmax=4, cmap='magma')
        ax.set(title=LABELS[name], xlabel='Patrol step', ylabel='Inspection point')
    fig.colorbar(im, ax=axes, label='Age / deadline; display capped at 4')
    fig.suptitle('Fixed display: first baffle test map, pressure, clean, checkpoint seed 0')
    fig.savefig(args.output / 'm7-age-traces.png', dpi=170)
    fig.savefig(args.output / 'm7-age-traces.svg')
    plt.close(fig)
    print('M7 REPORT', json.dumps({name: {'violation': tables['all'][name]['deadline_violation']['mean'],
                                           'coverage': tables['all'][name]['target_coverage']['mean']} for name in names}))


if __name__ == '__main__':
    main()
