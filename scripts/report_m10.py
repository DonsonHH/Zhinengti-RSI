"""Transparent grouping and data-bound static scientific figures for M10."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from patrol_rsi.m10 import load_case


METHODS = ('hold', 'linear', 'nominal_model', 'rknn', 'gp64', 'cvae_current', 'cvae_history')
LABELS = ('Hold', 'Linear', 'Nominal physics', 'rKNN', 'GP (64)', 'CVAE current', 'CVAE history')


def summarise(rows):
    result = {}
    for method in METHODS:
        selected = [r for r in rows if r['method'] == method]
        by_seed = {}
        for seed in sorted({r['training_seed'] for r in selected}, key=lambda x: -1 if x is None else x):
            items = [r for r in selected if r['training_seed'] == seed]
            metrics = {}
            for metric in ('rmse_c', 'weighted_rmse_c', 'unmeasured_rmse_c', 'mae_c',
                           'max_absolute_error_c', 'r2'):
                values = [r['metrics'][metric] for r in items if r['metrics'][metric] is not None]
                metrics[metric] = statistics.mean(values) if values else None
            counts = {key: sum(r['metrics'][key] for r in items) for key in ('tp', 'fp', 'fn', 'hot_count')}
            metrics.update(counts)
            metrics['pooled_precision'] = counts['tp'] / (counts['tp'] + counts['fp']) if counts['tp'] + counts['fp'] else None
            metrics['pooled_recall'] = counts['tp'] / (counts['tp'] + counts['fn']) if counts['tp'] + counts['fn'] else None
            metrics['incremental_CPU_ms_per_frame'] = 1000 * statistics.mean(r['mean_incremental_CPU_seconds'] for r in items)
            by_seed['nonlearned' if seed is None else str(seed)] = dict(cases=len(items), metrics=metrics)
        seed_values = [v['metrics']['rmse_c'] for v in by_seed.values()]
        result[method] = {'per_training_seed': by_seed,
                          'mean_case_RMSE_c': statistics.mean(seed_values),
                          'training_seed_SD_of_mean_RMSE_c': statistics.stdev(seed_values) if len(seed_values) > 1 else None}
    return result


def save(fig, output, name):
    fig.savefig(output / (name + '.png'), dpi=170, bbox_inches='tight')
    fig.savefig(output / (name + '.svg'), bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = json.loads((args.results / 'metrics.json').read_text())
    index = json.loads((args.data / 'index.json').read_text())
    groups = {}
    for condition in ('nominal', 'noise', 'sparse'):
        groups['test/' + condition] = summarise([r for r in rows if r['split'] == 'test' and r['condition'] == condition])
    groups['test/nominal/unseen_combinations'] = summarise(
        [r for r in rows if r['split'] == 'test' and r['condition'] == 'nominal' and '+' in r['scenario']])
    groups['gw/nominal'] = summarise([r for r in rows if r['split'] == 'gw'])
    paired = {}
    for group_name, group in groups.items():
        differences = [group['cvae_history']['per_training_seed'][str(seed)]['metrics']['rmse_c'] -
                       group['cvae_current']['per_training_seed'][str(seed)]['metrics']['rmse_c'] for seed in (0, 1, 2)]
        paired[group_name] = {'history_minus_current_per_training_seed': differences,
                              'mean': statistics.mean(differences), 'sample_SD': statistics.stdev(differences)}
    gw = [r for r in index if r['split'] == 'gw']
    common = {}
    for split in ('test', 'gw'):
        values = [float(load_case(args.data, r)['common_update_seconds'].mean() * 6 * 1000)
                  for r in index if r['split'] == split and r['condition'] == 'nominal']
        common[split] = {'mean_common_update_ms_per_30s_frame': statistics.mean(values),
                         'semantics': 'Six public-state updates; maintains nominal observer too. Not included in incremental method bars.'}
    result = {'aggregation': 'Mean of per-case RMSE, then mean over three training seeds; nonlearned has one result. SD is across training-seed means, not independent test repeats.',
              'groups': groups, 'paired_history_effect': paired,
              'shared_preparation_cost': common,
              'gw_power': {'supply_design_budget_MW': 1000, 'IT_nameplate_MW': 800,
                           'peak_source_input_MW': max(r['peak_source_input_MW'] for r in gw),
                           'minimum_observed_budget_margin_MW': min(1000 - r['peak_source_input_MW'] for r in gw),
                           'per_case': [{'case': r['case'], 'peak_source_input_MW': r['peak_source_input_MW']} for r in gw],
                           'global_power_limiter_implemented': False}}
    (args.output / 'aggregate.json').write_text(json.dumps(result, indent=2))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#8594a8', '#b7a891', '#0b456d', '#4c9c87', '#a386ba', '#dc9949', '#c34e59']
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.7), sharey=True)
    for ax, group_name, title in zip(axes, list(groups)[:4],
                                   ('New seeds', 'Sensor noise', 'Half sensors', 'Unseen combinations')):
        group = groups[group_name]
        values = [group[m]['mean_case_RMSE_c'] for m in METHODS]
        deviations = [group[m]['training_seed_SD_of_mean_RMSE_c'] or 0 for m in METHODS]
        ax.bar(np.arange(7), values, yerr=deviations, color=colors, capsize=2)
        ax.set_xticks(range(7), LABELS, rotation=65, ha='right')
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Mean per-case temperature RMSE (degC)')
    fig.suptitle('M10 synthetic state estimation; error bars: three training-seed means')
    fig.tight_layout()
    save(fig, args.output, 'm10-comparison')
    training = json.loads((args.models / 'training.json').read_text())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for row in training['training_records']:
        color = colors[5 if row['method'] == 'cvae_current' else 6]
        ax.plot([v['update'] for v in row['learning']], [v['validation_rmse_c'] for v in row['learning']],
                color=color, alpha=.65, label=row['method'] + ' seed ' + str(row['seed']))
    ax.set(xlabel='Optimizer updates', ylabel='Validation RMSE (degC)',
           title='Fixed budgets and validation-only snapshot selection')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig, args.output, 'm10-learning')
    selected = next(r for r in index if r['case'] == 'test--nominal--aux_feeder_loss+cdu_fouling--feedback--300')
    data = load_case(args.data, selected)
    with np.load(args.results / 'private-predictions' / (selected['case'] + '.npz'), allow_pickle=False) as pred:
        frame = 29  # Preselected 900s, not chosen by estimator performance.
        maps = [data['temperature_c'][frame], pred['nominal_model'][frame],
                pred['cvae_current--seed-0'][frame], pred['cvae_history--seed-0'][frame]]
    low, high = min(v.min() for v in maps), max(v.max() for v in maps)
    fig, axes = plt.subplots(1, 4, figsize=(13, 5.2), constrained_layout=True)
    for ax, values, title in zip(axes, maps, ('Process truth', 'Nominal model', 'CVAE current (seed 0)', 'CVAE history (seed 0)')):
        image = ax.imshow(values.reshape(16, 8), vmin=low, vmax=high, cmap='inferno', aspect='auto')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Rack within pod')
    axes[0].set_ylabel('Pod index (schematic, not floor plan)')
    fig.colorbar(image, ax=axes, label='Current temperature (degC)', shrink=.8)
    fig.suptitle('Unseen auxiliary-power + CDU combination, 900s; independent seed 300')
    save(fig, args.output, 'm10-temperature-map')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    power = result['gw_power']
    axes[0].bar(['Supply budget', 'IT nameplate', 'Peak observed input'],
                [1000, 800, power['peak_source_input_MW']], color=['#0b456d', '#8594a8', '#dc9949'])
    axes[0].set(ylabel='Power (MW)', title='GW supply envelope: different physical quantities')
    for i, value in enumerate([1000, 800, power['peak_source_input_MW']]):
        axes[0].text(i, value + 12, f'{value:.1f}', ha='center')
    axes[0].set_ylim(0, 1120)
    x = np.arange(7)
    for offset, group_name, label in [(-.18, 'test/nominal', '128 racks'), (.18, 'gw/nominal', '3,200 racks')]:
        group = groups[group_name]
        values = [statistics.mean(v['metrics']['incremental_CPU_ms_per_frame'] for v in group[m]['per_training_seed'].values()) for m in METHODS]
        axes[1].bar(x + offset, values, width=.36, label=label)
    axes[1].set_xticks(x, LABELS, rotation=50, ha='right')
    axes[1].set_yscale('log')
    axes[1].set(ylabel='Incremental CPU inference (ms / frame, log)',
                title='Same CPU and one thread; prepared inputs')
    axes[1].legend()
    fig.tight_layout()
    save(fig, args.output, 'm10-gw-and-runtime')
    print('M10_REPORT_COMPLETE', len(rows), 'method-condition records; 4 static figures')


if __name__ == '__main__':
    main()
