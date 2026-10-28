"""Finite local opportunity counts, actual branch tradeoffs and public controls."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    index = json.loads((args.results / 'case-index.json').read_text())
    branches = json.loads((args.results / 'branch-index.json').read_text())
    processes = json.loads((args.results / 'process-index.json').read_text())
    lookup = {r['case']: r for r in index}
    main_rows = [r for r in index if r['counterfactual_source'] is None]
    groups, opportunity, missed_catalogue = {}, {}, []
    for estimator in ('nominal', 'history'):
        groups[estimator] = {}
        for method in ('tour', 'cue_rule', 'actor'):
            rows = [r for r in main_rows if r['specification']['estimator'] == estimator and r['specification']['controller'] == method]
            n = sum(r['metrics']['anomaly_segments'] for r in rows);m = sum(r['metrics']['missed_segments'] for r in rows)
            weight = sum(r['event_weight_sum'] for r in rows)
            groups[estimator][method] = {'cases': len(rows), 'anomaly_segments': n, 'missed_segments': m,
                'pooled_miss_rate': m / n if n else None,
                'weighted_censored_delay_s': sum(r['weighted_delay_sum'] for r in rows) / weight if weight else None,
                'maximum_exact_gap_ratio': max(r['metrics']['maximum_exact_gap_ratio'] for r in rows),
                'mean_extra_checks': float(np.mean([r['metrics']['selected_detours'] for r in rows])),
                'mean_travel_m': float(np.mean([r['metrics']['travel_m'] for r in rows])),
                'mean_service_s': float(np.mean([r['metrics']['service_s'] for r in rows])),
                'mean_estimator_calls': float(np.mean([r['metrics']['estimator_calls'] for r in rows])),
                'false_measured_confirmations': sum(r['metrics']['false_measured_confirmations'] for r in rows)}
        baselines = [r for r in main_rows if r['specification']['estimator'] == estimator and r['specification']['controller'] == 'tour']
        missed = []
        for baseline in baselines:
            for row in baseline['evaluator_probe']['segments']:
                if not row['baseline_detected']:
                    missed.append(row)
                    missed_catalogue.append({'baseline_case': baseline['case'], 'specification': baseline['specification'], **row})
        selected_branches = [b for b in branches if lookup[b['baseline_case']]['specification']['estimator'] == estimator]
        branch_tradeoffs = []
        for branch in selected_branches:
            old, new = branch['baseline_intervals'], branch['branch_intervals']
            recovered = sum(not a['detected'] and b['detected'] for a, b in zip(old, new))
            lost = sum(a['detected'] and not b['detected'] for a, b in zip(old, new))
            branch_tradeoffs.append({'case': branch['case'], 'baseline_case': branch['baseline_case'],
                'target_segment_index': branch['segment_index'], 'recovered_segments': recovered, 'lost_segments': lost,
                'total_missed_delta': branch['total_missed_delta'], 'travel_delta_m': branch['travel_delta_m'],
                'service_delta_s': branch['service_delta_s']})
        opportunity[estimator] = {'tour_cases': len(baselines), 'missed_segment_records': len(missed),
            'classes': dict(Counter(r['opportunity_class'] for r in missed)),
            'missed_with_any_prespecified_cue_witness': sum(r['any_cue_witness_count'] > 0 for r in missed),
            'missed_with_unconfirmed_cue_witness': sum(r['unconfirmed_cue_witness_count'] > 0 for r in missed),
            'actual_one_detour_branches': len(selected_branches), 'branch_tradeoffs': branch_tradeoffs,
            'branches_with_collateral_miss': sum(r['lost_segments'] > 0 for r in branch_tradeoffs),
            'branches_net_fewer_missed': sum(r['total_missed_delta'] < 0 for r in branch_tradeoffs),
            'scope': 'Separate one-action experiments on original tour states; never sum recoveries as a jointly executable schedule.'}
    result = {'groups': groups, 'opportunity': opportunity, 'main_cases': len(main_rows),
        'evaluator_branches': len(branches), 'all_actual_cases': len(index), 'physical_processes': len(processes),
        'note': 'Known layout, paired estimator/noise references on16physical processes; no globally optimal oracle or impossible-miss conclusion'}
    (args.output / 'aggregate.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'missed-opportunity-catalogue.json').write_text(json.dumps(missed_catalogue, indent=2) + '\n')
    (args.output / 'compact-metrics.json.gz').write_bytes(gzip.compress(json.dumps(index).encode(), mtime=0))
    first = branches[0] if branches else None
    baseline_case = first['baseline_case'] if first else next(r['case'] for r in main_rows if r['specification']['controller'] == 'tour')
    baseline = lookup[baseline_case]
    conditions = {k: v for k, v in baseline['specification'].items() if k != 'controller'}
    controls = {method: next(r['case'] for r in main_rows if r['specification']['controller'] == method and
        {k: v for k, v in r['specification'].items() if k != 'controller'} == conditions) for method in ('tour', 'cue_rule', 'actor')}
    demo = {'selection_rule': 'First branch in saved branch-index order, otherwise first tour case; never selected by score',
        'baseline_case': baseline_case, 'branch_case': first['case'] if first else None,
        'segment_index': first['segment_index'] if first else None, 'control_cases': controls,
        'process_key': baseline['process_key']}
    (args.output / 'demo-selection.json').write_text(json.dumps(demo, indent=2) + '\n')
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    x, labels = np.arange(3), ['Fixed tour', 'Public cue rule', 'Frozen PPO']
    for estimator, offset, color in [('nominal', -.18, '#254e66'), ('history', .18, '#b7704f')]:
        values = [groups[estimator][m] for m in ('tour', 'cue_rule', 'actor')]
        for ax, key, factor, title in [(axes[0], 'pooled_miss_rate', 100, 'Missed anomaly records (%)'),
                (axes[1], 'weighted_censored_delay_s', 1, 'Weighted censored wait (s)'),
                (axes[2], 'mean_extra_checks', 1, 'Optional checks / episode')]:
            ax.bar(x + offset, [v[key] * factor for v in values], width=.35, color=color, label=estimator)
            ax.set_title(title, fontsize=12)
            ax.set_xticks(x, labels, rotation=25, ha='right')
            ax.grid(axis='y', alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(fontsize=9)
    fig.suptitle('C1-A: same observation permissions, staffing and complete-return constraints', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    for ext in ('png', 'svg'):
        fig.savefig(args.output / f'c1a-public-controls.{ext}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    kinds = [('one_detour_with_prespecified_unconfirmed_cue', '#b7704f', 'One-detour + unconfirmed cue'),
        ('one_detour_without_prespecified_unconfirmed_cue', '#63869a', 'One-detour, no such cue'),
        ('no_one_detour_witness_on_original_tour_states', '#b9c4cb', 'No local witness found')]
    bottom = np.zeros(2)
    for kind, color, label in kinds:
        values = np.asarray([opportunity[e]['classes'].get(kind, 0) for e in ('nominal', 'history')])
        axes[0].bar(np.arange(2), values, bottom=bottom, color=color, label=label)
        bottom += values
    axes[0].set(xticks=[0, 1], xticklabels=['Nominal', 'History'], ylabel='Tour-missed segment records', title='Finite witnesses on original tour states')
    axes[0].legend(fontsize=8, loc='upper left')
    items = ['actual_one_detour_branches', 'branches_with_collateral_miss', 'branches_net_fewer_missed']
    for estimator, offset, color in [('nominal', -.18, '#254e66'), ('history', .18, '#b7704f')]:
        axes[1].bar(np.arange(3) + offset, [opportunity[estimator][k] for k in items], width=.35, color=color, label=estimator)
    axes[1].set(xticks=np.arange(3), xticklabels=['Separate branches', 'New collateral miss', 'Net fewer misses'], ylabel='Independent branch experiments', title='Actual replay checks side effects')
    axes[1].tick_params(axis='x', rotation=25);axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(axis='y', alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Evaluator-only diagnosis: neither a deployable policy nor a global bound', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    for ext in ('png', 'svg'):
        fig.savefig(args.output / f'c1a-local-opportunity.{ext}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    if first is None:
        axes[0].text(.5, .5, 'No one-detour branch in this dataset.\nNo general impossibility conclusion follows.',
            transform=axes[0].transAxes, ha='center', va='center')
    else:
        base = json.loads(gzip.decompress((args.results / baseline['path']).read_bytes()))
        branch = json.loads(gzip.decompress((args.results / lookup[first['case']]['path']).read_bytes()))
        interval = base['anomaly_intervals'][first['segment_index']]
        task_id, kind = interval['task_id'], interval['kind']
        target = int(task_id.split(':')[1])
        pack = np.load(args.results / processes[baseline['process_key']]['path'], allow_pickle=False)
        if kind == 'thermal':
            truth, field, threshold, unit = pack['rack_temperature_c'][:, target], 'temperature_c', 65., 'degC'
        elif kind == 'electrical':
            truth, field, threshold, unit = pack['contact_temperature_c'][:, target], 'contact_temperature_c', 60., 'degC'
        elif kind == 'cooling':
            truth, field, threshold, unit = pack['rack_flow_kg_s'].reshape(360, 16, 8).sum(2)[:, target], 'pod_flow_kg_s', 33.6, 'kg/s'
        else:
            truth = abs((pack['public_sensor_temperature_c'] - pack['rack_temperature_c']).reshape(360, 16, 8).mean(2)[:, target])
            field, threshold, unit = 'controller_sensor_error_c', 1., 'abs degC'
        axes[0].plot(np.arange(1, 361) * 5, truth, color='#a7afb5', label='Hidden reference (evaluator only)')
        axes[0].axhline(threshold, color='#b3484b', linestyle='--', label='Synthetic evaluation threshold')
        for record, color, marker, label in [(base, '#254e66', 'o', 'Original tour'), (branch, '#b7704f', 'x', 'One evaluator detour')]:
            checks = [e for e in record['inspection_events'] if e['task_id'] == task_id]
            measurements = [abs(e['measurement'][field]) if kind == 'control' else e['measurement'][field] for e in checks]
            axes[0].scatter([e['time_s'] for e in checks], measurements, color=color, marker=marker, s=40, label=label, zorder=4)
            times = [0.] + [e['time_s'] for e in checks] + [1800.]
            xx, yy = [0.], [0.]
            for a, b in zip(times, times[1:]):
                xx.extend([b, b]);yy.extend([b - a, 0.])
            xx.pop();yy.pop()
            axes[1].plot(xx, yy, color=color, label=label)
        for ax in axes:
            ax.axvspan(interval['onset_s'], interval['end_s'], color='#d8b077', alpha=.2)
            ax.axvline(first['intervention']['dispatch_time_s'], color='#464b4f', linestyle=':')
            ax.grid(alpha=.2);ax.legend(fontsize=8, loc='upper right')
        axes[0].set(ylabel=unit, title=f'First saved branch: {task_id}; shaded interval is evaluator-only')
        axes[1].set(ylabel='Exact completion / tail age (s)', xlabel='Time (s)')
        axes[1].axhline(base['workflow_contract']['deadlines_s'][interval['task']], color='#b3484b', linestyle='--')
        axes[1].set_xlim(max(0, interval['onset_s'] - 150), min(1800, interval['end_s'] + 200))
        pack.close()
    fig.suptitle('One local witness actually executed; other segments still evaluated', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    for ext in ('png', 'svg'):
        fig.savefig(args.output / f'c1a-counterfactual.{ext}', dpi=180)
    plt.close(fig)
    print('C1A_REPORT_COMPLETE', len(main_rows), 'main', len(branches), 'branches', flush=True)


if __name__ == '__main__':
    main()
