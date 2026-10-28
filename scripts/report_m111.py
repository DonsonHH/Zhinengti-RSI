"""Startup prototype figures; no training-seed performance inference."""
import argparse
import gzip
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from patrol_rsi.aisles import build_agents
from patrol_rsi.active_inspection import graph_for

METHODS = ('cycle', 'risk_free', 'risk_certified', 'untrained_certified')
LABELS = ('Owned tour', 'Risk free', 'Risk + return plan', 'Untrained + return plan')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    index = json.loads((args.results / 'case-index.json').read_text())
    records = {r['case']: json.loads(gzip.decompress((args.results / r['path']).read_bytes())) for r in index}
    groups, compact = {}, []
    for item in index:
        record = records[item['case']]
        intervals = record['anomaly_intervals']
        compact.append(dict(item, event_weight_sum=sum(r['weight'] for r in intervals),
                            weighted_delay_sum=sum(r['weight'] * r['delay_or_censor_s'] for r in intervals)))
    for estimator in ('nominal', 'history'):
        groups[estimator] = {}
        for method in METHODS:
            items = [r for r in compact if r['specification']['estimator'] == estimator and r['specification']['selector'] == method]
            segments = sum(r['metrics']['anomaly_segments'] for r in items)
            missed = sum(r['metrics']['missed_segments'] for r in items)
            weights = sum(r['event_weight_sum'] for r in items)
            groups[estimator][method] = {'cases': len(items), 'segments': segments, 'missed_segments': missed,
                'pooled_miss_rate': missed / segments if segments else None,
                'weighted_censored_delay_s': sum(r['weighted_delay_sum'] for r in items) / weights if weights else None,
                'maximum_exact_gap_ratio': max(r['metrics']['maximum_exact_gap_ratio'] for r in items),
                'cases_with_exact_gap_violation': sum(r['metrics']['tasks_violating_exact_gap'] > 0 for r in items),
                'mean_coverage': statistics.mean(r['metrics']['final_professional_coverage'] for r in items),
                'mean_overdue_fraction': statistics.mean(r['metrics']['mean_overdue_fraction'] for r in items),
                'mean_extra_inspections': statistics.mean(r['metrics']['selected_detours'] for r in items)}
    service_by_process = {}
    for item in index:
        value = records[item['case']]['service_evidence_events']
        if item['process_key'] in service_by_process and value != service_by_process[item['process_key']]:
            raise RuntimeError('Independent remote service evidence differs across selectors')
        service_by_process[item['process_key']] = value
    result = {'scope': 'Startup prototype, no new policy training; pooled16 paired process fixtures per estimator/method; not a policy-training-seed significance study',
              'groups': groups,
              'remote_service_evidence_identical_across_all_selectors': True,
              'remote_service_by_process': {key: {'events': events, 'root_cause_diagnosed': False} for key, events in service_by_process.items()}}
    (args.output / 'aggregate.json').write_text(json.dumps(result, indent=2))
    (args.output / 'compact-metrics.json.gz').write_bytes(gzip.compress(json.dumps(compact).encode(), mtime=0))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'nominal': '#26647c', 'history': '#bd6a53'}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    x = np.arange(4)
    for estimator, shift in (('nominal', -.18), ('history', .18)):
        data = [groups[estimator][m] for m in METHODS]
        for ax, metric, scale in zip(axes,
            ('maximum_exact_gap_ratio', 'pooled_miss_rate', 'weighted_censored_delay_s'), (1, 100, 1)):
            ax.bar(x + shift, [r[metric] * scale for r in data], .36, color=colors[estimator], label=estimator)
            ax.set_xticks(x, LABELS, rotation=35, ha='right')
            ax.grid(axis='y', alpha=.2)
    axes[0].axhline(1, color='#ac3349', ls='--')
    axes[0].set(ylabel='Worst exact completion gap / reference',
                title='Initial and terminal gaps included')
    axes[1].set(ylabel='Missed anomaly segments (%)', title='Same known task ownership')
    axes[2].set(ylabel='Impact-weighted censored delay (s)', title='Missing events retained')
    axes[2].legend()
    fig.suptitle('M11.1 startup pilot: coverage preservation and task outcomes')
    fig.tight_layout()
    save(fig, args.output, 'm111-pilot')
    selected = [r for r in index if r['specification']['scenario'] == 'coldplate_fouling'
        and r['specification']['duration_s'] == 180 and r['specification']['seed'] == 5101
        and r['specification']['estimator'] == 'history']
    selected.sort(key=lambda r: METHODS.index(r['specification']['selector']))
    (args.output / 'demo-selection.json').write_text(json.dumps({'cases': [r['case'] for r in selected],
        'rule': 'Predeclared coldplate180/seed5101/history; all four selectors shown'}, indent=2))
    graph = graph_for(128, 'open')
    rack = (5101 % 16) * 8 + (5101 * 3) % 8
    owner = next(a['id'] for a in build_agents(graph, 'provisioned') if rack in a['tour'])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    palette = ['#286581', '#bf9a45', '#be5c55', '#8874a3']
    for row, color in zip(selected, palette):
        record = records[row['case']]
        values = next(r for r in record['exact_task_gaps'] if r['task'] == rack)
        stamps = values['completion_times_s']
        times, ages, last = [0.], [0.], 0.
        for stamp in stamps:
            times.extend([stamp, stamp])
            ages.extend([stamp - last, 0.])
            last = stamp
        times.append(1800.)
        ages.append(1800 - last)
        label = row['specification']['selector']
        axes[0].plot(times, ages, color=color, label=label)
        if label != 'risk_free':
            certificates = [r for r in record['selected_certificates'] if r['agent'] == owner]
            axes[1].scatter([r['clock_s'] for r in certificates],
                [r['certificate']['minimum_projected_slack_s'] for r in certificates],
                s=9, color=color, alpha=.7, label=label)
    axes[0].axhline(300, color='#a93445', ls='--')
    axes[0].set(xlabel='Time (s)', ylabel='Exact age at completion / episode tail (s)',
                title=f'Predeclared case, representative rack {rack}')
    axes[1].axhline(0, color='#a93445', ls='--')
    axes[1].set(xlabel='Dispatch time (s)', ylabel='Minimum projected full-round slack (s)',
                title=f'Selected plans of fixed owner {owner}')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig, args.output, 'm111-return-plans')
    service_case = next(r for r in index if r['specification']['scenario'] == 'it_feeder_loss' and
        r['specification']['duration_s'] == 180 and r['specification']['seed'] == 5101 and
        r['specification']['estimator'] == 'nominal' and r['specification']['selector'] == 'cycle')
    process_index = json.loads((args.results / 'process-index.json').read_text())
    with np.load(args.results / process_index[service_case['process_key']]['path'], allow_pickle=False) as pack:
        scalar = pack['process_scalars']
    events = [e for e in records[service_case['case']]['service_evidence_events'] if e['pod'] == 5101 % 16]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(scalar[:, 0], scalar[:, 1], color='#286581')
    axes[0].set(ylabel='Unserved IT demand (kW)', title='Observed service deficit; root cause remains undiagnosed')
    markers = {'new_measured_alert': (0, '#b64c4c'), 'persistent_recheck': (1, '#c19743'),
               'recovery_confirmation': (2, '#3b897c')}
    for kind, (y, color) in markers.items():
        axes[1].scatter([e['time_s'] for e in events if e['kind'] == kind],
                        [y] * sum(e['kind'] == kind for e in events), color=color, label=kind)
    axes[1].set_yticks(range(3), ['New observed alert', 'Persistent recheck', 'Recovery confirmation'])
    axes[1].set(xlabel='Time (s)', xlim=(0, 1800), ylim=(-.5, 2.5))
    axes[1].legend(fontsize=8, loc='upper right')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig, args.output, 'm111-service-evidence')
    print('M111_REPORT_COMPLETE', len(index), 'cases; three static figures')


def save(fig, folder, name):
    fig.savefig(folder / (name + '.png'), dpi=170, bbox_inches='tight')
    fig.savefig(folder / (name + '.svg'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
