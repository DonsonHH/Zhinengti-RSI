"""Audited M11 resource, discovery, revisit and fixed-case figures."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from patrol_rsi.aisles import AisleGraph, ROLES, build_agents
from patrol_rsi.facility import FacilityConfig

METHODS = ('cycle', 'urgency', 'risk', 'rolling', 'ppo_free', 'ppo_guard')
LABELS = ('Fixed tour', 'Urgency', 'Risk greedy', 'Two-step plan', 'PPO free', 'PPO guarded')


def statistics_for(items):
    segments = sum(row['metrics']['anomaly_segments'] for row in items)
    missed = sum(row['metrics']['missed_segments'] for row in items)
    weight = sum(row['event_weight_sum'] for row in items)
    result = {'cases': len(items), 'anomaly_segments': segments, 'missed_segments': missed,
              'pooled_miss_rate': missed / segments if segments else None,
              'weighted_censored_delay_s': sum(row['weighted_delay_sum'] for row in items) / weight if weight else None,
              'mean_overdue_fraction': statistics.mean(row['metrics']['mean_overdue_fraction'] for row in items),
              'mean_final_coverage': statistics.mean(row['metrics']['final_professional_coverage'] for row in items),
              'maximum_age_s_across_cases': max(row['metrics']['maximum_inspection_age_s'] for row in items),
              'mean_episode_maximum_age_s': statistics.mean(row['metrics']['maximum_inspection_age_s'] for row in items),
              'mean_travel_m': statistics.mean(row['metrics']['travel_m'] for row in items),
              'mean_service_s': statistics.mean(row['metrics']['service_s'] for row in items),
              'mean_decision_CPU_s': statistics.mean(row['metrics']['decision_CPU_seconds'] for row in items),
              'mean_estimator_CPU_s': statistics.mean(row['metrics']['history_estimator_inference_s'] for row in items),
              'mean_guard_forced_choices': statistics.mean(row['metrics']['guard_forced_choices'] for row in items)}
    result['by_role'] = {}
    for role in ROLES:
        count = sum(row['by_role'][role]['segments'] for row in items)
        found = sum(row['by_role'][role]['confirmed'] for row in items)
        result['by_role'][role] = {'segments': count, 'confirmed': found,
            'pooled_miss_rate': 1 - found / count if count else None,
            'mean_overdue_fraction': statistics.mean(row['by_role'][role]['mean_overdue_fraction'] for row in items),
            'mean_final_coverage': statistics.mean(row['by_role'][role]['final_coverage'] for row in items)}
    return result


def groups_for(rows):
    groups = {}
    for group in ('main', 'scarce', 'aisle_change', 'gw'):
        groups[group] = {}
        for estimator in ('nominal', 'history'):
            groups[group][estimator] = {}
            for method in METHODS:
                selected = [row for row in rows if row['specification']['group'] == group
                            and row['specification']['estimator'] == estimator
                            and row['specification']['method'] == method]
                if not selected:
                    continue
                seeds = sorted({row['specification']['training_seed'] for row in selected},
                               key=lambda s: -1 if s is None else s)
                per_seed = {}
                for seed in seeds:
                    items = [row for row in selected if row['specification']['training_seed'] == seed]
                    per_seed['rule' if seed is None else str(seed)] = statistics_for(items)
                means, deviations = {}, {}
                keys = [k for k in next(iter(per_seed.values())) if k not in ('cases', 'by_role')]
                for key in keys:
                    values = [v[key] for v in per_seed.values() if v[key] is not None]
                    means[key] = statistics.mean(values) if values else None
                    deviations[key] = statistics.stdev(values) if len(values) > 1 else None
                means['by_role'] = {}
                for role in ROLES:
                    role_values = [v['by_role'][role] for v in per_seed.values()]
                    means['by_role'][role] = {}
                    for key in role_values[0]:
                        values = [r[key] for r in role_values if r[key] is not None]
                        means['by_role'][role][key] = statistics.mean(values) if values else None
                groups[group][estimator][method] = {'per_training_seed': per_seed, 'mean': means,
                                                   'training_seed_SD': deviations}
    return groups


def save(fig, output, name):
    fig.savefig(output / (name + '.png'), dpi=170, bbox_inches='tight')
    fig.savefig(output / (name + '.svg'), bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--capacity', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    index = json.loads((args.results / 'case-index.json').read_text())
    compact = []
    for item in index:
        record = json.loads(gzip.decompress((args.results / item['record']).read_bytes()))
        intervals = record['anomaly_intervals']
        compact.append(dict(item, event_weight_sum=sum(row['weight'] for row in intervals),
                            weighted_delay_sum=sum(row['weight'] * row['delay_or_censor_s'] for row in intervals)))
    groups = groups_for(compact)
    pairs = {}
    for group, modes in groups.items():
        pairs[group] = {}
        for mode, methods in modes.items():
            guarded, free = methods['ppo_guard']['per_training_seed'], methods['ppo_free']['per_training_seed']
            pairs[group][mode] = {}
            for metric in ('pooled_miss_rate', 'weighted_censored_delay_s', 'mean_overdue_fraction',
                           'mean_episode_maximum_age_s', 'mean_final_coverage'):
                differences = [guarded[seed][metric] - free[seed][metric] for seed in guarded
                               if guarded[seed][metric] is not None and free[seed][metric] is not None]
                pairs[group][mode][metric] = {'per_training_seed': differences,
                    'mean': statistics.mean(differences) if differences else None,
                    'sample_SD': statistics.stdev(differences) if len(differences) > 1 else None}
    result = {'aggregation': 'Pooled anomaly segments/impact weights per policy training seed; averages and SD across the three trained seeds. GW uses only predeclared seed0.',
              'groups': groups, 'paired_guard_minus_free': pairs,
              'scope': 'Synthetic zoned qualified task scheduling. Age guard is an inference intervention on the same policy; not a guaranteed deadline theorem.'}
    (args.output / 'aggregate.json').write_text(json.dumps(result, indent=2))
    (args.output / 'compact-case-metrics.json.gz').write_bytes(gzip.compress(json.dumps(compact).encode(), mtime=0))
    capacity = json.loads((args.capacity / 'feasibility.json').read_text())
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)
    y = np.arange(len(capacity))
    for ax, role in zip(axes, ROLES):
        routes = [max(c['closed_quantized_cycle_s'] for c in r['roles'][role]['constructed_cycles']) /
                  r['roles'][role]['reference_revisit_s'] for r in capacity]
        service = [r['roles'][role]['service_only_utilization'] for r in capacity]
        ax.barh(y, routes, color=['#a6b1c1' if r['level'] == 'scarce' else '#206b8d' for r in capacity])
        ax.scatter(service, y, color='#dc8b37', label='Service-only bound', zorder=3)
        ax.axvline(1, color='#ac3548', ls='--', label='Revisit reference')
        ax.set_title(role.capitalize())
        ax.set_xlabel('Cycle / reference time')
        ax.set_yticks(y, [f"{r['racks']} racks\n{r['variant']} / {r['level']}" for r in capacity])
        ax.grid(axis='x', alpha=.2)
    axes[0].invert_yaxis()
    axes[-1].legend(loc='lower right', fontsize=8)
    fig.suptitle('Constructed aisle tours: a service-only bound is insufficient')
    fig.tight_layout()
    save(fig, args.output, 'm11-feasibility')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex='col')
    x = np.arange(6)
    for column, group in enumerate(('main', 'scarce')):
        for mode, offset, color in (('nominal', -.18, '#1d6082'), ('history', .18, '#c16550')):
            for row, metric, scale in ((0, 'weighted_censored_delay_s', 1), (1, 'pooled_miss_rate', 100)):
                values = [groups[group][mode][method]['mean'][metric] * scale for method in METHODS]
                sd = [(groups[group][mode][method]['training_seed_SD'][metric] or 0) * scale for method in METHODS]
                axes[row, column].bar(x + offset, values, width=.36, yerr=sd, capsize=2, color=color, label=mode)
                axes[row, column].grid(axis='y', alpha=.2)
        axes[0, column].set_title('Provisioned resources' if group == 'main' else 'Scarce resources')
        axes[1, column].set_xticks(x, LABELS, rotation=30, ha='right')
    axes[0, 0].set_ylabel('Impact-weighted censored delay (s)')
    axes[1, 0].set_ylabel('Missed anomaly segments (%)')
    axes[0, 1].legend()
    fig.suptitle('Qualified confirmation: missed segments remain in the delay metric')
    fig.tight_layout()
    save(fig, args.output, 'm11-discovery')
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.7), sharey=True)
    for ax, role in zip(axes, ROLES):
        for group, marker, color in (('main', 'o', '#216580'), ('scarce', 's', '#ba694e')):
            values = [groups[group]['history'][method]['mean']['by_role'][role]['mean_overdue_fraction'] * 100 for method in METHODS]
            ax.plot(x, values, marker=marker, color=color, label=group)
        ax.set_title(role.capitalize())
        ax.set_xticks(x, LABELS, rotation=55, ha='right')
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Professional revisit overdue fraction (%)')
    axes[-1].legend()
    fig.suptitle('History-estimator configuration; professional checks remain separate')
    fig.tight_layout()
    save(fig, args.output, 'm11-revisit')
    chosen = [r for r in index if r['specification']['group'] == 'main' and
              r['specification']['scenario'] == 'coldplate_fouling' and
              r['specification']['control'] == 'fixed_commands' and
              r['specification']['seed'] == 3001 and r['specification']['estimator'] == 'history' and
              r['specification']['method'] in ('cycle', 'risk', 'ppo_free', 'ppo_guard') and
              r['specification']['training_seed'] in (None, 0)]
    chosen.sort(key=lambda r: ('cycle', 'risk', 'ppo_free', 'ppo_guard').index(r['specification']['method']))
    (args.output / 'demo-selection.json').write_text(json.dumps({'case_names': [r['case'] for r in chosen],
        'selection': 'Predeclared coldplate/fixed_commands/seed3001/history/seed0; all four methods shown'}, indent=2))
    graph = AisleGraph(FacilityConfig())
    agents = build_agents(graph, 'provisioned')
    target_rack = (3001 % 16) * 8 + (3001 * 3) % 8
    target_task = next(j for j, task in enumerate(graph.tasks) if task.kind == 'thermal' and task.index == target_rack)
    owner = next(a['id'] for a in agents if target_task in a['tour'])
    process_index = json.loads((args.results / 'process-index.json').read_text())
    process = process_index[chosen[0]['process_key']]
    with np.load(args.results / process['path'], allow_pickle=False) as pack:
        true_temperature = pack['rack_temperature_c'][:, target_rack]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    for a, neighbors in graph.adjacency.items():
        for b in neighbors:
            if a < b:
                ax.plot([a[0], b[0]], [a[1], b[1]], color='#c9d1db', lw=.5)
    for pod in range(16):
        xx, yy = pod % 4 * 12, pod // 4 * 12
        ax.add_patch(Rectangle((xx - .3, yy - .3), 4.2, 2.1, facecolor='#e3e7ec', edgecolor='#8c96a3'))
    records = [json.loads(gzip.decompress((args.results / row['record']).read_bytes())) for row in chosen]
    first_route = [r for r in records[0]['assignments'] if r['agent'] == owner][:8]
    for assignment in first_route:
        route = np.asarray(assignment['route'])
        ax.plot(route[:, 0], route[:, 1], color='#267192', lw=1.3)
    point = graph.tasks[target_task].position_m
    ax.scatter(*point, s=90, color='#b73543', marker='*', zorder=5)
    ax.set_title(f'Synthetic aisles; fixed-tour executor {owner}, first 8 assignments')
    ax.set_aspect('equal')
    ax.set(xlabel='m', ylabel='m')
    time_axis = np.arange(1, 361) * 5
    axes[0, 1].plot(time_axis, true_temperature, color='#24384e', label='Process temperature')
    axes[0, 1].axhline(65, color='#aa5547', ls='--', label='Synthetic threshold')
    colors = ['#216580', '#c39745', '#8b769f', '#c05755']
    for record, color in zip(records, colors):
        method = record['specification']['method']
        events = [e for e in record['inspection_events'] if e['task'] == target_task]
        axes[0, 1].scatter([e['time_s'] for e in events], [e['measurement']['temperature_c'] for e in events],
                           s=15, color=color, label=method)
        ages, last, cursor = [], 0., 0
        for t in time_axis:
            while cursor < len(events) and events[cursor]['time_s'] <= t:
                last = events[cursor]['time_s']
                cursor += 1
            ages.append(t - last)
        axes[1, 0].plot(time_axis, ages, color=color, label=method)
    axes[0, 1].set(xlabel='Time (s)', ylabel='Target temperature (degC)', title=f'Fixed independent case, rack {target_rack}')
    axes[0, 1].legend(fontsize=8, ncol=2)
    axes[1, 0].axhline(300, color='#a53642', ls='--')
    axes[1, 0].set(xlabel='Time (s)', ylabel='Time since qualified thermal check (s)', title='Task age counts professional completion only')
    axes[1, 0].legend(fontsize=8, ncol=2)
    for j, (record, color) in enumerate(zip(records, colors)):
        intervals = [r for r in record['anomaly_intervals'] if r['task'] == target_task]
        for interval in intervals:
            axes[1, 1].plot([interval['onset_s'], interval['end_s']], [j, j], color='#b8bec7', lw=10)
            if interval['first_confirmation_s'] is not None:
                axes[1, 1].scatter(interval['first_confirmation_s'], j, s=35, color=color, zorder=3)
        axes[1, 1].plot([], [], 'o', color=color, label=record['specification']['method'])
    axes[1, 1].set_yticks(range(4), [r['specification']['method'] for r in records])
    axes[1, 1].set(xlabel='Time (s)', title='Gray: true anomaly segment; dot: qualified confirmation')
    axes[1, 1].set_xlim(0, 1800)
    fig.tight_layout()
    save(fig, args.output, 'm11-fixed-case')
    print('M11_REPORT_COMPLETE', len(index), 'episodes; 4 static figures', flush=True)


if __name__ == '__main__':
    main()
