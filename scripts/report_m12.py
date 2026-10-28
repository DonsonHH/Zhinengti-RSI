"""Paired M12 pilot summaries, attribution and predeclared case figures."""
import argparse
from collections import Counter
import gzip
import hashlib
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
    lookup = {r['case']: r for r in index}
    groups = json.loads((args.results / 'test-groups.json').read_text())
    behavior = {}
    demo = {}
    for row in index:
        if row['specification']['phase'] != 'test':
            continue
        record = json.loads(gzip.decompress((args.results / row['path']).read_bytes()))
        counts = Counter()
        for decision in record['workflow_decisions']:
            if decision['legal_candidates'] == 1:
                kind = 'forced'
            elif decision['rule_index'] is not None:
                kind = 'rule'
            elif decision['tool'] == 'ppo':
                kind = 'ppo'
            else:
                kind = 'other'
            counts[kind] += 1
        behavior[row['case']] = dict(counts)
        spec = row['specification']
        if spec['seed'] == 6301 and spec['scenario'] == 'coldplate_fouling' and spec['duration_s'] == 180:
            demo[row['case']] = record

    def summarize(keys):
        rows = [lookup[k] for k in keys]
        segments = sum(r['metrics']['anomaly_segments'] for r in rows)
        missed = sum(r['metrics']['missed_segments'] for r in rows)
        weight = sum(r['event_weight_sum'] for r in rows)
        counts = sum((Counter(behavior[k]) for k in keys), Counter())
        dispatches = sum(counts.values())
        return {'case_references': len(keys), 'unique_case_records': len(set(keys)),
            'anomaly_segments': segments, 'missed_segments': missed,
            'pooled_miss_rate': missed / segments if segments else None,
            'weighted_censored_delay_s': sum(r['weighted_delay_sum'] for r in rows) / weight if weight else None,
            'maximum_exact_gap_ratio': max(r['metrics']['maximum_exact_gap_ratio'] for r in rows),
            'exact_violating_tasks': sum(r['metrics']['tasks_violating_exact_gap'] for r in rows),
            'mean_extra_checks': float(np.mean([r['metrics']['selected_detours'] for r in rows])),
            'mean_estimator_calls': float(np.mean([r['metrics']['estimator_calls'] for r in rows])),
            'mean_travel_m': float(np.mean([r['metrics']['travel_m'] for r in rows])),
            'mean_service_s': float(np.mean([r['metrics']['service_s'] for r in rows])),
            'false_measurement_confirmations': sum(r['metrics']['false_measured_confirmations'] for r in rows),
            'mean_first_alert_task_weight': float(np.mean([r['public_selection']['first_alert_task_weight'] for r in rows])),
            'behavior_counts': dict(counts),
            'behavior_fractions': {k: counts[k] / dispatches for k in ('forced', 'rule', 'ppo', 'other')},
            'different_from_actor_fraction': sum(r['public_selection']['different_from_actor'] for r in rows) / dispatches}

    detail = {name: summarize(keys) for name, keys in groups.items()}
    names = ['cycle', 'fixed_actor', 'human_risk_rules', 'random_search', 'llm_no_memory', 'llm_memory']
    main_groups = {}
    for name in names:
        keys = groups[name] if name in groups else [k for seed in (6101, 6102, 6103) for k in groups[f'{name}/{seed}']]
        main_groups[name] = summarize(keys)
    proposals = [json.loads(p.read_text()) for p in sorted((args.results / 'proposals').glob('*.json'))]
    cost = {}
    for method in ('random_search', 'llm_no_memory', 'llm_memory'):
        rows = [r for r in proposals if r['method'] == method]
        generated = [r['generation'] for r in rows if 'generation' in r]
        cost[method] = {'candidate_slots': len(rows), 'valid': sum(r['valid'] for r in rows),
            'accepted': sum(r['accepted'] for r in rows),
            'duplicates': sum(r['reason'] == 'duplicate_consumes_slot' for r in rows),
            'model_calls': len(generated), **{k: sum(g[k] for g in generated) for k in ('input_tokens', 'output_tokens', 'generation_seconds')}}
    result = {'main_groups': main_groups, 'all_groups': detail, 'proposal_costs': cost,
        'test_processes': 16, 'test_group_references': sum(len(v) for v in groups.values()),
        'test_unique_case_records': len(behavior),
        'note': 'Generation repetitions reuse16physical conditions; no independent plant or new neural-training interpretation',
        'selection_and_hidden_outcomes_are_distinct': True}
    (args.output / 'aggregate.json').write_text(json.dumps(result, indent=2) + '\n')
    compact = [r | {'behavior_counts': behavior.get(r['case'])} for r in index]
    (args.output / 'compact-metrics.json.gz').write_bytes(gzip.compress(json.dumps(compact).encode(), mtime=0))

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.titlesize': 12})
    labels = ['Fixed tour', 'Frozen PPO', 'Human rules', 'Random search', 'LLM no memory', 'LLM memory']
    colors = ['#7a8793', '#17445e', '#768f97', '#c29b3b', '#648e9e', '#9c5b49']
    x = np.arange(6)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), layout='constrained')
    for ax, key, title in zip(axes, ('pooled_miss_rate', 'weighted_censored_delay_s', 'mean_extra_checks'),
            ('Missed anomaly segments (%)', 'Weighted censored delay (s)', 'Extra checks per episode')):
        values = [main_groups[n][key] for n in names]
        if key == 'pooled_miss_rate':
            values = np.asarray(values) * 100
        ax.bar(x, values, color=colors)
        ax.set_xticks(x, labels, rotation=32, ha='right')
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('M12-A pilot: task outcomes after frozen public-evidence selection', fontsize=14)
    for ext in ('png', 'svg'):
        fig.savefig(args.output / f'm12-outcomes.{ext}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    bottom = np.zeros(6)
    for kind, color, label in [('forced', '#d2d9dd', 'Only one legal candidate'),
            ('ppo', '#17445e', 'PPO fallback with choice'), ('rule', '#b77751', 'Rule tool with choice'),
            ('other', '#798a6e', 'Tour / heuristic choice')]:
        values = np.asarray([main_groups[n]['behavior_fractions'][kind] * 100 for n in names])
        axes[0].bar(x, values, bottom=bottom, color=color, label=label)
        bottom += values
    axes[0].set_xticks(x, labels, rotation=32, ha='right')
    axes[0].set_ylabel('Actual dispatches (%)')
    axes[0].set_title('Who actually selected the task?')
    axes[0].legend(fontsize=8, loc='upper left', bbox_to_anchor=(0, 1.29), ncol=2)
    axes[1].bar(x, [main_groups[n]['mean_estimator_calls'] for n in names], color=colors)
    axes[1].set_xticks(x, labels, rotation=32, ha='right')
    axes[1].set_ylabel('CVAE calls / episode')
    axes[1].set_title('Tool use is executed and counted')
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    for ext in ('png', 'svg'):
        fig.savefig(args.output / f'm12-attribution.{ext}', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    selected_demo = {}
    for group, color in zip(('fixed_actor', 'human_risk_rules', 'random_search/6101', 'llm_memory/6101'),
            ('#17445e', '#768f97', '#c29b3b', '#9c5b49')):
        key = next(k for k in groups[group] if k in demo)
        record = demo[key]
        gap = next(g for g in record['exact_task_gaps'] if g['task_id'] == 'thermal:111')
        times = [0.] + gap['completion_times_s'] + [1800.]
        xx, yy = [0.], [0.]
        for t0, t1 in zip(times, times[1:]):
            xx.extend([t1, t1])
            yy.extend([t1 - t0, 0.])
        # The terminal point is an episode tail, not an actual completion.
        xx.pop(); yy.pop()
        ax.plot(xx, yy, label=group, color=color, alpha=.85)
        selected_demo[group] = {'case': key, 'version_id': record['version_id'], 'workflow': record['workflow']}
    ax.axhline(300, color='#ae3949', linestyle='--', label='Revisit requirement')
    ax.set(xlabel='Time (s)', ylabel='Exact check age at completion / tail (s)',
        title='Predeclared coldplate180 / seed6301 / thermal rack111')
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)
    for ext in ('png', 'svg'):
        fig.savefig(args.output / f'm12-case.{ext}', dpi=180)
    plt.close(fig)
    (args.output / 'demo-selection.json').write_text(json.dumps(selected_demo, indent=2) + '\n')
    print('M12_REPORT_COMPLETE', json.dumps({'test_unique_cases': len(behavior), 'references': result['test_group_references']}))


if __name__ == '__main__':
    main()
