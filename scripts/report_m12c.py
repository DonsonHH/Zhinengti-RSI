"""Evidence representation, search budget and deployment attribution results."""
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
    trials, costs, compact, signatures, process_keys, demos = {}, {}, [], set(), set(), []
    for t in range(3):
        folder = args.results / f'trial-{t}'
        index = json.loads((folder / 'case-index.json').read_text())
        lookup = {r['case']: r for r in index}
        groups = json.loads((folder / 'test-groups.json').read_text())
        compact.extend(r | {'trial': t} for r in index)
        signatures.update(lookup)
        process_keys.update(r['process_key'] for r in index)
        results = {}
        for name, keys in groups.items():
            rows = [lookup[k] for k in keys]
            count = sum(r['metrics']['anomaly_segments'] for r in rows)
            missed = sum(r['metrics']['missed_segments'] for r in rows)
            weight = sum(r['event_weight_sum'] for r in rows)
            dispatches = sum(r['public_selection']['dispatches'] for r in rows)
            results[name] = {'observation_conditions': len(keys), 'anomaly_segments': count, 'missed_segments': missed,
                'miss_rate': missed / count if count else None,
                'weighted_censored_delay_s': sum(r['weighted_delay_sum'] for r in rows) / weight if weight else None,
                'exact_violating_tasks': sum(r['metrics']['tasks_violating_exact_gap'] for r in rows),
                'maximum_exact_gap_ratio': max(r['metrics']['maximum_exact_gap_ratio'] for r in rows),
                'mean_extra_checks': float(np.mean([r['metrics']['selected_detours'] for r in rows])),
                'mean_estimator_calls': float(np.mean([r['metrics']['estimator_calls'] for r in rows])),
                'mean_travel_m': float(np.mean([r['metrics']['travel_m'] for r in rows])),
                'mean_service_s': float(np.mean([r['metrics']['service_s'] for r in rows])),
                'false_confirmations': sum(r['metrics']['false_measured_confirmations'] for r in rows),
                'ppo_tool_fraction': sum(r['public_selection']['ppo_dispatches'] for r in rows) / dispatches,
                'rule_tool_fraction': sum(r['public_selection']['rule_dispatches'] for r in rows) / dispatches,
                'single_candidate_fraction': sum(r['public_selection']['forced_single_candidate'] for r in rows) / dispatches,
                'different_from_actor_fraction': sum(r['public_selection']['different_from_actor'] for r in rows) / dispatches}
            if t == 0 and name in ('tour', 'summary3', 'events3', 'redacted3'):
                example = next(r for r in rows if r['specification']['scenario'] == 'coldplate_fouling'
                    and r['specification']['duration_s'] == 180 and r['specification']['seed'] == 7401 and r['specification']['noise_c'] == 1.)
                demos.append({'trial': t, 'group': name, 'case': example['case'], 'path': example['path'],
                    'workflow': example['workflow'], 'version_id': example['version_id']})
        trials[str(t)] = results
        proposals = [json.loads(p.read_text()) for p in (folder / 'proposals').glob('*.json')]
        costs[str(t)] = {}
        for method in ('summary', 'events', 'redacted', 'random'):
            entries = [r for r in proposals if r['method'] == method]
            generations = [r['generation'] for r in entries if 'generation' in r]
            costs[str(t)][method] = {'candidate_slots': len(entries), 'model_calls': len(generations),
                'accepted': sum(r['accepted'] for r in entries), 'duplicates': sum(r['reason'] == 'duplicate' for r in entries),
                'invalid': sum(not r['valid'] for r in entries),
                **{k: sum(r[k] for r in generations) for k in ('input_tokens', 'output_tokens', 'generation_seconds')}}
    main_groups = {}
    for name in trials['0']:
        values = [trials[str(t)][name] for t in range(3)]
        main_groups[name] = {'per_generation_trial': values,
            'mean_miss_rate': float(np.mean([r['miss_rate'] for r in values])),
            'mean_weighted_censored_delay_s': float(np.mean([r['weighted_censored_delay_s'] for r in values])),
            'mean_extra_checks': float(np.mean([r['mean_extra_checks'] for r in values]))}
    result = {'trials': trials, 'main_groups': main_groups, 'cost_by_trial': costs,
        'actual_closed_loop_records': len(compact), 'distinct_case_signatures': len(signatures),
        'distinct_physical_processes': len(process_keys), 'test_physical_processes': 16,
        'test_observation_conditions': 32, 'test_group_case_references': 1056,
        'note': 'Three generation repetitions share physical/noise conditions; no independent-plant error bars or new neural-training claims'}
    (args.output / 'aggregate.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'compact-metrics.json.gz').write_bytes(gzip.compress(json.dumps(compact).encode(), mtime=0))
    (args.output / 'demo-selection.json').write_text(json.dumps(demos, indent=2) + '\n')
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    names = ['tour', 'actor', 'human', 'random9', 'summary3', 'summary9', 'redacted3', 'events3']
    labels = ['Tour', 'Frozen PPO', 'Human rules', 'Random9', 'Summary3', 'Summary9', 'Redacted3', 'Events3']
    colors = ['#7c8992', '#234c64', '#7c948b', '#c19b45', '#4e7d97', '#3e6479', '#a68d83', '#b35b3e']
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, key, field, title, scale in [(axes[0], 'mean_miss_rate', 'miss_rate', 'Missed anomaly segments (%)', 100),
            (axes[1], 'mean_weighted_censored_delay_s', 'weighted_censored_delay_s', 'Weighted censored delay (s)', 1)]:
        ax.bar(x, [main_groups[n][key] * scale for n in names], color=colors)
        for t, offset in enumerate((-.15, 0, .15)):
            ax.scatter(x + offset, [trials[str(t)][n][field] * scale for n in names], color='#222222', s=15, zorder=4)
        ax.set_xticks(x, labels, rotation=38, ha='right')
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('M12-C0: summary versus traceable public events on shared test conditions', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .94))
    for ext in ('png', 'svg'):
        fig.savefig(args.output / f'm12c-information.{ext}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    methods, method_labels = ['summary', 'events', 'redacted', 'random'], ['Summary9', 'Events3', 'Redacted3', 'Random9']
    for ax, field, title in [(axes[0], 'candidate_slots', 'Evaluated proposal opportunities'),
            (axes[1], 'input_tokens', 'Actual model input tokens'), (axes[2], 'duplicates', 'Duplicate proposals consumed')]:
        values = [float(np.mean([costs[str(t)][m][field] for t in range(3)])) for m in methods]
        ax.bar(np.arange(4), values, color=['#3e6479', '#b35b3e', '#a68d83', '#c19b45'])
        ax.set_xticks(np.arange(4), method_labels, rotation=35, ha='right')
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(args.output / f'm12c-proposal-cost.{ext}', dpi=180)
    plt.close(fig)

    entry = json.loads((args.results / 'trial-0/proposals/events-0.json').read_text())
    evidence = entry['generation']['prompt']['event_evidence']
    (args.output / 'predeclared-event-input.json').write_text(json.dumps(evidence, indent=2) + '\n')
    cells = []
    for row in evidence['events']:
        c = row['card']
        previous = '-' if c['previous_value'] is None else f"{c['previous_value']:.1f}"
        old_time = 'start' if c['previous_completed_at_s'] is None else f"{c['previous_completed_at_s']:.0f}"
        cells.append([row['category'].replace('_', ' '), c['task_id'], f"{previous} -> {c['value']:.1f} {c['unit']}",
            f"{old_time} -> {c['completed_at_s']:.0f}", f"{c['gap_over_deadline']:.2f}",
            f"{c['travel_s_at1m_per_s']:.1f} / {c['service_s']:.0f}"])
    fig, ax = plt.subplots(figsize=(14, 4.9))
    ax.axis('off')
    table = ax.table(cellText=cells, colLabels=['Observed event', 'Task', 'Previous -> measured', 'Check times (s)', 'Gap / period', 'Move / service (s)'],
        loc='center', cellLoc='left', colLoc='left', colWidths=[.23, .13, .24, .14, .1, .16])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.9)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#d9e0e5')
        if r == 0:
            cell.set_facecolor('#224b63');cell.get_text().set_color('white')
        elif r % 2 == 0:
            cell.set_facecolor('#f0f4f6')
    ax.set_title('Predeclared trial0 / events proposal0: public observations, not fault-onset labels', pad=18, fontsize=13)
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(args.output / f'm12c-evidence.{ext}', dpi=180)
    plt.close(fig)
    print('M12C_REPORT_COMPLETE', json.dumps({k: result[k] for k in ('actual_closed_loop_records', 'distinct_case_signatures', 'distinct_physical_processes')}), flush=True)


if __name__ == '__main__':
    main()
