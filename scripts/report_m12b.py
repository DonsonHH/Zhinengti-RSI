"""M12-B trial pairing, protocol-update evidence and bounded-cost figures."""
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
    p.add_argument('--old-review', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    trials, compact, meta, costs = {}, [], [], {}
    unique_cases, process_keys = set(), set()
    demo_rows = []
    for trial in range(3):
        folder = args.results / f'trial-{trial}'
        index = json.loads((folder / 'case-index.json').read_text())
        lookup = {r['case']: r for r in index}
        groups = json.loads((folder / 'test-groups.json').read_text())
        selections = json.loads((folder / 'selections-frozen.json').read_text())
        compact.extend([r | {'trial': trial} for r in index])
        unique_cases.update(lookup)
        process_keys.update(r['process_key'] for r in index)
        grouped = {}
        for group, keys in groups.items():
            rows = [lookup[k] for k in keys]
            weights = sum(r['event_weight_sum'] for r in rows)
            count = sum(r['metrics']['anomaly_segments'] for r in rows)
            miss = sum(r['metrics']['missed_segments'] for r in rows)
            dispatches = sum(r['public_selection']['dispatches'] for r in rows)
            grouped[group] = {'observed_conditions': len(rows), 'anomaly_segments': count, 'missed_segments': miss,
                'miss_rate': miss / count if count else None,
                'weighted_censored_delay_s': sum(r['weighted_delay_sum'] for r in rows) / weights if weights else None,
                'exact_violating_tasks': sum(r['metrics']['tasks_violating_exact_gap'] for r in rows),
                'maximum_exact_gap_ratio': max(r['metrics']['maximum_exact_gap_ratio'] for r in rows),
                'mean_extra_checks': float(np.mean([r['metrics']['selected_detours'] for r in rows])),
                'mean_estimator_calls': float(np.mean([r['metrics']['estimator_calls'] for r in rows])),
                'mean_travel_m': float(np.mean([r['metrics']['travel_m'] for r in rows])),
                'mean_service_s': float(np.mean([r['metrics']['service_s'] for r in rows])),
                'false_confirmations': sum(r['metrics']['false_measured_confirmations'] for r in rows),
                'different_from_actor_fraction': sum(r['public_selection']['different_from_actor'] for r in rows) / dispatches,
                'ppo_tool_fraction': sum(r['public_selection']['ppo_dispatches'] for r in rows) / dispatches,
                'rule_tool_fraction': sum(r['public_selection']['rule_dispatches'] for r in rows) / dispatches,
                'single_candidate_fraction': sum(r['public_selection']['forced_single_candidate'] for r in rows) / dispatches}
            if trial == 0 and group in ('tour', 'direct3', 'model_meta3', 'random_search13'):
                chosen = next(r for r in rows if r['specification']['scenario'] == 'coldplate_fouling'
                    and r['specification']['duration_s'] == 180 and r['specification']['seed'] == 6601 and r['specification']['noise_c'] == 1.)
                demo_rows.append({'trial': trial, 'group': group, 'case': chosen['case'], 'path': chosen['path'],
                    'version_id': chosen['version_id'], 'workflow': chosen['workflow']})
        trials[str(trial)] = grouped
        proposals = [json.loads(p.read_text()) for p in sorted((folder / 'proposals').glob('*.json'))]
        updates = [json.loads(p.read_text()) for p in sorted((folder / 'meta').glob('*.json'))]
        meta.extend([{'trial': trial, 'method': r['method'], 'round': r['round'], 'valid': r['valid'],
            'accepted': r['accepted'], 'incumbent_id': r['incumbent_id'], 'candidate_id': r['candidate_id'],
            'retained_id': r['retained_id'], 'old_validator': r['arms']['incumbent']['selected']['validator'],
            'new_validator': r['arms']['candidate']['selected']['validator']} for r in updates])
        method_cost = {}
        for method in ('direct', 'random_search', 'random_meta', 'model_meta'):
            entries = [r for r in proposals if r['tag'] == method or r['tag'].startswith(method + '-')]
            generations = [r['generation'] for r in entries if 'generation' in r]
            generations += [r['generation'] for r in updates if r['method'] == method and 'generation' in r]
            method_cost[method] = {'task_candidate_slots': len(entries), 'protocol_attempts': sum(r['method'] == method for r in updates),
                'model_calls': len(generations), 'accepted_task_proposals': sum(r['accepted'] for r in entries),
                'duplicate_task_proposals': sum(r['reason'] == 'duplicate' for r in entries),
                'invalid_task_proposals': sum(not r['valid'] for r in entries),
                **{k: sum(r[k] for r in generations) for k in ('input_tokens', 'output_tokens', 'generation_seconds')}}
        costs[str(trial)] = method_cost
    names = ['tour', 'actor', 'human', 'random_search13', 'direct3', 'direct13', 'random_meta3', 'random_meta5', 'model_meta3']
    main_groups = {}
    for name in names + ['model_meta3/no_actor', 'model_meta3/nominal']:
        values = [trials[str(t)][name] for t in range(3)]
        main_groups[name] = {'per_generation_trial': values,
            'mean_miss_rate': float(np.mean([r['miss_rate'] for r in values])),
            'mean_weighted_censored_delay_s': float(np.mean([r['weighted_censored_delay_s'] for r in values])),
            'mean_extra_checks': float(np.mean([r['mean_extra_checks'] for r in values])),
            'maximum_exact_gap_ratio': max(r['maximum_exact_gap_ratio'] for r in values)}
    result = {'trials': trials, 'main_groups': main_groups, 'meta_updates': meta, 'cost_by_trial': costs,
        'actual_closed_loop_records': len(compact), 'distinct_case_signatures_across_trials': len(unique_cases),
        'distinct_physical_processes_across_trials': len(process_keys), 'test_physical_processes': 16,
        'test_observation_conditions': 32, 'total_test_group_references': 1056,
        'note': 'Three proposal repetitions on the same physical/noise conditions; no new policy-training seeds or independent-plant error bars'}
    (args.output / 'aggregate.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'compact-metrics.json.gz').write_bytes(gzip.compress(json.dumps(compact).encode(), mtime=0))
    (args.output / 'demo-selection.json').write_text(json.dumps(demo_rows, indent=2) + '\n')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    old = json.loads(args.old_review.read_text())
    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout='constrained')
    programs = old['selection_rank_ordered_programs']
    ax.scatter([r['mean_public_weighted_age'] for r in programs], [r['weighted_censored_delay_s'] for r in programs],
        color='#8c9aa4', s=48, label='All11development programs')
    for key, color, label in [('initial_program', '#17445e', 'Initial'), ('selected_model_program', '#b75e40', 'Selected byLLM')]:
        r = old[key]
        ax.scatter(r['mean_public_weighted_age'], r['weighted_censored_delay_s'], color=color, s=90, label=label)
        ax.annotate(label, (r['mean_public_weighted_age'], r['weighted_censored_delay_s']), xytext=(7, 7), textcoords='offset points')
    ax.set(xlabel='Public mean inspection-age ratio (lower is better)', ylabel='Hidden weighted censored delay (s)',
        title='M12-A retrospective: a better proxy can hide a worse task outcome')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=.2)
    for ext in ('png', 'svg'):
        fig.savefig(args.output / f'm12b-proxy-review.{ext}', dpi=180)
    plt.close(fig)

    labels = ['Tour', 'FrozenPPO', 'Human', 'Random13', 'Fixed3', 'Fixed13', 'RandomMeta3', 'RandomMeta5', 'ModelMeta3']
    x = np.arange(len(names))
    colors = ['#7b8992', '#274d64', '#79948c', '#bb9b47', '#527f98', '#406779', '#9b807a', '#8a6d68', '#b45e40']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    for ax, metric, field, title, factor in [
        (axes[0], 'mean_miss_rate', 'miss_rate', 'Missed anomaly segments (%)', 100),
        (axes[1], 'mean_weighted_censored_delay_s', 'weighted_censored_delay_s', 'Weighted censored delay (s)', 1),
        (axes[2], 'mean_extra_checks', 'mean_extra_checks', 'Optional checks / episode', 1)]:
        ax.bar(x, [main_groups[n][metric] * factor for n in names], color=colors)
        for trial, offset in enumerate((-.15, 0, .15)):
            ax.scatter(x + offset, [trials[str(trial)][n][field] * factor for n in names], color='#222222', s=12, zorder=4)
        ax.set_xticks(x, labels, rotation=45, ha='right')
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('M12-B fresh-task test: dots are3proposal trials on shared32observation conditions', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    for ext in ('png', 'svg'):
        fig.savefig(args.output / f'm12b-task-results.{ext}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for method, offset, color in [('random_meta', -.15, '#8a6d68'), ('model_meta', .15, '#b45e40')]:
        counts = [sum(r['accepted'] for r in meta if r['trial'] == t and r['method'] == method) for t in range(3)]
        axes[0].bar(np.arange(3) + offset, counts, width=.3, color=color, label=method)
    axes[0].set(xticks=np.arange(3), xticklabels=['Trial0', 'Trial1', 'Trial2'], ylabel='Accepted protocol updates /2attempts', ylim=(0, 2.4), title='Did the improver change?')
    axes[0].legend(fontsize=8)
    for method, color in [('direct3', '#527f98'), ('random_meta3', '#8a6d68'), ('model_meta3', '#b45e40')]:
        axes[1].plot(np.arange(3), [trials[str(t)][method]['weighted_censored_delay_s'] for t in range(3)],
            marker='o', color=color, label=method)
    axes[1].set(xticks=np.arange(3), xticklabels=['Trial0', 'Trial1', 'Trial2'], ylabel='Fresh-task weighted delay (s)', title='Same3task-proposal budget')
    axes[1].legend(fontsize=8)
    methods = ['direct', 'random_search', 'random_meta', 'model_meta']
    axes[2].bar(np.arange(4), [np.mean([costs[str(t)][m]['model_calls'] for t in range(3)]) for m in methods], color=['#406779', '#bb9b47', '#8a6d68', '#b45e40'])
    axes[2].set(xticks=np.arange(4), xticklabels=['Fixed13', 'Random13', 'RandomMeta', 'ModelMeta'], ylabel='All preparation + task model calls', title='Full model-call accounting')
    axes[2].tick_params(axis='x', rotation=35)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(args.output / f'm12b-meta-and-cost.{ext}', dpi=180)
    plt.close(fig)
    print('M12B_REPORT_COMPLETE', json.dumps({k: result[k] for k in ('actual_closed_loop_records', 'distinct_case_signatures_across_trials', 'distinct_physical_processes_across_trials')}), flush=True)


if __name__ == '__main__':
    main()
