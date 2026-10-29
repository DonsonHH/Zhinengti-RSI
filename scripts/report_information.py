"""Aggregate every original condition; descriptive figures, no model choice."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


FOLDERS = ('development-0', 'development-1', 'validation-0', 'validation-1',
           'test-0', 'test-1', 'test-2', 'test-3')
BINS = ('0-30', '30-60', '60-120', '120-300', '300-600', '600-1800')


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def finalize_errors(values):
    result = {}
    for label, row in values.items():
        n = row['n']
        result[label] = dict(row,
            mean_error=row['sum_error'] / n,
            MAE=row['sum_absolute_error'] / n,
            RMSE=(row['sum_squared_error'] / n)**.5,
            false_positive_fraction_of_predicted=row['false_positive'] / row['predicted_positive'] if row['predicted_positive'] else None,
            recall=row['true_positive'] / row['reference_positive'] if row['reference_positive'] else None)
    return result


def frame_summary(frames):
    counts = Counter()
    for row in frames:
        counts['frames'] += 1
        counts['positive_current'] += int(row['q_check'] > 0)
        counts['positive_current_wait'] += int(row['q_check'] > 0 and row['actual_action'] == 'WAIT')
        counts['WAIT_with_net_helpful_current'] += int(row['actual_action'] == 'WAIT' and row['net_helpful_current_exists'])
        counts['WAIT_defers_best_current_net_helpful'] += int(row['layer'] == 'WAIT_defers_net_helpful_current_action')
        counts['future_positive_reference_zero'] += int(row['future_selected_predicted_signal'] is not None and
            row['future_selected_predicted_signal'] > 0 and row['future_selected_reference_signal'] <= 0)
        counts['future_best_net_helpful'] += int(row['best_future_missed_delta'] is not None and
            (row['best_future_missed_delta'] < 0 or row['best_future_missed_delta'] == 0 and
             row['best_future_wait_delta_s'] is not None and row['best_future_wait_delta_s'] < -1e-9))
    return dict(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    conditions, frames, source_cases = [], [], []
    errors = defaultdict(Counter)
    layers = defaultdict(Counter)
    fusion, cache, sources = Counter(), Counter(), Counter()
    folder_totals = {}
    for name in FOLDERS:
        folder = args.root / name
        top = read(folder / 'summary.json')
        if top['conditions'] != 16:
            raise RuntimeError('Incomplete original condition coverage')
        folder_totals[name] = top
        for index in range(16):
            path = folder / f'condition-{index:02}'
            row = read(path / 'summary.json')
            conditions.append(dict(row, folder=name, condition_index=index))
            source_cases.extend(row['source_cases'])
            for key, value in row['error_statistics'].items():
                errors[key].update(value)
            for alias, value in row['actual_revised_decision_layers'].items():
                layers[alias].update(value)
            fusion.update(row['thermal_fusion_unique_and_tied_positive_counts'])
            cache.update(row['thermal_raw_and_assimilated_cache_provenance'])
            sources.update(row['all_candidate_winner_source_counts'])
            with gzip.open(path / 'revised-frame-diagnostics.jsonl.gz', 'rt', encoding='utf-8') as stream:
                for line in stream:
                    item = json.loads(line)
                    item.update(folder=name, condition_index=index, specification=row['condition'])
                    frames.append(item)
    if len(conditions) != 128 or len(source_cases) != 768 or len({v['case'] for v in source_cases}) != 768:
        raise RuntimeError('Not all original 768 cases and 128 conditions were bound')
    per_case = defaultdict(list)
    for frame in frames:
        per_case[frame['case']].append(frame)
    case_summaries = []
    for case, values in per_case.items():
        first = values[0]
        case_summaries.append({'case': case, 'alias': first['alias'], 'specification': first['specification'],
            'episode_never_checked': first['episode_never_checked'],
            'frame_counts': frame_summary(values), 'layers': dict(Counter(v['layer'] for v in values)),
            'any_net_helpful_current': any(v['net_helpful_current_exists'] for v in values),
            'any_positive_current': any(v['q_check'] > 0 for v in values)})
    strata = {}
    for field in ('phase', 'seed', 'scenario', 'fault_start_s', 'noise_c'):
        strata[field] = {}
        for value in sorted({str(v['specification'][field]) for v in frames}):
            own = [v for v in frames if str(v['specification'][field]) == value]
            strata[field][value] = {'frames': frame_summary(own),
                'layers': dict(Counter(v['layer'] for v in own))}
    overall = {'stage': 'C1-B4-A', 'original_cases_bound': 768, 'old_physical_files': 64,
        'old_Tour_observation_conditions_rebuilt': 128,
        'old_public_history_frames_reconstructed': sum(v['common_Tour_history_frames'] for v in conditions),
        'original_public_SHA_frames_verified': sum(v['original_public_SHA_frames_verified'] for v in conditions),
        'factorized_candidate_root_pairs': sum(v['factorized_candidate_root_pairs'] for v in conditions),
        'hard_legal_actions_across_conditions': sum(v['oracle']['hard_legal_dispatch_candidates'] for v in conditions),
        'all_nonwinner_scores_checked_against_old_frontier': sum(v['positive_frontier_nonwinner_scores_checked'] for v in conditions),
        'actual_original_branches_verified': sum(v['oracle']['executed_original_branches_verified'] for v in conditions),
        'conditions_with_net_helpful_legal_check': sum(v['oracle']['net_helpful_candidates'] > 0 for v in conditions),
        'net_helpful_legal_checks': sum(v['oracle']['net_helpful_candidates'] for v in conditions),
        'net_harmful_legal_checks': sum(v['oracle']['net_harmful_candidates'] for v in conditions),
        'history_positive_influence_roots': sum(v['history_channel_can_uniquely_increase_positive_signal_roots'] for v in conditions),
        'history_endpoint_no_positive_influence_roots': sum(v['history_channel_endpoint_no_positive_influence_proofs'] for v in conditions),
        'error_statistics': finalize_errors(errors),
        'actual_revised_layers': {k: dict(v) for k, v in layers.items()},
        'actual_revised_frames': frame_summary(frames),
        'thermal_fusion_unique_and_tied_positive_counts': dict(fusion),
        'thermal_raw_and_assimilated_cache_provenance': dict(cache),
        'lexical_winner_sources_including_zero_signal': dict(sources),
        'case_summaries': case_summaries, 'actual_revised_strata': strata,
        'folder_totals': folder_totals,
        'new_policy_trials': 0, 'new_physical_instances': 0, 'neural_updates': 0, 'LLM_calls': 0,
        'scope': 'Post hoc descriptive old-data diagnostic. All candidate roots include shared histories and future opportunities, not independent samples. Actual revised decisions exist only in development/validation. Test data retain original Tour fallback.',
        'analytic_scope': 'One legal CHECK then fixed Tour only; deterministic measurement-noise semantics retained. An existential beneficial action uses evaluator future truth and is not an implementable online oracle.'}
    write(args.output / 'aggregate.json', overall)
    write(args.output / 'all-condition-summaries.json', conditions)
    with gzip.open(args.output / 'all-revised-frames.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for row in frames:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    colors = ('#2d638e', '#5a8c82', '#ad7c4b', '#8065a4', '#b95d50')
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8))
    methods = [('fused', 'all_legal/thermal'), ('history cache', 'thermal_source/history_assimilated'),
               ('nominal', 'thermal_source/nominal'), ('bounded trend', 'thermal_source/measured_bounded_trend'),
               ('raw CVAE', 'thermal_source/cvae_raw')]
    for color, (label, prefix) in zip(colors, methods):
        values = [overall['error_statistics'].get(prefix + '/' + horizon) for horizon in BINS]
        axes[0].plot(range(6), [v['mean_error'] if v else np.nan for v in values], marker='o', color=color, label=label)
        axes[1].plot(range(6), [100 * v['false_positive_fraction_of_predicted'] if v and v['false_positive_fraction_of_predicted'] is not None else np.nan for v in values], marker='o', color=color, label=label)
    for ax in axes:
        ax.set_xticks(range(6), BINS, rotation=20)
        ax.set_xlabel('Predicted completion minus current public time (s)')
        ax.grid(alpha=.2)
    axes[0].axhline(0., color='#999999', lw=.8)
    axes[0].set_ylabel('Mean predicted signal minus physical reference')
    axes[1].set_ylabel('Reference zero among predicted positive (%)')
    axes[1].set_ylim(-2, 102)
    axes[0].legend(fontsize=8, loc='best')
    fig.suptitle('All legal thermal candidates: forecast error versus completion horizon')
    fig.text(.5, .01, 'All 128 archived conditions; repeated candidate histories are dependent. Signals are not calibrated probabilities.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .94))
    for suffix in ('png', 'svg'):
        fig.savefig(args.output / ('b4a-horizon-error.' + suffix), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.))
    for ax, alias in zip(axes, ('C0', 'C1')):
        own = [v for v in frames if v['alias'] == alias]
        categories = [('WAIT, episode never checked', lambda v: v['actual_action'] == 'WAIT' and v['episode_never_checked'], '#8998a6', 'o', 6),
                      ('WAIT, episode later checked', lambda v: v['actual_action'] == 'WAIT' and not v['episode_never_checked'], '#6a8da8', 'o', 7),
                      ('WAIT, best current net-helpful (offline)', lambda v: v['layer'] == 'WAIT_defers_net_helpful_current_action', '#c37943', 'x', 20),
                      ('Actual CHECK', lambda v: v['actual_action'] == 'CHECK', '#398575', '*', 75)]
        for label, take, color, marker, size in categories:
            part = [v for v in own if take(v)]
            ax.scatter([v['q_check'] for v in part], [v['q_wait'] for v in part], c=color, marker=marker, s=size, alpha=.65, label=label)
        limit = max([max(v['q_check'], v['q_wait']) for v in own] + [.01]) * 1.04
        ax.plot([0, limit], [0, limit], '--', color='#555555', lw=.8)
        ax.set(xlim=(-.005, limit), ylim=(-.005, limit), xlabel='Best admissible current CHECK surrogate', ylabel='Best future WAIT surrogate', title=alias + ': original development/validation only')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, loc='upper left')
    fig.suptitle('Why original revised decisions CHECK or WAIT')
    fig.text(.5, .01, 'Offline helpfulness is an evaluator label, not public information. Test split executed Tour fallback; no C0/C1 test invented.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .94))
    for suffix in ('png', 'svg'):
        fig.savefig(args.output / ('b4a-check-wait.' + suffix), dpi=180)
    plt.close(fig)
    print('INFORMATION_REPORT_DONE', len(conditions), len(source_cases), len(frames), flush=True)


if __name__ == '__main__':
    main()
