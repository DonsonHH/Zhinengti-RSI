"""Deployable one-check skill: inputs are explicit public candidate snapshots.

Targets, scenario names, seeds and future process arrays are not accepted here.
The one-global-action budget matches the offline tour-continuation target.
"""
from __future__ import annotations

import numpy as np

from .public_dispatch import SIGNALS, choose as tour_choose

FEATURE_NAMES = tuple(f'candidate_{i}' for i in range(16)) + SIGNALS + ('raw_remote_thermal_excess',) + (
    'return_slack_over_300', 'finish_eta_over_300', 'baseline_age', 'baseline_eta',
    'baseline_severity', 'baseline_return_slack_over_300', 'slack_difference_over_300',
    'mean_legal_age', 'max_legal_age', 'baseline_completed_over_32', 'legal_count_over_32', 'free_agents_over_18')
METHODS = ('tour', 'cue_rule', 'actor', 'net', 'target', 'untrained', 'ridge')


def feature_rows(groups):
    """Return (group, optional candidate, 34-vector) without reading an environment."""
    result = []
    for g in groups:
        rows = g['candidates']
        if not rows:
            continue
        base = next(r for r in rows if r['task'] == g['baseline_next'])
        ages = [r['features'][0] for r in rows]
        for r in rows:
            if not r['is_optional']:
                continue
            x = list(r['features']) + [r['signals'][k] for k in SIGNALS + ('raw_remote_thermal_excess',)] + [
                r['minimum_return_slack_s'] / 300, (r['candidate_finish_s'] - g['time_s']) / 300,
                base['features'][0], base['features'][1], base['features'][2], base['minimum_return_slack_s'] / 300,
                (base['minimum_return_slack_s'] - r['minimum_return_slack_s']) / 300,
                float(np.mean(ages)), max(ages), g['baseline_completed'] / 32, len(rows) / 32, len(groups) / 18]
            x = np.asarray(x, dtype=np.float32)
            if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all():
                raise ValueError('Public skill feature schema')
            result.append((g, r, x))
    return result


def choose(groups, method, *, used=False, threshold=0., score=None):
    if method not in METHODS:
        raise ValueError('Unknown one-check skill')
    choices, logs = tour_choose(groups, 'tour')
    if used or method == 'tour' or threshold == 'off':
        return choices, logs, False
    candidates = feature_rows(groups)
    ranked = []
    if method in ('net', 'target', 'untrained', 'ridge'):
        if score is None:
            raise ValueError('Learned skill requires a frozen scorer')
        values = np.asarray(score(np.stack([x for _, _, x in candidates]))) if candidates else np.zeros(0)
        if values.shape != (len(candidates),) or not np.isfinite(values).all():
            raise ValueError('Invalid skill values')
        ranked = [(float(v), g, r) for (g, r, _), v in zip(candidates, values) if v > float(threshold)]
    elif method == 'cue_rule':
        ranked = [((.01 + r['cue_strength']) * r['asset_weight_scaled'] / (.03 + r['eta_over_deadline']), g, r)
            for g, r, _ in candidates if r['cue_strength'] >= .05 and not r['known_measured_alarm']]
    elif method == 'actor':
        for g, r, _ in candidates:
            baseline = next(a for a in g['candidates'] if a['task'] == g['baseline_next'])
            advantage = r['actor_score'] - baseline['actor_score']
            if advantage > 0:
                ranked.append((advantage, g, r))
    if not ranked:
        return choices, logs, False
    value, group, candidate = max(ranked, key=lambda v: v[0])
    choices[group['agent']] = candidate['task']
    log = next(r for r in logs if r['agent'] == group['agent'])
    log.update(task=candidate['task'], tool='one_' + method, rule_index=None,
        cue_strength=candidate['cue_strength'], known_measured_alarm=candidate['known_measured_alarm'], skill_value=value)
    return choices, logs, True
