"""Explicit public candidate snapshots; controllers never receive simulator truth."""
from __future__ import annotations

import numpy as np

from .improver_task import typed_signal

METHODS = ('tour', 'cue_rule', 'actor')
SIGNALS = ('thermal_excess', 'thermal_disagreement', 'contact_heat', 'flow_deficit', 'verified_sensor_bias')


def snapshot(env, scorer):
    groups, reserved = [], env.reserved()
    for raw_i in np.roll(np.arange(len(env.agents)), -(env.tick % len(env.agents))):
        i = int(raw_i)
        if env.agents[i]['task'] is not None:
            continue
        ids, f, _ = env.options(i, reserved)
        baseline = env.plans[i].tour[env.protected_cursor[i]]
        values = {name: typed_signal(env, ids, f, name)[0] for name in SIGNALS}
        actor_scores = np.asarray(scorer(f)) if len(ids) else np.zeros(0)
        if actor_scores.shape != (len(ids),) or not np.isfinite(actor_scores).all():
            raise ValueError('Invalid public actor scores')
        packet = env.current_packet
        remote = dict(zip(packet['temperature_indices'].tolist(), packet['temperature_c'].tolist()))
        candidates = []
        for k, raw_j in enumerate(ids):
            j, task = int(raw_j), env.tasks[int(raw_j)]
            cert = env.candidate_certificate(i, j)
            if not cert['ok']:
                raise RuntimeError('Public candidate mask disagrees with its certificate')
            cues = {name: float(values[name][k]) for name in SIGNALS}
            cues['raw_remote_thermal_excess'] = max(remote[task.index] - 65, 0) / 20 if task.kind == 'thermal' and task.index in remote else 0.
            row = {'task': j, 'task_id': task.task_id, 'profession': task.kind,
                'actor_score': float(actor_scores[k]), 'features': f[k].tolist(), 'signals': cues,
                'known_measured_alarm': bool(env.measured_alert.get(j, False)),
                'cue_strength': float(max(cues.values())), 'asset_weight_scaled': float(f[k, 6]),
                'eta_over_deadline': float(f[k, 1]), 'candidate_finish_s': cert['candidate_finish_s'],
                'minimum_return_slack_s': cert['minimum_projected_slack_s'],
                'reference_measurement_stamp_s': float(env.last_inspection[j]),
                'raw_remote_temperature_c': float(remote[task.index]) if task.kind == 'thermal' and task.index in remote else None,
                'raw_remote_temperature_time_s': float(packet['time_s']) if task.kind == 'thermal' and task.index in remote else None,
                'nominal_temperature_c': float(env.state.observer.predicted_temperature[task.index]) if task.kind == 'thermal' else None,
                'history_temperature_c': float(env.current_history[task.index]) if task.kind == 'thermal' and env.mode == 'history' else None,
                'baseline_next': int(baseline), 'is_optional': j != baseline}
            candidates.append(row)
        groups.append({'time_s': float(env.clock), 'agent': i, 'role': env.agents[i]['role'],
            'baseline_next': int(baseline), 'position_m': list(env.agents[i]['position']),
            'detours_used': int(env.detours_used[i]), 'baseline_completed': int(env.baseline_completed[i]),
            'candidates': candidates})
    return groups


def choose(groups, method):
    if method not in METHODS:
        raise ValueError('Unknown public controller')
    choices, logs = {}, []
    for group in groups:
        candidates = group['candidates']
        if not candidates:
            choices[group['agent']] = None
            continue
        actor = max(candidates, key=lambda r: r['actor_score'])
        baseline = next(r for r in candidates if r['task'] == group['baseline_next'])
        if method == 'tour':
            selected, tool = baseline, 'return_tour'
        elif method == 'actor':
            selected, tool = actor, 'ppo'
        else:
            cued = [r for r in candidates if r['cue_strength'] >= .05 and not r['known_measured_alarm']]
            if cued:
                selected = max(cued, key=lambda r: (.01 + r['cue_strength']) * r['asset_weight_scaled'] / (.03 + r['eta_over_deadline']))
                tool = 'public_cue'
            else:
                selected, tool = baseline, 'return_tour'
        choices[group['agent']] = selected['task']
        logs.append({'time_s': group['time_s'], 'agent': group['agent'], 'task': selected['task'],
            'baseline_next': group['baseline_next'], 'actor_task': actor['task'], 'tool': tool,
            'rule_index': 0 if tool == 'public_cue' else None, 'legal_candidates': len(candidates),
            'cue_strength': selected['cue_strength'], 'known_measured_alarm': selected['known_measured_alarm'],
            'detours_used_before': group['detours_used'], 'baseline_completed_before': group['baseline_completed']})
    return choices, logs
