"""Evaluation-only event labels and censored confirmation delays."""
from __future__ import annotations

import numpy as np

from .aisles import ROLES


def task_labels(env):
    a, c = env.trace.arrays, env.config
    temperatures = a['rack_temperature_c']
    contact = a['contact_temperature_c']
    flow = a['rack_flow_kg_s'].reshape(360, c.pods, 8).sum(2)
    sensor = abs((a['public_sensor_temperature_c'] - temperatures).reshape(360, c.pods, 8).mean(2))
    labels, excess = [], []
    for task in env.tasks:
        if task.kind == 'thermal':
            value = temperatures[:, task.index]
            labels.append(value > 65)
            excess.append(np.maximum(value - 65, 0) / 20)
        elif task.kind == 'electrical':
            value = contact[:, task.index]
            labels.append(value > 60)
            excess.append(np.maximum(value - 60, 0) / 30)
        elif task.kind == 'cooling':
            value = flow[:, task.index]
            labels.append(value < 33.6)
            excess.append(np.maximum(1 - value / 33.6, 0))
        else:
            value = sensor[:, task.index]
            labels.append(value > 1)
            excess.append(value / 18)
    return np.stack(labels, axis=1), np.stack(excess, axis=1)


def anomaly_intervals(labels, excess, tasks, weights):
    rows = []
    for j, task in enumerate(tasks):
        values = np.r_[False, labels[:, j], False].astype(int)
        starts, ends = np.flatnonzero(np.diff(values) == 1), np.flatnonzero(np.diff(values) == -1)
        for start, end in zip(starts, ends):
            if end - start < 3:
                continue
            onset, finish = int(start) * 5., min(1800., int(end) * 5.)
            # Binary endpoint labels represent their preceding 5s bins.
            # Three positive bins span 15s; first confirmation is still a
            # completion measurement at a labeled-positive sampled endpoint.
            rows.append({'task': j, 'task_id': task.task_id, 'kind': task.kind,
                         'onset_s': onset, 'end_s': finish, 'samples': int(end - start),
                         'weight': float(weights[j] * (1 + excess[start:end, j].mean()))})
    return rows


def evaluate_episode(env):
    labels, excess = task_labels(env)
    intervals = anomaly_intervals(labels, excess, env.tasks, env.asset_weights)
    alarms = {}
    false_confirmations = 0
    for event in env.events:
        if event['alarm']:
            index = min(359, int(event['time_s'] / 5) - 1)
            if labels[index, event['task']]:
                alarms.setdefault(event['task'], []).append(event['time_s'])
            else:
                false_confirmations += 1
    for item in intervals:
        found = [t for t in alarms.get(item['task'], []) if item['onset_s'] <= t <= item['end_s']]
        item['first_confirmation_s'] = min(found) if found else None
        item['detected'] = bool(found)
        item['delay_or_censor_s'] = (min(found) if found else item['end_s']) - item['onset_s']
    weights = np.array([r['weight'] for r in intervals])
    delays = np.array([r['delay_or_censor_s'] for r in intervals])
    detected = [r for r in intervals if r['detected']]
    metrics = {'anomaly_segments': len(intervals), 'confirmed_segments': len(detected),
               'missed_segments': len(intervals) - len(detected),
               'miss_rate': 1 - len(detected) / len(intervals) if intervals else None,
               'consequence_weighted_censored_delay_s': float((weights * delays).sum() / weights.sum()) if len(weights) else None,
               'detected_only_mean_delay_s': float(np.mean([r['delay_or_censor_s'] for r in detected])) if detected else None,
               'false_measured_confirmations': false_confirmations,
               'completed_inspections': len(env.events),
               'final_professional_coverage': float(env.ever.mean()),
               'mean_overdue_fraction': float(np.mean([r['overdue_fraction'] for r in env.scalar_trace])),
               'maximum_inspection_age_s': max(r['maximum_age_s'] for r in env.scalar_trace),
               'travel_m': float(sum(a['travel_m'] for a in env.agents)),
               'service_s': float(sum(a['service_s'] for a in env.agents)),
               'idle_s': float(sum(a['idle_s'] for a in env.agents)),
               'history_estimator_inference_s': env.inference_seconds,
               'public_return': float(sum(r.get('public_reward', 0) for r in env.scalar_trace)),
               'invalid_assignments': 0, 'duplicate_reservations': 0,
               'unserved_IT_kwh': env.trace.metadata['unserved_IT_kwh']}
    by_role = {}
    for role in ROLES:
        segments = [r for r in intervals if r['kind'] == role]
        flags = [t.kind == role for t in env.tasks]
        by_role[role] = {'segments': len(segments), 'confirmed': sum(r['detected'] for r in segments),
                         'final_coverage': float(env.ever[flags].mean()),
                         'mean_overdue_fraction': float(np.mean([r['per_role_overdue'][role] for r in env.scalar_trace])),
                         'maximum_inspection_age_s': max(r['per_role_maximum_age_s'][role] for r in env.scalar_trace)}
    return metrics, by_role, intervals
