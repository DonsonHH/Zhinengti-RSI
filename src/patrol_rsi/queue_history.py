"""Read-only adapter from completed public history to a free queue state.

Callers must cut histories before calling. Future measurements are rejected,
and assignments at the current dispatch are not yet part of the public state.
"""
from __future__ import annotations

import math

from .queue_diagnostics import Measurement, QueueState, TaskEvidence

MEASUREMENTS = {'thermal': ('temperature_c', 'degC'),
    'electrical': ('contact_temperature_c', 'degC'),
    'cooling': ('pod_flow_kg_s', 'kg/s'),
    'control': ('controller_sensor_error_c', 'degC')}


def history_state(plan, *, now, agent, assignments, events, weights, global_used):
    if any(r['time_s'] > now for r in events):
        raise ValueError('Future measurements cannot enter public history')
    if any(r['time_s'] >= now for r in assignments):
        raise ValueError('Only assignments strictly before this dispatch are public')
    if any(not r['calibrated'] or r['role'] != plan.tasks[r['task']].required_role for r in events):
        raise ValueError('Only qualified, calibrated reference measurements are accepted')
    last = [-1.] * len(plan.tasks)
    history = {}
    for event in sorted(events, key=lambda r: r['time_s']):
        task = event['task'];kind = plan.tasks[task].kind
        key, unit = MEASUREMENTS[kind]
        reading = Measurement(float(event['time_s']), float(event['measurement'][key]), unit)
        last[task] = reading.time_s
        history.setdefault(task, []).append((reading, bool(event['alarm'])))
    evidence = []
    for task, readings in history.items():
        evidence.append(TaskEvidence(task, readings[-1][0], readings[-2][0] if len(readings) >= 2 else None, readings[-1][1]))
    cursor, completed, detours = 0, 0, 0
    position = tuple(agent['position'])
    for row in sorted((r for r in assignments if r['agent'] == agent['id']), key=lambda r:r['time_s']):
        finish = row['time_s'] + math.ceil((row['distance_m'] + row['service_s']) / 5 - 1e-10) * 5
        if finish > now:
            raise ValueError('A busy inspector cannot start another check')
        task = row['task']
        if task not in plan.tour:
            raise ValueError('Historical assignment is outside ownership')
        if task == plan.tour[cursor]:
            cursor = (cursor + 1) % len(plan.tour)
            completed += 1
            if completed == len(plan.tour):
                completed = detours = 0
        else:
            detours += 1
        position = plan.tasks[task].position_m
    return QueueState(float(now), agent['id'], agent['role'], tuple(position), cursor,
        tuple(last), tuple(float(w) for w in weights), tuple(evidence), detours, completed, global_used)
