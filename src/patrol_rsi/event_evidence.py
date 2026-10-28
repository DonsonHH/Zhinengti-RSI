"""Bounded, traceable diagnostics derived only from public execution events."""
from __future__ import annotations

import copy

SCHEMAS = {'thermal': ('temperature_c', 'degC'), 'electrical': ('contact_temperature_c', 'degC'),
    'cooling': ('pod_flow_kg_s', 'kg/s'), 'control': ('controller_sensor_error_c', 'degC')}
CATEGORIES = ('new_measured_alert', 'persistent_recheck', 'recovery_confirmation', 'long_completion_gap')
TOOL_CARDS = {
    'coverage': 'Public completed-task coverage, exact recorded check ages and first measured-alert task count; not hidden anomaly detection.',
    'professional_evidence': 'Per-profession valid completed measurements, their units, measured alerts and rechecks; no fault labels.',
    'efficiency': 'Actual movement, service, optional checking and state-estimator calls.',
    'thermal_disagreement': 'Difference between historyCVAE and nominal temperature; thermal only, not calibrated uncertainty.',
    'novelty': 'Candidate-program repetition, number of previous proposals and seen version IDs. Does not detect new plant anomalies.',
    'event_evidence': 'Bounded public measurement sequences with task, time, unit and cost; first observed alarm time is not hidden fault onset.'}


def event_candidates(record):
    """Do not read specification, process, anomaly intervals or hidden metrics."""
    by_agent, offsets = {}, {}
    if len(record['assignments']) != len(record['workflow_decisions']):
        raise ValueError('Every executed dispatch requires one decision record')
    for assignment, decision in zip(record['assignments'], record['workflow_decisions']):
        if (assignment['agent'], assignment['task'], assignment['time_s']) != (decision['agent'], decision['task'], decision['time_s']):
            raise ValueError('Dispatch and public workflow log mismatch')
        by_agent.setdefault(assignment['agent'], []).append((assignment, decision))
    previous, candidates = {}, []
    deadlines = record['workflow_contract']['deadlines_s']
    for number, event in enumerate(record['inspection_events']):
        i, j, t = event['agent'], event['task'], event['time_s']
        offset = offsets.get(i, 0)
        assignment, decision = by_agent[i][offset]
        offsets[i] = offset + 1
        if assignment['task'] != j or assignment['time_s'] >= t:
            raise ValueError('Measurement must follow matching public dispatch')
        if not event['calibrated'] or event['role'] != event['kind']:
            raise ValueError('Only valid qualified observations enter diagnostic evidence')
        field, unit = SCHEMAS[event['kind']]
        value = float(event['measurement'][field])
        old = previous.get(j)
        previous_time, previous_value = (old['time_s'], old['value']) if old else (None, None)
        if previous_time is not None and previous_time >= t:
            raise ValueError('Task measurement clock must increase')
        gap = float(t - (previous_time if previous_time is not None else 0))
        card = {'task_id': event['task_id'], 'profession': event['kind'], 'measurement_field': field,
            'unit': unit, 'completed_at_s': float(t), 'value': value,
            'previous_completed_at_s': previous_time, 'previous_value': previous_value,
            'observed_value_delta': value - previous_value if previous_value is not None else None,
            'interval_basis': 'previous_measurement' if old else 'mission_start_without_prior_measurement',
            'completion_gap_s': gap, 'gap_over_deadline': gap / deadlines[j],
            'observed_transition': event['measurement_transition'], 'measured_alarm': event['alarm'],
            'dispatched_at_s': assignment['time_s'], 'travel_s_at1m_per_s': assignment['distance_m'],
            'service_s': assignment['service_s'], 'optional_check': j != decision['baseline_next'],
            'selected_tool': decision['tool'],
            'source': {'event_index': number, 'agent': i, 'task': j}}
        previous[j] = {'time_s': float(t), 'value': value}
        # Each source observation is a candidate for its observed lifecycle and
        # the long-gap view. The final selector never duplicates a source row.
        if event['measurement_transition'] in CATEGORIES[:3]:
            candidates.append({'category': event['measurement_transition'], 'card': card})
        candidates.append({'category': 'long_completion_gap', 'card': card})
    return candidates


def bounded_case_candidates(record):
    candidates = event_candidates(record)
    selected = []
    for category in CATEGORIES:
        rows = [r for r in candidates if r['category'] == category]
        rows.sort(key=lambda r: (-r['card']['gap_over_deadline'], r['card']['completed_at_s'], r['card']['task_id']))
        selected.extend(rows[:2])
    return selected


def batch_events(case_views, redacted=False):
    """Same public selection for event and redacted arms, at most8source rows."""
    selected, seen, pool = [], set(), []
    for episode, rows in enumerate(case_views):
        for row in rows:
            pool.append(copy.deepcopy(row) | {'episode_alias': f'public_run_{episode}'})
    for category in CATEGORIES:
        rows = [r for r in pool if r['category'] == category]
        rows.sort(key=lambda r: (-r['card']['gap_over_deadline'], r['card']['completed_at_s'],
            r['episode_alias'], r['card']['task_id']))
        taken = 0
        for row in rows:
            source = (row['episode_alias'], row['card']['source']['event_index'])
            if source in seen:
                continue
            seen.add(source)
            selected.append(row)
            taken += 1
            if taken == 2:
                break
    counts = {name: sum(r['category'] == name for r in selected) for name in CATEGORIES}
    if redacted:
        selected = [{'category': r['category'], 'details': 'Task, time, measurement and cost fields withheld by the diagnostic control'} for r in selected]
    return {'maximum_source_events': 8, 'selected_source_events': len(selected), 'category_counts': counts,
        'details_redacted': redacted, 'events': selected,
        'meaning': 'Only executed valid observations; no hidden anomaly onset or correctness label; gaps may begin at mission start.'}


def scalar_cards(record):
    """Separate mean of episode maxima from the global observed worst value."""
    by_role = {}
    for role in SCHEMAS:
        gaps = [r['maximum_gap_including_initial_and_terminal_s'] for r in record['exact_task_gaps']
            if r['task_id'].startswith(role + ':')]
        by_role[role] = {'mean_task_maximum_gap_s': sum(gaps) / len(gaps), 'worst_task_gap_s': max(gaps)}
    return by_role
