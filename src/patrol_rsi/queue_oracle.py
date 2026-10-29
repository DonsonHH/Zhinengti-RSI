"""Offline window Oracle and observed outcomes. Never imported by controllers.

The Oracle knows hidden windows and assumes a within-window qualified visit
can confirm them. Actual alarms remain separate, including measurement noise.
It is an explanatory reference for these recorded branches, not a policy or
an optimum over all possible actions.
"""
from __future__ import annotations


def _confirmation(events, window):
    times = [r['time_s'] for r in events if r['task'] == window['task'] and r['alarm']
        and window['onset_s'] <= r['time_s'] <= window['end_s']]
    return min(times) if times else None


def offline_oracle(diagnostic, tour_record, detour_record):
    windows = tour_record['anomaly_intervals']
    coords = lambda rows: [(r['task'], r['onset_s'], r['end_s'], r['weight']) for r in rows]
    if coords(windows) != coords(detour_record['anomaly_intervals']):
        raise ValueError('Paired branches must share identical hidden windows')
    actual, opportunities = [], []
    owned = {r['task'] for r in diagnostic['all_normal_tour_comparisons']}
    for window in windows:
        before = _confirmation(tour_record['inspection_events'], window)
        after = _confirmation(detour_record['inspection_events'], window)
        actual.append({'task': window['task'], 'task_id': window['task_id'],
            'onset_s': window['onset_s'], 'end_s': window['end_s'],
            'tour_confirmation_s': before, 'detour_confirmation_s': after})
        if window['task'] in owned and window['end_s'] >= diagnostic['time_s']:
            prefix = [r['time_s'] for r in tour_record['inspection_events'] if r['task'] == window['task']
                and r['time_s'] <= diagnostic['time_s']]
            found = []
            for branch in ('tour_plan', 'detour_plan'):
                times = prefix + [r['time_s'] for r in diagnostic[branch]['mission_completions']
                    if r['task'] == window['task']]
                found.append(any(window['onset_s'] <= t <= window['end_s'] for t in times))
            opportunities.append({'task_id': window['task_id'], 'onset_s': window['onset_s'],
                'end_s': window['end_s'], 'tour_has_window_visit': found[0], 'detour_has_window_visit': found[1]})
    missed = [sum(r[key] is None for r in actual) for key in ('tour_confirmation_s', 'detour_confirmation_s')]
    for record, n in zip((tour_record, detour_record), missed):
        if record['metrics']['missed_segments'] != n:
            raise ValueError('Independent window/alarm count differs from saved metrics')
    target = next((r for r in actual if r['task'] == diagnostic['target']
        and r['tour_confirmation_s'] is not None and r['detour_confirmation_s'] is not None
        and r['end_s'] >= diagnostic['time_s']), None)
    return {'view': 'offline_evaluator_only',
        'not_for_online_input_or_candidate_selection': True,
        'oracle_scope': 'Hidden-window visit opportunity for the two already recorded branches, not a global optimum',
        'actual_missed_before': missed[0], 'actual_missed_after': missed[1],
        'actual_missed_delta': missed[1] - missed[0],
        'target_confirmation_lead_s': target['tour_confirmation_s'] - target['detour_confirmation_s'] if target else None,
        'lost_confirmations': [r for r in actual if r['tour_confirmation_s'] is not None and r['detour_confirmation_s'] is None],
        'gained_confirmations': [r for r in actual if r['tour_confirmation_s'] is None and r['detour_confirmation_s'] is not None],
        'changed_confirmations': [r for r in actual if r['tour_confirmation_s'] != r['detour_confirmation_s']],
        'window_opportunity_missed_delta': sum(int(r['tour_has_window_visit']) - int(r['detour_has_window_visit']) for r in opportunities),
        'window_opportunities': opportunities}
