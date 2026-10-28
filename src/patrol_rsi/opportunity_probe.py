"""Evaluator-only local one-detour witnesses, quarantined from public policies."""
from __future__ import annotations

from .active_inspection import measured_anomaly
from .inspection_metrics import task_labels, anomaly_intervals


def category(row):
    if row['unconfirmed_cue_witness_count']:
        return 'one_detour_with_prespecified_unconfirmed_cue'
    if row['witness_count']:
        return 'one_detour_without_prespecified_unconfirmed_cue'
    return 'no_one_detour_witness_on_original_tour_states'


class OneDetourProbe:
    def __init__(self, env):
        labels, excess = task_labels(env)
        self.intervals = anomaly_intervals(labels, excess, env.tasks, env.asset_weights)
        self.at_completion = {}
        for number, row in enumerate(self.intervals):
            for tick in range(int(row['onset_s'] / 5), int(row['end_s'] / 5)):
                self.at_completion[(row['task'], (tick + 1) * 5.)] = number
        self.rows = [{'segment_index': i, 'task': r['task'], 'task_id': r['task_id'], 'profession': r['kind'],
            'onset_s_evaluator_only': r['onset_s'], 'end_s_evaluator_only': r['end_s'],
            'witness_count': 0, 'any_cue_witness_count': 0, 'unconfirmed_cue_witness_count': 0,
            'first_witness': None, 'first_unconfirmed_cue_witness': None}
            for i, r in enumerate(self.intervals)]
        self.scanned_free_agent_frames, self.scanned_legal_candidates = 0, 0

    def observe(self, env, groups):
        self.scanned_free_agent_frames += len(groups)
        for group in groups:
            for candidate in group['candidates']:
                self.scanned_legal_candidates += 1
                if not candidate['is_optional']:
                    continue
                key = (candidate['task'], candidate['candidate_finish_s'])
                number = self.at_completion.get(key)
                if number is None:
                    continue
                tick = int(candidate['candidate_finish_s'] / 5) - 1
                measurement = env.trace.measurement(env.tasks[candidate['task']], tick)
                if not measured_anomaly(env.tasks[candidate['task']], measurement):
                    continue
                row = self.rows[number]
                cued = candidate['cue_strength'] >= .05
                unconfirmed = cued and not candidate['known_measured_alarm']
                witness = {'dispatch_time_s': group['time_s'], 'agent': group['agent'],
                    'position_m': group['position_m'], 'candidate': candidate,
                    'completion_measurement_evaluator_only': measurement}
                row['witness_count'] += 1
                row['any_cue_witness_count'] += cued
                row['unconfirmed_cue_witness_count'] += unconfirmed
                if row['first_witness'] is None or (group['time_s'], group['agent']) < (
                        row['first_witness']['dispatch_time_s'], row['first_witness']['agent']):
                    row['first_witness'] = witness
                if unconfirmed and (row['first_unconfirmed_cue_witness'] is None or
                    (group['time_s'], group['agent']) < (row['first_unconfirmed_cue_witness']['dispatch_time_s'], row['first_unconfirmed_cue_witness']['agent'])):
                    row['first_unconfirmed_cue_witness'] = witness

    def finish(self, evaluated_intervals):
        if len(evaluated_intervals) != len(self.rows):
            raise RuntimeError('Probe and event evaluator disagree')
        for probe, interval in zip(self.rows, evaluated_intervals):
            if probe['task'] != interval['task'] or probe['onset_s_evaluator_only'] != interval['onset_s']:
                raise RuntimeError('Mismatched evaluator segment')
            probe['baseline_detected'] = interval['detected']
            probe['baseline_first_confirmation_s'] = interval['first_confirmation_s']
            probe['opportunity_class'] = category(probe)
        return {'scope': 'Evaluator-only one-detour witnesses on original tour states; not an optimal schedule or global bound',
            'scanned_free_agent_frames': self.scanned_free_agent_frames,
            'scanned_legal_candidates': self.scanned_legal_candidates, 'segments': self.rows}
