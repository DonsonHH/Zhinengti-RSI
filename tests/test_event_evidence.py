import copy
import json
import unittest

from patrol_rsi.event_evidence import (TOOL_CARDS, batch_events, bounded_case_candidates,
    event_candidates, scalar_cards)


def public_record():
    events, assignments, decisions = [], [], []
    for time, value, transition in [(30., 70., 'new_measured_alert'), (60., 72., 'persistent_recheck'),
            (150., 60., 'recovery_confirmation')]:
        events.append({'agent': 0, 'task': 0, 'task_id': 'thermal:0', 'kind': 'thermal', 'role': 'thermal',
            'time_s': time, 'measurement': {'temperature_c': value}, 'measurement_transition': transition,
            'alarm': value > 65, 'calibrated': True})
        assignments.append({'agent': 0, 'task': 0, 'time_s': time - 15, 'distance_m': 5., 'service_s': 10.})
        decisions.append({'agent': 0, 'task': 0, 'time_s': time - 15, 'baseline_next': 0, 'tool': 'return_tour'})
    return {'inspection_events': events, 'assignments': assignments, 'workflow_decisions': decisions,
        'workflow_contract': {'deadlines_s': [300.]},
        'exact_task_gaps': [{'task_id': 'thermal:0', 'maximum_gap_including_initial_and_terminal_s': 90.},
            {'task_id': 'electrical:0', 'maximum_gap_including_initial_and_terminal_s': 300.},
            {'task_id': 'cooling:0', 'maximum_gap_including_initial_and_terminal_s': 400.},
            {'task_id': 'control:0', 'maximum_gap_including_initial_and_terminal_s': 450.}]}


class PublicEvidenceTests(unittest.TestCase):
    def test_actual_measurement_sequence_and_units(self):
        candidates = event_candidates(public_record())
        alarm = next(r['card'] for r in candidates if r['category'] == 'new_measured_alert')
        self.assertEqual(alarm['unit'], 'degC')
        self.assertIsNone(alarm['previous_value'])
        self.assertIsNone(alarm['previous_completed_at_s'])
        self.assertEqual(alarm['completion_gap_s'], 30.)
        recheck = next(r['card'] for r in candidates if r['category'] == 'persistent_recheck')
        self.assertEqual((recheck['previous_value'], recheck['previous_completed_at_s']), (70., 30.))
        self.assertEqual(recheck['observed_value_delta'], 2.)
        self.assertEqual(recheck['travel_s_at1m_per_s'] + recheck['service_s'], 15.)

    def test_hidden_records_do_not_affect_the_diagnostic(self):
        original = public_record()
        changed = copy.deepcopy(original)
        changed.update(specification={'scenario': 'future_hidden_fault'}, process={'private_truth': [999]},
            anomaly_intervals=[{'onset_s': 1, 'end_s': 2}], metrics={'missed_segments': 1000})
        self.assertEqual(event_candidates(original), event_candidates(changed))
        self.assertEqual(batch_events([bounded_case_candidates(original)]), batch_events([bounded_case_candidates(changed)]))

    def test_bounded_batch_does_not_duplicate_source_observations(self):
        rows = bounded_case_candidates(public_record())
        batch = batch_events([rows, rows, rows])
        self.assertLessEqual(batch['selected_source_events'], 8)
        sources = [(r['episode_alias'], r['card']['source']['event_index']) for r in batch['events']]
        self.assertEqual(len(set(sources)), len(sources))
        self.assertTrue(all(n <= 2 for n in batch['category_counts'].values()))

    def test_redacted_preserves_categories_without_task_time_or_measurement(self):
        rows = [bounded_case_candidates(public_record())]
        full, masked = batch_events(rows), batch_events(rows, redacted=True)
        self.assertEqual(full['category_counts'], masked['category_counts'])
        self.assertEqual(full['selected_source_events'], masked['selected_source_events'])
        self.assertEqual([r['category'] for r in full['events']], [r['category'] for r in masked['events']])
        raw = json.dumps(masked['events'])
        for field in ('thermal:0', 'completed_at_s', 'previous_value', 'event_index', '70.0'):
            self.assertNotIn(field, raw)

    def test_invalid_or_mismatched_measurements_are_rejected(self):
        for change in ('calibration', 'dispatch', 'clock'):
            record = public_record()
            if change == 'calibration':
                record['inspection_events'][0]['calibrated'] = False
            elif change == 'dispatch':
                record['workflow_decisions'].pop()
            else:
                record['inspection_events'][1]['time_s'] = 30.
            with self.assertRaises(ValueError):
                event_candidates(record)

    def test_mean_and_worst_have_distinct_names_and_definitions(self):
        record = public_record()
        record['exact_task_gaps'].append({'task_id': 'thermal:1', 'maximum_gap_including_initial_and_terminal_s': 150.})
        values = scalar_cards(record)
        self.assertEqual(values['thermal']['mean_task_maximum_gap_s'], 120.)
        self.assertEqual(values['thermal']['worst_task_gap_s'], 150.)

    def test_novelty_tool_has_no_anomaly_detection_claim(self):
        self.assertIn('repetition', TOOL_CARDS['novelty'])
        self.assertIn('Does not detect', TOOL_CARDS['novelty'])


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Prompt integration requires neural runtime')
class EventPromptTests(unittest.TestCase):
    def test_only_evidence_representation_changes_and_no_hidden_scores(self):
        from patrol_rsi.improver_task import PatrolProgram
        from patrol_rsi.m12c import diagnostic_prompt, gap_extremes, signal_inactive
        from patrol_rsi.improver_task import TypedRule
        evaluation = {'public': {'coverage': {}, 'professional_evidence': {}, 'efficiency': {}},
            'validator': {'missed_segments': 99, 'hidden_score': 100},
            'event_evidence': batch_events([bounded_case_candidates(public_record())]),
            'gap_extremes': gap_extremes([scalar_cards(public_record())])}
        summary = diagnostic_prompt(PatrolProgram(), evaluation, [], 'summary')
        events = diagnostic_prompt(PatrolProgram(), evaluation, [], 'events')
        redacted = diagnostic_prompt(PatrolProgram(), evaluation, [], 'redacted')
        self.assertNotIn('event_evidence', summary)
        self.assertEqual(events['tool_cards'], summary['tool_cards'])
        self.assertEqual(events['public_diagnostics'], summary['public_diagnostics'])
        self.assertEqual(events['event_evidence']['category_counts'], redacted['event_evidence']['category_counts'])
        for prompt in (summary, events, redacted):
            self.assertNotIn('missed_segments', json.dumps(prompt))
            self.assertNotIn('hidden_score', json.dumps(prompt))
        program = PatrolProgram(rules=(TypedRule('age', .5, 'ppo'), TypedRule('thermal_excess', .25, 'urgency')))
        inactive = signal_inactive(program)
        self.assertEqual(inactive.rules, program.rules[:1])
        self.assertEqual(inactive.fallback, program.fallback)


if __name__ == '__main__':
    unittest.main()
