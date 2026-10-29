import copy
import gzip
import json
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from patrol_rsi.opportunity_data import (ACTION_FEATURE_NAMES, FEATURE_NAMES, FeatureHistory,
    action_features, completion_lookup, contract, evaluator_labels, legal_completions, sample_targets)


class OpportunityDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'research/2026-10-29-m12c1b32-formal/evidence/explanatory-cases/development-positive.json.gz'
        cls.record = json.loads(gzip.decompress(path.read_bytes()))
        cls.bundle = cls.record['selected_public_snapshots'][0]

    def test_features_keep_qualified_history_and_cvae_channels_separate(self):
        first = copy.deepcopy(self.bundle)
        extractor = FeatureHistory()
        values = extractor.append(first)
        self.assertEqual(values.shape, (176, len(FEATURE_NAMES)))
        self.assertEqual(values[148, FEATURE_NAMES.index('flow.value')], first['tasks'][148]['flow']['value'])
        self.assertEqual(values[0, FEATURE_NAMES.index('cvae_raw.value')], np.float32(first['tasks'][0]['cvae_raw']['value']))
        self.assertEqual(values[0, FEATURE_NAMES.index('history_assimilated.value')], np.float32(first['tasks'][0]['history_assimilated']['value']))
        self.assertNotIn('seed', FEATURE_NAMES)
        self.assertNotIn('scenario', FEATURE_NAMES)
        self.assertEqual(values[0, FEATURE_NAMES.index('qualified_history_0.time_s')], first['tasks'][0]['qualified']['time_s'])

    def test_future_channel_or_evaluator_metadata_is_rejected(self):
        bundle = copy.deepcopy(self.bundle)
        bundle['tasks'][0]['cvae_raw']['context_cutoff_s'] = 755.
        with self.assertRaises(ValueError):
            FeatureHistory().append(bundle)
        bundle = copy.deepcopy(self.bundle)
        bundle['future_truth'] = [100.]
        with self.assertRaises(ValueError):
            FeatureHistory().append(bundle)

    def test_legal_lookup_uses_exact_completion_without_outcome_selection(self):
        rows = [{'task': 148, 'agent': 14, 'dispatch_s': 750., 'finish_s': 795., 'optional': True},
                {'task': 148, 'agent': 14, 'dispatch_s': 760., 'finish_s': 795., 'optional': True}]
        selected, count = legal_completions(735., 60, rows)
        self.assertEqual(selected[148]['dispatch_s'], 750.)
        self.assertEqual(count[148], 2)
        self.assertIsNone(selected[147])
        absent, count = legal_completions(755., 30, rows)
        self.assertTrue(all(v is None for v in absent))

    def test_A_B_validity_and_retained_event_are_different(self):
        arrays = {'rack_temperature_c': np.full((360, 128), 50.),
                  'contact_temperature_c': np.full((360, 16), 40.),
                  'rack_flow_kg_s': np.full((360, 128), 4.5),
                  'public_sensor_temperature_c': np.full((360, 128), 50.)}
        arrays['rack_flow_kg_s'][11:14, :8] = 2.1
        arrays['contact_temperature_c'][11, 0] = 70.
        labels, events = evaluator_labels(arrays, 8500)
        selected = [None] * 176
        selected[144] = {'agent': 14}
        a, reference, b = sample_targets(labels, 30, 30, selected)
        self.assertEqual((a[144], reference[144], b[144]), (1, 1, 1))
        self.assertEqual((a[128], reference[128], b[128]), (1, 1, -1))
        self.assertEqual(labels['retained_event_id'][11, 128], 0)
        self.assertGreater(labels['retained_event_id'][11, 144], 0)
        a, reference, b = sample_targets(labels, 1795, 30, selected)
        self.assertTrue((a == -1).all() and (reference == -1).all() and (b == -1).all())

    def test_publicly_known_inflight_Tour_completion_is_a_valid_B_opportunity(self):
        bundle = copy.deepcopy(self.bundle)
        agent = bundle['agents'][12]
        agent.update(busy_target=132, busy_finish_s=780.)
        rows = [{'task': 132, 'agent': 12, 'dispatch_s': 745., 'finish_s': 780.,
                 'optional': False, 'tour_finish_s': 780., 'return_slack_s': 0.},
                {'task': 148, 'agent': 14, 'dispatch_s': 745., 'finish_s': 780.,
                 'optional': True, 'tour_finish_s': 990., 'return_slack_s': 225.}]
        selected, count = legal_completions(750., 30., rows, bundle)
        self.assertIsNotNone(selected[132])
        self.assertIsNone(selected[148])
        values = action_features(FeatureHistory().append(bundle), 750., 30., selected, count)
        self.assertEqual(values[132, ACTION_FEATURE_NAMES.index('inflight_protected_Tour')], 1.)
        self.assertEqual(values[132, ACTION_FEATURE_NAMES.index('planned_dispatch_offset_s')], -5.)

    def test_closed_action_features_do_not_accept_evaluator_labels(self):
        base = FeatureHistory().append(self.bundle)
        rows = [None] * 176
        rows[148] = {'optional': True, 'dispatch_s': 750., 'finish_s': 810.,
                     'tour_finish_s': 990., 'return_slack_s': 225.}
        values = action_features(base, 750., 60., rows, np.ones(176, dtype=np.int16))
        self.assertEqual(values.shape, (176, len(ACTION_FEATURE_NAMES)))
        self.assertEqual(values[148, ACTION_FEATURE_NAMES.index('complete_return_slack_s')], 225.)
        self.assertEqual(values[148, ACTION_FEATURE_NAMES.index('horizon_s')], 60.)
        self.assertFalse(any('oracle' in name or 'anomaly' in name for name in FEATURE_NAMES + ACTION_FEATURE_NAMES))
        with self.assertRaises(ValueError):
            action_features(base, 735., 60., rows, np.ones(176, dtype=np.int16))

    def test_coverage_cell_keeps_horizon_metadata_and_deduplicates_events(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('opportunity_report_for_test', root / 'scripts/report_opportunity_data.py')
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
        cell = report.Coverage()
        row = {'kind': 'thermal', 'horizon_s': 30, 'total_samples': 10,
            'A_valid': 10, 'A_positive': 2, 'A_negative': 8,
            'B_reference_valid': 10, 'B_reference_positive': 2,
            'B_action_valid': 5, 'B_action_positive': 1, 'B_action_negative': 4,
            'B_action_invalid': 5, 'physical_event_count': 1, 'retained_event_count': 1,
            'B_positive_event_ids': [1], 'B_positive_retained_event_ids': [1],
            'availability_in_valid_B_samples': {}}
        condition = {'condition': {'seed': 8500}, 'numeric_process_fingerprint': 'same-physical-trace'}
        events = {1: {'task': 0, 'kind': 'thermal', 'first_tick': 2, 'last_exclusive_tick': 5}}
        cell.add(row, condition, events)
        cell.add(row, condition, events)
        values = cell.value()
        self.assertEqual(values['horizon_s'], 30)
        self.assertEqual(values['total_samples'], 20)
        self.assertEqual(values['B_reference_negative'], 16)
        self.assertEqual(values['distinct_B_positive_retained_events'], 1)
        self.assertEqual(values['numeric_processes'], 1)


if __name__ == '__main__':
    unittest.main()
