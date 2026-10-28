import copy
import unittest

import numpy as np

from patrol_rsi.improver_task import ImproverTask, PatrolProgram
from patrol_rsi.public_dispatch import snapshot
from patrol_rsi.inspection_value import FEATURE_NAMES, feature_rows, choose
from patrol_rsi.value_dataset import Collector, labels
from patrol_rsi.value_run import run
from patrol_rsi.value_learning import fit, scorer, ridge, identity


def spec():
    return dict(phase='smoke', seed=7899, scenario='coldplate_fouling', duration_s=180,
        fault_start_s=335, noise_c=.2, estimator='nominal', control='feedback', resources='provisioned')


class ValueSkillTests(unittest.TestCase):
    def groups(self):
        env = ImproverTask(spec(), PatrolProgram())
        return snapshot(env, lambda f: f[:, 0])

    def test_schema_hidden_metadata_and_input_immutability(self):
        groups = self.groups();original = copy.deepcopy(groups)
        rows = feature_rows(groups)
        self.assertEqual(len(FEATURE_NAMES), 34)
        self.assertTrue(all(x.shape == (34,) for _, _, x in rows))
        altered = copy.deepcopy(groups)
        for g in altered:
            g.update(scenario='secret', seed=1234, future_labels=[1, 0])
            for r in g['candidates']:
                r.update(future_temperature_c=999, evaluator_label=True)
        for a, b in zip(rows, feature_rows(altered)):
            np.testing.assert_array_equal(a[2], b[2])
        self.assertEqual(choose(groups, 'net', score=lambda x: np.ones(len(x))),
            choose(altered, 'net', score=lambda x: np.ones(len(x))))
        self.assertEqual(groups, original)

    def test_global_budget_off_threshold_and_rejection(self):
        groups = self.groups()
        baseline = choose(groups, 'tour')[0]
        choices, _, consumed = choose(groups, 'net', score=lambda x: np.ones(len(x)))
        self.assertTrue(consumed)
        self.assertEqual(sum(choices[i] != baseline[i] for i in choices), 1)
        self.assertEqual(choose(groups, 'net', used=True)[0], baseline)
        self.assertEqual(choose(groups, 'net', threshold='off')[0], baseline)
        self.assertFalse(choose(groups, 'net', score=lambda x: np.zeros(len(x)))[2])
        with self.assertRaises(ValueError):
            choose(groups, 'net', score=lambda x: np.full(len(x), np.nan))
        with self.assertRaises(ValueError):
            choose(groups, 'oracle')

    def test_full_episode_one_optional_and_exact_revisit(self):
        record, _ = run(spec(), None, lambda f: f[:, 0], 'net', score=lambda x: np.ones(len(x)))
        self.assertEqual(record['metrics']['selected_detours'], 1)
        self.assertEqual(record['metrics']['tasks_violating_exact_gap'], 0)
        self.assertEqual(sum(r['tool'] == 'one_net' for r in record['workflow_decisions']), 1)
        with self.assertRaises(ValueError):
            run(spec(), None, lambda f: f[:, 0], 'net', observer=Collector())

    def test_observer_is_passive_and_natural_sample_has_no_teacher_at_test(self):
        first, _ = run(spec(), None, lambda f: f[:, 0])
        collector = Collector()
        observed, _ = run(spec(), None, lambda f: f[:, 0], observer=collector, probe_enabled=True)
        self.assertEqual(first['action_log'], observed['action_log'])
        self.assertEqual(first['inspection_events'], observed['inspection_events'])
        self.assertEqual(sum(collector.confusion.values()), len(collector.public))
        self.assertEqual(len(collector.quality), 59)
        sample = collector.sample(observed, test=True)
        self.assertLessEqual(len(sample), 4)
        self.assertTrue(all(p.startswith('natural_') for r in sample for p in r['provenance']))
        self.assertEqual(sample, collector.sample(observed, test=True))

    def test_target_reward_can_be_positive_when_total_misses_worsen(self):
        def record(flags, delays, travel):
            intervals = [dict(task=j, onset_s=10, end_s=100, weight=1., detected=flag, delay_or_censor_s=delay)
                for j, (flag, delay) in enumerate(zip(flags, delays))]
            return dict(anomaly_intervals=intervals, metrics=dict(missed_segments=flags.count(False),
                consequence_weighted_censored_delay_s=sum(delays)/3, travel_m=travel, service_s=20.))
        values = labels(record([False, True, True], [90, 20, 20], 100), record([True, False, False], [10, 90, 90], 120), 0)
        self.assertGreater(values['target'], 0)
        self.assertLess(values['net'], 0)
        self.assertEqual(values['collateral_lost_segments'], 2)
        self.assertEqual(values['missed_delta'], 1)

    def test_no_event_labels_are_defined_without_fabricated_delay(self):
        base = dict(anomaly_intervals=[], metrics=dict(missed_segments=0, consequence_weighted_censored_delay_s=None,
            travel_m=10., service_s=20.))
        branch = copy.deepcopy(base);branch['metrics']['travel_m'] += 5
        values = labels(base, branch, 0)
        self.assertEqual(values['target'], 0)
        self.assertAlmostEqual(values['net'], -5/100000)

    def test_model_freeze_roundtrip_and_paired_initialization(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Training verified on scheduled torch runtime')
        torch.set_num_threads(1)
        rows = [dict(x=[float(j)/10]*34, labels=dict(net=float(j % 2), target=0.)) for j in range(8)]
        packs, log = fit(rows, 0, updates=5)
        repeat, _ = fit(rows, 0, updates=5)
        self.assertEqual(identity(packs['untrained']), identity(repeat['untrained']))
        self.assertEqual(identity(packs['net']), identity(repeat['net']))
        self.assertNotEqual(packs['net']['state'], packs['target']['state'])
        self.assertEqual(log['net']['parameters'], 2209)
        for pack in list(packs.values()) + [ridge(rows)]:
            self.assertTrue(np.isfinite(scorer(pack)(np.zeros((2, 34), dtype=np.float32))).all())


if __name__ == '__main__':
    unittest.main()
