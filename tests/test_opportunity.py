import copy
from types import SimpleNamespace
import unittest

import numpy as np

from patrol_rsi.improver_task import ImproverTask, PatrolProgram
from patrol_rsi.opportunity_run import run_public, run_evaluator_branch
from patrol_rsi.opportunity_probe import OneDetourProbe, category
from patrol_rsi.public_dispatch import snapshot, choose


def spec():
    return dict(phase='smoke', seed=7699, scenario='coldplate_fouling', duration_s=180,
        fault_start_s=335, noise_c=.2, estimator='nominal', control='feedback', resources='provisioned')


class PublicControllerTests(unittest.TestCase):
    def group(self):
        rows = [{'task': j, 'actor_score': 1. if j == 1 else 0., 'cue_strength': .2 if j == 2 else 0.,
            'known_measured_alarm': False, 'asset_weight_scaled': 1., 'eta_over_deadline': .1} for j in range(3)]
        return [{'agent': 0, 'time_s': 5., 'baseline_next': 0, 'detours_used': 0, 'baseline_completed': 0, 'candidates': rows}]

    def test_controller_uses_only_public_snapshot(self):
        group = self.group()
        self.assertEqual(choose(group, 'tour')[0], {0: 0})
        self.assertEqual(choose(group, 'actor')[0], {0: 1})
        self.assertEqual(choose(group, 'cue_rule')[0], {0: 2})
        changed = copy.deepcopy(group)
        changed[0]['hidden_labels'] = [True, False, True]
        changed[0]['fault_start_s'] = -100
        self.assertEqual(choose(group, 'cue_rule'), choose(changed, 'cue_rule'))
        with self.assertRaises(ValueError):
            choose(group, 'evaluator_one_detour')

    def test_previously_measured_alarm_is_not_rewarded_as_new_cue(self):
        group = self.group()
        group[0]['candidates'][2]['known_measured_alarm'] = True
        self.assertEqual(choose(group, 'cue_rule')[0], {0: 0})

    def test_snapshot_keeps_disjoint_ownership_and_complete_certificate(self):
        env = ImproverTask(spec(), PatrolProgram())
        groups = snapshot(env, lambda f: f[:, 0])
        owner = {}
        for group in groups:
            for row in group['candidates']:
                self.assertNotIn(row['task'], owner)
                owner[row['task']] = group['agent']
                self.assertGreaterEqual(row['minimum_return_slack_s'], 0)
                self.assertGreater(row['candidate_finish_s'], env.clock)
                if row['raw_remote_temperature_c'] is not None:
                    self.assertEqual(row['profession'], 'thermal')
                    self.assertEqual(row['raw_remote_temperature_time_s'], env.clock)


class ProbeBoundaryTests(unittest.TestCase):
    def test_probe_requires_truth_positive_completion_and_preserves_cue_family_scope(self):
        probe = object.__new__(OneDetourProbe)
        probe.at_completion = {(0, 20.): 0}
        probe.rows = [{'witness_count': 0, 'any_cue_witness_count': 0, 'unconfirmed_cue_witness_count': 0,
            'first_witness': None, 'first_unconfirmed_cue_witness': None}]
        probe.scanned_free_agent_frames = probe.scanned_legal_candidates = 0
        env = SimpleNamespace(tasks=[SimpleNamespace(kind='thermal')],
            trace=SimpleNamespace(measurement=lambda task, tick: {'temperature_c': 70.}))
        candidate = {'task': 0, 'candidate_finish_s': 20., 'is_optional': True, 'cue_strength': 0., 'known_measured_alarm': False}
        group = {'agent': 1, 'time_s': 5., 'position_m': [0, 0], 'candidates': [candidate]}
        probe.observe(env, [group])
        self.assertEqual(probe.rows[0]['witness_count'], 1)
        self.assertEqual(category(probe.rows[0]), 'one_detour_without_prespecified_unconfirmed_cue')
        candidate['cue_strength'] = .2
        probe.observe(env, [group])
        self.assertEqual(category(probe.rows[0]), 'one_detour_with_prespecified_unconfirmed_cue')
        candidate['candidate_finish_s'] = 25.
        probe.observe(env, [group])
        self.assertEqual(probe.rows[0]['witness_count'], 2)
        candidate['candidate_finish_s'] = 20.
        env.trace.measurement = lambda task, tick: {'temperature_c': 60.}
        probe.observe(env, [group])
        self.assertEqual(probe.rows[0]['witness_count'], 2)

    def test_empty_probe_is_not_named_an_impossibility_proof(self):
        row = {'witness_count': 0, 'unconfirmed_cue_witness_count': 0}
        self.assertEqual(category(row), 'no_one_detour_witness_on_original_tour_states')


class CounterfactualExecutionTests(unittest.TestCase):
    def test_one_override_executes_and_remains_revisit_feasible(self):
        env = ImproverTask(spec(), PatrolProgram())
        groups = snapshot(env, lambda f: f[:, 0])
        group, candidate = next((g, r) for g in groups for r in g['candidates'] if r['is_optional'])
        intervention = {'dispatch_time_s': 5., 'agent': group['agent'], 'task': candidate['task']}
        record, _ = run_evaluator_branch(spec(), None, lambda f: f[:, 0], intervention)
        altered = [r for r in record['workflow_decisions'] if r['tool'] == 'evaluator_one_detour']
        self.assertEqual(len(altered), 1)
        self.assertEqual(altered[0]['task'], candidate['task'])
        self.assertEqual(record['metrics']['tasks_violating_exact_gap'], 0)
        self.assertIsNone(record['evaluator_probe'])

    def test_probe_observation_does_not_change_public_actions(self):
        without, _ = run_public(spec(), None, lambda f: f[:, 0], 'tour')
        with_probe, _ = run_public(spec(), None, lambda f: f[:, 0], 'tour', probe_enabled=True)
        self.assertEqual(without['action_log'], with_probe['action_log'])
        self.assertEqual(without['inspection_events'], with_probe['inspection_events'])
        self.assertEqual(with_probe['metrics']['selected_detours'], 0)
        self.assertEqual(with_probe['metrics']['tasks_violating_exact_gap'], 0)
        self.assertIsNotNone(with_probe['evaluator_probe'])
        with self.assertRaises(ValueError):
            run_public(spec(), None, lambda f: f[:, 0], 'cue_rule', probe_enabled=True)


if __name__ == '__main__':
    unittest.main()
