import copy
import unittest

import numpy as np

from patrol_rsi.facility_workflow import (FacilityWorkflow, Rule, WorkflowInspection,
    public_outcome, rule_match, run_workflow, selection_rank, workflow_choices)


def spec(mode='nominal'):
    return dict(phase='smoke', seed=6200, scenario='coldplate_fouling', duration_s=180,
        estimator=mode, resources='provisioned', variant='open', control='feedback')


class FacilityWorkflowContractTests(unittest.TestCase):
    def test_rejects_extra_execution_fields_and_invalid_types(self):
        value = FacilityWorkflow().to_dict()
        for edit in ({'python': 'print(1)'}, {'refresh_s': True}, {'cooldown_s': 5}, {'rules': 'risk'}):
            with self.assertRaises(ValueError):
                FacilityWorkflow.from_dict(value | edit)
        with self.assertRaises(ValueError):
            FacilityWorkflow.from_dict(value | {'rules': [{'role': 'thermal', 'signal': 'truth', 'threshold': .5, 'ranker': 'risk'}]})

    def test_roles_and_thresholds_only_filter_observable_features(self):
        features = np.zeros((3, 16))
        features[:, 2] = [.1, .3, .8]
        rule = Rule('thermal', 'risk', .25, 'risk')
        np.testing.assert_array_equal(rule_match(rule, features, 'thermal'), [False, True, True])
        self.assertFalse(rule_match(rule, features, 'cooling').any())
        program = FacilityWorkflow(rules=(rule,))
        self.assertEqual(FacilityWorkflow.from_dict(program.to_dict()), program)

    def test_repeated_measured_alarms_do_not_increase_discovery_score(self):
        record = {'inspection_events': [{'task': 0, 'time_s': 30., 'alarm': True, 'measurement_transition': 'new_measured_alert'}],
            'workflow_contract': {'deadlines_s': [300, 600]},
            'metrics': {'tasks_violating_exact_gap': 0, 'final_professional_coverage': 1.,
                'travel_m': 10., 'service_s': 5., 'estimator_calls': 0, 'selected_detours': 0},
            'assignments': [{}], 'workflow_decisions': []}
        before = public_outcome(record, [2., 1.])
        record['inspection_events'].append({'task': 0, 'time_s': 35., 'alarm': True, 'measurement_transition': 'persistent_recheck'})
        record['hidden_anomaly_intervals'] = 'not available to the selection interface'
        after = public_outcome(record, [2., 1.])
        self.assertEqual(before['first_alert_task_weight'], after['first_alert_task_weight'])
        self.assertEqual(after['measurement_transitions']['persistent_recheck'], 1)
        self.assertNotEqual(before['weighted_age_ratio'], after['weighted_age_ratio'])
        changed = copy.deepcopy(record)
        changed['hidden_anomaly_intervals'] = 'arbitrary alternative evaluator labels'
        self.assertEqual(public_outcome(changed, [2., 1.]), after)

    def test_selection_prioritizes_coverage_before_alerts(self):
        base = dict(violating_tasks=0, coverage=1, first_alert_task_weight=1, weighted_age_ratio=.4, travel_m=10)
        self.assertLess(selection_rank(base), selection_rank(base | {'violating_tasks': 1, 'first_alert_task_weight': 100}))
        self.assertLess(selection_rank(base | {'first_alert_task_weight': 2}), selection_rank(base))


class FacilityWorkflowExecutionTests(unittest.TestCase):
    def test_cooldown_never_blocks_mandatory_baseline(self):
        env = WorkflowInspection(spec(), FacilityWorkflow(cooldown_s=120))
        baseline = env.plans[0].tour[0]
        env.last_inspection[baseline] = env.clock
        env.detours_used[0] = 2
        ids, _, _ = env.options(0, set())
        self.assertEqual(ids.tolist(), [baseline])

    def test_exhausted_quota_removes_extra_choices(self):
        env = WorkflowInspection(spec(), FacilityWorkflow(cooldown_s=0))
        env.detours_used[:] = 2
        for i in range(len(env.agents)):
            ids, _, _ = env.options(i, set())
            self.assertEqual(ids.tolist(), [env.plans[i].tour[0]])

    def test_estimator_tool_period_changes_real_calls_but_not_frame_clock(self):
        for period in (30, 60, 120):
            calls = []
            def predict(context):
                calls.append(context.copy())
                return np.full(128, 50.)
            env = WorkflowInspection(spec('history'), FacilityWorkflow(refresh_s=period), predict)
            for _ in range(24):
                choices, _ = workflow_choices(env, lambda f: f[:, 0])
                env.step(choices)
            self.assertEqual(env.clock, 125)
            self.assertEqual(env.estimator_calls, 120 // period)
            self.assertEqual(len(calls), 120 // period)
            self.assertEqual(len(env.frames), 4)

    def test_real_dispatch_quota_resets_only_on_complete_baseline_round(self):
        env = WorkflowInspection(spec(), FacilityWorkflow(cooldown_s=0))
        for _ in range(120):
            choices, _ = workflow_choices(env, lambda f: -f[:, 1])
            env.step(choices)
            self.assertTrue((env.detours_used <= 2).all())
        completed = np.zeros(len(env.agents), dtype=int)
        extras = np.zeros(len(env.agents), dtype=int)
        pending = {}
        # Independently merge dispatch and completion clocks.
        for t in range(5, 611, 5):
            for event in env.events:
                if event['time_s'] == t:
                    i = event['agent']
                    row = pending[i]
                    if event['task'] == row['baseline_next']:
                        completed[i] += 1
                        if completed[i] == len(env.plans[i].tour):
                            completed[i] = extras[i] = 0
            for row in env.selected_certificates:
                if row['clock_s'] == t:
                    i = row['agent']
                    pending[i] = row
                    extras[i] += row['task'] != row['baseline_next']
                    self.assertLessEqual(extras[i], 2)

    def test_full_episode_preserves_exact_revisit_including_tail(self):
        record, env = run_workflow(spec(), FacilityWorkflow(rules=(Rule('all', 'age', .25, 'urgency'),)),
            None, lambda f: f[:, 0])
        self.assertEqual(record['metrics']['tasks_violating_exact_gap'], 0)
        self.assertEqual(record['metrics']['final_professional_coverage'], 1.)
        self.assertEqual(len(record['assignments']), len(record['workflow_decisions']))
        self.assertGreater(record['public_selection']['rule_dispatches'], 0)

    def test_frozen_cycle_ignores_workflow_ranking(self):
        record, _ = run_workflow(spec(), FacilityWorkflow(), None, lambda f: -f[:, 0], control='cycle')
        self.assertEqual(record['metrics']['selected_detours'], 0)
        self.assertTrue(all(r['task'] == r['baseline_next'] for r in record['workflow_decisions']))


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Full integration schema test requires the neural runtime')
class FacilityProposalTests(unittest.TestCase):
    def test_prompt_excludes_hidden_record_and_parser_does_not_repair(self):
        from patrol_rsi.m12 import parse_proposal, prompt
        import json
        public = dict(violating_tasks=0, coverage=1, first_alert_task_weight=1, weighted_age_ratio=.4, travel_m=10)
        message = prompt(FacilityWorkflow(), public, [], memory=True)
        self.assertEqual(message['grammar']['rule_keys'], ['role', 'signal', 'threshold', 'ranker'])
        self.assertIn('threshold', message['grammar']['rule_domains'])
        raw = json.dumps(message)
        for forbidden in ('anomaly_intervals', 'onset_s', 'first_confirmation_s', 'scenario', '6300', '6301'):
            self.assertNotIn(forbidden, raw)
        invalid = json.dumps({'diagnosis': 'x', 'workflow': FacilityWorkflow().to_dict() | {'refresh_s': 45}, 'expected_effect': 'y'})
        with self.assertRaises(ValueError):
            parse_proposal(invalid)


if __name__ == '__main__':
    unittest.main()
