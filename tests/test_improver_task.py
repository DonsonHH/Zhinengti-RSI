import json
import unittest

import numpy as np

from patrol_rsi.improver_task import ImproverTask, PatrolProgram, TypedRule, choose_program, run_program, typed_signal
from patrol_rsi.improver_protocol import Improver, program_prompt, validation_rank, validation_summary


def spec(noise=.2, phase=335):
    return dict(phase='smoke', scenario='coldplate_fouling', seed=6990, duration_s=180,
        fault_start_s=phase, noise_c=noise, estimator='nominal', control='feedback', resources='provisioned')


class TypedProgramTests(unittest.TestCase):
    def test_grammar_rejects_unbounded_execution_or_mistyped_thresholds(self):
        value = PatrolProgram().to_dict()
        for change in [{'fallback': 'exec'}, {'python': 'x=1'}, {'refresh_s': True}, {'cooldown_s': 5}]:
            with self.assertRaises(ValueError):
                PatrolProgram.from_dict(value | change)
        with self.assertRaises(ValueError):
            PatrolProgram.from_dict(value | {'rules': [{'signal': 'age', 'threshold': '0.5', 'action': 'ppo'}]})
        self.assertEqual(PatrolProgram.from_dict(value), PatrolProgram())

    def test_temperature_disagreement_never_targets_electrical_flow_or_control(self):
        env = ImproverTask(spec(), PatrolProgram())
        ids = np.arange(len(env.tasks))
        features = np.zeros((len(ids), 16))
        env.current_history[:] = 100
        env.mode = 'history'
        values, valid = typed_signal(env, ids, features, 'thermal_disagreement')
        np.testing.assert_array_equal(valid, [t.kind == 'thermal' for t in env.tasks])
        self.assertTrue((values[~valid] == 0).all())
        self.assertTrue((values[valid] > 0).all())

    def test_actual_electrical_and_flow_evidence_has_own_mask(self):
        env = ImproverTask(spec(), PatrolProgram())
        ids = np.arange(len(env.tasks));f = np.zeros((len(ids), 16))
        env.state.observer.last_contact[:] = 90
        values, valid = typed_signal(env, ids, f, 'contact_heat')
        self.assertTrue((values[valid] == 1).all())
        self.assertTrue(all(env.tasks[j].kind == 'electrical' for j in ids[valid]))
        env.state.observer.last_flow[:] = 0
        values, valid = typed_signal(env, ids, f, 'flow_deficit')
        self.assertTrue((values[valid] == 1).all())
        self.assertTrue(all(env.tasks[j].kind == 'cooling' for j in ids[valid]))

    def test_fault_phase_and_packet_noise_do_not_depend_on_program(self):
        a = ImproverTask(spec(.2, 335), PatrolProgram())
        b = ImproverTask(spec(1., 335), PatrolProgram(fallback='ppo'))
        self.assertIs(a.trace, b.trace)
        self.assertEqual(a.trace.metadata['configuration']['fault_start_s'], 335)
        raw = a.trace.packets[0]['temperature_c']
        np.testing.assert_allclose(b.current_packet['temperature_c'] - raw, 5 * (a.current_packet['temperature_c'] - raw), atol=1e-12)
        c = ImproverTask(spec(.2, 705), PatrolProgram())
        self.assertNotEqual(a.trace.fingerprint, c.trace.fingerprint)

    def test_new_default_expresses_strong_tour_and_no_detours(self):
        record, _ = run_program(spec(), PatrolProgram(), None, lambda f: -f[:, 1])
        self.assertEqual(record['metrics']['selected_detours'], 0)
        self.assertEqual(record['metrics']['tasks_violating_exact_gap'], 0)
        self.assertLessEqual(record['metrics']['maximum_exact_gap_ratio'], .8)
        self.assertTrue(all(r['task'] == r['baseline_next'] for r in record['workflow_decisions']))

    def test_return_rule_overrides_actor_without_changing_validity(self):
        program = PatrolProgram(fallback='ppo', rules=(TypedRule('age', .25, 'return_tour'),))
        env = ImproverTask(spec(), program)
        for _ in range(140):
            choices, logs = choose_program(env, lambda f: -f[:, 1])
            env.step(choices)
            self.assertTrue((env.detours_used <= 2).all())
            for row in logs:
                if row['tool'] == 'return_tour':
                    self.assertEqual(row['task'], row['baseline_next'])

    def test_validation_rank_matches_anomaly_objective_before_travel(self):
        row = {'violating_tasks': 0, 'coverage': 1., 'missed_segments': 5,
            'weighted_censored_delay_s': 70., 'mean_travel_m': 1000.}
        self.assertLess(validation_rank(row), validation_rank(row | {'missed_segments': 6, 'mean_travel_m': 1.}))
        self.assertLess(validation_rank(row), validation_rank(row | {'weighted_censored_delay_s': 90.}))

    def test_protocol_only_sees_public_diagnostics_and_limited_feedback(self):
        views = {'coverage': {'coverage': 1., 'violating_tasks': 0, 'weighted_age_ratio': .4, 'first_alert_tasks': 1},
            'professional_evidence': {'thermal': {'unit': 'degC', 'mean_measured_value': 60}},
            'efficiency': {'extra_checks': 2}}
        message = program_prompt(PatrolProgram(), views, [], Improver())
        self.assertEqual(message['maximum_rule_count'], 3)
        raw = json.dumps(message)
        for name in ('onset_s', 'anomaly_intervals', 'weighted_censored_delay_s', 'missed_segments', '6600', 'test_results'):
            self.assertNotIn(name, raw)
        with self.assertRaises(ValueError):
            Improver.from_dict({'diagnostics': ['truth'], 'memory': 'recent', 'instruction': 'x'})


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Protocol integration requires neural runtime')
class MetaResetTests(unittest.TestCase):
    def test_paired_meta_arms_and_transfer_restart_from_same_task_program(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from patrol_rsi.m12b import adapt, meta_update, parse
        class Caller:
            def generate(self, user, seed):
                if user['task'] == 'propose_improver_protocol':
                    protocol = Improver(instruction='Prefer typed evidence and distinct candidates.').to_dict()
                    return {'raw_response': json.dumps({'diagnosis': 'Try a changed procedure', 'protocol': protocol, 'expected_effect': 'Improve proposal efficiency'})}
                program = PatrolProgram(refresh_s=120).to_dict()
                return {'raw_response': json.dumps({'diagnosis': 'Test tool frequency', 'workflow': program, 'expected_effect': 'Reduce calls'})}
        class Store:
            def __init__(self, output):
                self.output = output
                for name in ('proposals', 'selections', 'meta'):
                    (output / name).mkdir()
            def event(self, *args, **kwargs):
                pass
            def evaluate(self, cases, program):
                return {'validator': {'cases': len(cases), 'anomaly_segments': 1, 'missed_segments': 0,
                    'weighted_censored_delay_s': 10. if program.refresh_s == 120 else 20.,
                    'violating_tasks': 0, 'coverage': 1., 'mean_travel_m': 100., 'scope': 'fixture'},
                    'public': {'coverage': {}, 'professional_evidence': {}, 'efficiency': {}}, 'case_ids': ['fixture']}
        with TemporaryDirectory() as directory:
            store, caller = Store(Path(directory)), Caller()
            protocol, rounds = meta_update(store, caller, 'model_meta', 7101, smoke=True)
            for arm in rounds[0]['arms'].values():
                self.assertEqual(arm['selected']['task_start']['version_id'], PatrolProgram().version_id)
                self.assertEqual(arm['selected']['history_start_length'], 0)
            initial = store.evaluate([], PatrolProgram())
            transfer = adapt(store, caller, protocol, 'transfer', [], 100, 1, initial=initial)
            self.assertEqual(transfer['selected']['task_start']['version_id'], PatrolProgram().version_id)
            self.assertEqual(transfer['selected']['history_start_length'], 0)
            with self.assertRaises(ValueError):
                parse(json.dumps({'diagnosis': 'x', 'protocol': {'python': 'exec'}, 'expected_effect': 'y'}), 'protocol')
            with self.assertRaises(ValueError):
                parse(json.dumps({'diagnosis': ['coverage'], 'protocol': Improver().to_dict(), 'expected_effect': 'y'}), 'protocol')


if __name__ == '__main__':
    unittest.main()
