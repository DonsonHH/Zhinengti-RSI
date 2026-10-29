import gzip
import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path
import unittest

from patrol_rsi.aisles import AisleGraph, build_agents
from patrol_rsi.coverage_plan import ReturnPlan
from patrol_rsi.facility import FacilityConfig
from patrol_rsi.queue_diagnostics import diagnose_queue
from patrol_rsi.queue_history import history_state


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'research/2026-10-29-m12c1b2-confirm/evidence'


def saved_case(phase=335, method='lookahead', seed=8301):
    directory = OLD / ('failure-examples' if seed == 8301 else 'demo')
    path = next(directory.glob(f'*--{seed}--{phase}--0.2--history--{method}.json.gz'))
    return json.loads(gzip.decompress(path.read_bytes()))


def public_example(phase=335):
    record = saved_case(phase)
    action = next(r for r in record['workflow_decisions'] if r['task'] != r['baseline_next'])
    graph = AisleGraph(FacilityConfig(racks=128), 'open')
    agent = build_agents(graph, 'provisioned')[action['agent']]
    deadlines = record['workflow_contract']['deadlines_s']
    plan = ReturnPlan(graph, graph.tasks, agent['tour'], deadlines)
    now = action['time_s']
    state = history_state(plan, now=now, agent=agent,
        assignments=[r for r in record['assignments'] if r['time_s'] < now],
        events=[r for r in record['inspection_events'] if r['time_s'] <= now],
        weights=record['workflow_contract']['task_weights'], global_used=False)
    return plan, state


class QueueDiagnosticsTests(unittest.TestCase):
    def test_public_schedule_predicts_saved_30_second_collateral_delay(self):
        plan, state = public_example()
        result = diagnose_queue(plan, state, 108)
        self.assertTrue(result['legal'])
        self.assertEqual(result['target_tour_completion_s'], 520.)
        self.assertEqual(result['target_detour_completion_s'], 490.)
        self.assertEqual(result['target_lead_s'], 30.)
        row = next(r for r in result['affected_tasks'] if r['task'] == 110)
        self.assertEqual(row['tour_completion_s'], 535.)
        self.assertEqual(row['detour_tour_completion_s'], 565.)
        self.assertEqual(row['delay_s'], 30.)
        self.assertEqual(result['detour_certificate']['minimum_projected_slack_s'], 125.)

    def test_legal_detour_has_a_separate_uncertain_public_cost(self):
        from patrol_rsi.queue_diagnostics import collateral_proxy
        plan, state = public_example()
        schedule = diagnose_queue(plan, state, 108)
        proxy = collateral_proxy(plan, state, schedule)
        self.assertTrue(schedule['legal'])
        self.assertGreater(proxy['collateral_cost'], 0.)
        self.assertEqual(proxy['view'], 'public_proxy_not_future_loss')
        row = next(r for r in proxy['task_costs'] if r['task'] == 110)
        self.assertEqual(row['measurement_age_s'], 75.)
        self.assertEqual(row['last_qualified_measurement_s'], 390.)
        self.assertNotIn('anomaly_intervals', proxy)

    def test_public_boundary_rejects_future_measurement_even_in_direct_state(self):
        plan, state = public_example()
        entry = next(r for r in state.evidence if r.task == 110)
        future = replace(entry, latest=replace(entry.latest, time_s=state.time_s + 5))
        changed = replace(state, evidence=tuple(future if r.task == 110 else r for r in state.evidence))
        with self.assertRaises(ValueError):
            diagnose_queue(plan, changed, 108)

    def test_offline_oracle_reproduces_two_saved_regressions(self):
        from patrol_rsi.queue_oracle import offline_oracle
        for phase, lost in ((335, ['thermal:110']), (520, ['thermal:104', 'thermal:106'])):
            with self.subTest(phase=phase):
                plan, state = public_example(phase)
                diagnostic = diagnose_queue(plan, state, 108)
                result = offline_oracle(diagnostic, saved_case(phase, 'tour'), saved_case(phase))
                self.assertEqual([r['task_id'] for r in result['lost_confirmations']], lost)
                self.assertEqual(result['actual_missed_delta'], len(lost))
                self.assertEqual(result['window_opportunity_missed_delta'], len(lost))
                self.assertEqual(result['target_confirmation_lead_s'], 30.)
                self.assertEqual(result['view'], 'offline_evaluator_only')

    def test_whole_remaining_queue_matches_actual_measurement_completions(self):
        for phase in (335, 520):
            plan, state = public_example(phase)
            diagnostic = diagnose_queue(plan, state, 108)
            for method, key in (('tour', 'tour_plan'), ('lookahead', 'detour_plan')):
                events = saved_case(phase, method)['inspection_events']
                actual = [(r['task'], r['time_s']) for r in events
                    if r['agent'] == state.agent and r['time_s'] > state.time_s]
                predicted = [(r['task'], r['time_s']) for r in diagnostic[key]['mission_completions']]
                self.assertEqual(predicted, actual)

    def test_qualification_ownership_cooldown_and_budget_are_not_proxy_overrides(self):
        plan, state = public_example()
        self.assertEqual(diagnose_queue(plan, replace(state, role='cooling'), 108)['reason'],
            'qualification_or_access_mismatch')
        self.assertEqual(diagnose_queue(plan, state, 111)['reason'], 'target_outside_owned_tour')
        self.assertFalse(diagnose_queue(plan, replace(state, global_used=True), 108)['legal'])
        self.assertFalse(diagnose_queue(plan, replace(state, detours_used=2), 108)['legal'])
        latest = list(state.last_checks_s);latest[108] = state.time_s - 5
        evidence = tuple(r for r in state.evidence if r.task != 108)
        self.assertEqual(diagnose_queue(plan, replace(state, last_checks_s=tuple(latest), evidence=evidence), 108)['reason'],
            'extra_check_cooldown')

    def test_expired_whole_return_and_off_map_positions_are_rejected(self):
        plan, state = public_example()
        latest = list(state.last_checks_s);latest[110] = 0.
        evidence = tuple(r for r in state.evidence if r.task != 110)
        self.assertEqual(diagnose_queue(plan, replace(state, last_checks_s=tuple(latest), evidence=evidence), 108)['reason'],
            'whole_return_plan_infeasible')
        with self.assertRaises(ValueError):
            diagnose_queue(plan, replace(state, position_m=(999., 999.)), 108)

    def test_typed_boundary_does_not_accept_raw_record_or_hidden_fields(self):
        plan, state = public_example()
        with self.assertRaises(TypeError):
            diagnose_queue(plan, saved_case(), 108)
        with self.assertRaises(TypeError):
            replace(state, future_labels=[True], scenario='aux_feeder_loss+cdu_fouling')
        before = asdict(state)
        diagnose_queue(plan, state, 108)
        self.assertEqual(before, asdict(state))

    def test_history_adapter_rejects_future_and_busy_or_uncalibrated_input(self):
        plan, state = public_example()
        record = saved_case();agent = build_agents(plan.graph, 'provisioned')[10]
        kwargs = dict(now=465., agent=agent,
            assignments=[r for r in record['assignments'] if r['time_s'] < 465],
            events=[r for r in record['inspection_events'] if r['time_s'] <= 465],
            weights=state.task_weights, global_used=False)
        future = next(r for r in record['inspection_events'] if r['time_s'] > 465)
        with self.assertRaises(ValueError):
            history_state(plan, **dict(kwargs, events=kwargs['events'] + [future]))
        event = dict(kwargs['events'][0], calibrated=False)
        with self.assertRaises(ValueError):
            history_state(plan, **dict(kwargs, events=[event]))
        with self.assertRaises(ValueError):
            history_state(plan, **dict(kwargs, now=460., events=[r for r in kwargs['events'] if r['time_s'] <= 460]))

    def test_public_cooling_evidence_can_affect_thermal_tasks_in_shared_pod(self):
        from patrol_rsi.queue_diagnostics import Measurement, TaskEvidence, collateral_proxy
        plan, state = public_example()
        # Isolate the added shared-branch signal from other recorded alarms.
        state = replace(state, evidence=tuple(r for r in state.evidence if r.task == 110))
        cooling = next(j for j,t in enumerate(plan.tasks) if t.kind == 'cooling' and t.index == 13)
        old = collateral_proxy(plan, state, diagnose_queue(plan, state, 108))
        latest = list(state.last_checks_s);latest[cooling] = state.time_s
        reading = TaskEvidence(cooling, Measurement(state.time_s, 0., 'kg/s'), alarm=True)
        changed = replace(state, last_checks_s=tuple(latest),
            evidence=tuple(r for r in state.evidence if r.task != cooling) + (reading,))
        new = collateral_proxy(plan, changed, diagnose_queue(plan, changed, 108))
        before = next(r for r in old['task_costs'] if r['task'] == 110)
        after = next(r for r in new['task_costs'] if r['task'] == 110)
        self.assertIn(cooling, after['linked_public_tasks'])
        self.assertGreater(after['estimated_loss_increment'], before['estimated_loss_increment'])

    def test_persistent_queue_delay_is_accounted_through_mission_end(self):
        from patrol_rsi.queue_diagnostics import collateral_proxy
        plan, state = public_example()
        diagnostic = diagnose_queue(plan, state, 108)
        row = next(r for r in diagnostic['affected_tasks'] if r['task'] == 110)
        # Saved completions 535,680,...,1695 all shift by 30s: nine visits.
        self.assertEqual(row['mission_delay_exposure_s'], 270.)
        self.assertEqual(len(row['delayed_normal_visits']), 9)
        cost = next(r for r in collateral_proxy(plan, state, diagnostic)['task_costs'] if r['task'] == 110)
        self.assertGreater(cost['estimated_loss_increment'], cost['first_round_loss_increment'])

    def test_mission_tail_keeps_the_original_baseline_can_finish_gate(self):
        plan, state = public_example()
        tail = replace(state, time_s=1790., position_m=plan.tasks[108].position_m,
            last_checks_s=tuple(1700. for _ in plan.tasks), evidence=())
        self.assertEqual(diagnose_queue(plan, tail, 108)['reason'], 'baseline_finishes_after_mission')

    def test_verifier_accepts_recorded_windows_and_posix_source_paths(self):
        spec = importlib.util.spec_from_file_location('queue_verify', ROOT / 'scripts/verify_queue_diagnosis.py')
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for value in (r'research\demo\case.json.gz', '/tmp/research/demo/case.json.gz'):
            self.assertEqual(module.source_basename(value), 'case.json.gz')

    def test_zero_duration_custom_plan_is_rejected_before_projection(self):
        plan, state = public_example()
        task = replace(plan.tasks[0], service_s=0.)
        empty = ReturnPlan(plan.graph, (task, replace(task, task_id='thermal:other')), (0,1), (300.,300.))
        tiny = replace(state, time_s=5., cursor=0, position_m=task.position_m,
            last_checks_s=(-1.,-1.), task_weights=(1.,1.), evidence=())
        with self.assertRaises(ValueError):
            diagnose_queue(empty, tiny, 1)


if __name__ == '__main__':
    unittest.main()
