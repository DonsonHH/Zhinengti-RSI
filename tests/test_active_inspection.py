import copy
import heapq
import unittest
from unittest.mock import patch

import numpy as np

from patrol_rsi.aisles import AisleGraph, build_agents, feasibility
from patrol_rsi.active_inspection import ActiveInspection, heuristic_decisions
from patrol_rsi.facility import FacilityConfig
from patrol_rsi.inspection_metrics import anomaly_intervals, evaluate_episode

try:
    import torch
    from patrol_rsi.inspection_model import TaskPolicy, policy_decisions, ppo_update
except ImportError:
    torch = None


def specification(**extra):
    return dict(racks=128, seed=9011, scenario='healthy', control='feedback',
                resources='provisioned', variant='open', estimator='nominal', **extra)


class AisleTests(unittest.TestCase):
    def test_rack_interior_cannot_be_shortcut(self):
        graph = AisleGraph(FacilityConfig(racks=8))
        distance, route = graph.path((0., -1.), (0., 2.5))
        self.assertAlmostEqual(distance, 5.5)
        for before, after in zip(route, route[1:]):
            self.assertIn(after, graph.adjacency[before])
        with self.assertRaises(ValueError):
            graph.path((0., 0.), (0., -1.))

    def test_astar_matches_independent_dijkstra(self):
        graph = AisleGraph(FacilityConfig(racks=32), 'restricted')
        source, target = graph.tasks[3].position_m, graph.tasks[30].position_m
        queue, costs = [(0., source)], {source: 0.}
        while queue:
            cost, node = heapq.heappop(queue)
            if cost > costs[node]:
                continue
            if node == target:
                break
            for other, length in graph.adjacency[node].items():
                candidate = cost + length
                if candidate < costs.get(other, float('inf')):
                    costs[other] = candidate
                    heapq.heappush(queue, (candidate, other))
        self.assertAlmostEqual(graph.path(source, target)[0], costs[target])

    def test_resource_lower_bound_and_nonempty_partition(self):
        graph = AisleGraph(FacilityConfig())
        scarce, provisioned = feasibility(graph, 'scarce'), feasibility(graph, 'provisioned')
        self.assertGreater(scarce['roles']['thermal']['service_only_utilization'], 1)
        self.assertFalse(scarce['roles']['thermal']['service_only_necessary_condition'])
        self.assertEqual(provisioned['roles']['thermal']['service_only_minimum_agents'], 5)
        agents = build_agents(graph, 'provisioned')
        self.assertEqual(len(agents), 18)
        own = [task for agent in agents for task in agent['tour']]
        self.assertEqual(sorted(own), list(range(len(graph.tasks))))
        self.assertTrue(all(v['all_constructed_cycles_within_revisit'] for v in provisioned['roles'].values()))

    def test_spatial_rank_keeps_consecutive_pods_adjacent_at_row_ends(self):
        graph = AisleGraph(FacilityConfig())
        pods = sorted(range(graph.config.pods), key=graph.spatial_rank)
        self.assertEqual([graph.spatial_rank(p) for p in pods], list(range(graph.config.pods)))
        for a, b in zip(pods, pods[1:]):
            ax, ay = a % graph.columns, a // graph.columns
            bx, by = b % graph.columns, b // graph.columns
            self.assertEqual(abs(ax - bx) + abs(ay - by), 1)


class QualifiedClockTests(unittest.TestCase):
    def test_no_measurement_before_travel_and_service_finish(self):
        env = ActiveInspection(specification())
        first = env.agents[0]['tour'][0]
        env.step(heuristic_decisions(env, 'cycle'))
        self.assertFalse(env.ever[first])
        env.step(heuristic_decisions(env, 'cycle'))
        self.assertTrue(env.ever[first])
        event = next(e for e in env.events if e['task'] == first)
        self.assertEqual(event['time_s'], 15)
        self.assertEqual(event['role'], 'thermal')

    def test_duplicate_and_ineligible_choices_are_atomic(self):
        env = ActiveInspection(specification())
        all_free = {agent['id']: None for agent in env.agents}
        first = env.agents[0]['allowed'][0]
        all_free[0] = all_free[1] = first
        before = copy.deepcopy(env.agents)
        with self.assertRaises(ValueError):
            env.step(all_free)
        self.assertEqual(env.agents, before)
        self.assertEqual(env.clock, 5)
        electrical = next(i for i, task in enumerate(env.tasks) if task.kind == 'electrical')
        all_free[0], all_free[1] = electrical, None
        with self.assertRaises(ValueError):
            env.step(all_free)
        self.assertEqual(env.agents, before)

    def test_busy_inspector_cannot_be_reassigned(self):
        env = ActiveInspection(specification())
        env.step(heuristic_decisions(env, 'cycle'))
        with self.assertRaises(ValueError):
            env.step({a['id']: a['tour'][0] for a in env.agents})

    def test_reward_features_ignore_unmeasured_true_labels(self):
        env = ActiveInspection(specification())
        ids, first, mask = env.options(0, set())
        value = env.value_features()
        old = env.trace.arrays['rack_temperature_c'].copy()
        try:
            env.trace.arrays['rack_temperature_c'][:] = -900
            np.testing.assert_equal(first, env.options(0, set())[1])
            np.testing.assert_equal(value, env.value_features())
        finally:
            env.trace.arrays['rack_temperature_c'][:] = old

    def test_guard_only_selects_legal_urgent_tasks(self):
        env = ActiveInspection(specification())
        env.clock = 500
        ids, features, allowed = env.options(0, {env.agents[0]['allowed'][0]}, guard=True)
        self.assertEqual(allowed.sum(), 1)
        self.assertNotIn(env.agents[0]['allowed'][0], ids)
        self.assertEqual(int(np.argmax(allowed)), int(np.argmax(features[:, 0] + features[:, 1])))

    def test_complete_tour_conserves_time_and_does_not_repair_plant(self):
        spec = specification()
        env = ActiveInspection(spec)
        fingerprint = env.trace.fingerprint
        while not env.done:
            env.step(heuristic_decisions(env, 'cycle'))
        self.assertEqual(env.trace.fingerprint, fingerprint)
        for agent in env.agents:
            self.assertAlmostEqual(agent['travel_m'] + agent['service_s'] + agent['idle_s'], 1795)
        self.assertEqual(len(env.scalar_trace), 360)
        self.assertTrue(env.ever.all())
        self.assertEqual(evaluate_episode(env)[0]['invalid_assignments'], 0)


class ConfirmationMetricTests(unittest.TestCase):
    def test_three_bins_span_fifteen_seconds_and_short_events_excluded(self):
        env = ActiveInspection(specification())
        label = np.zeros((360, len(env.tasks)), dtype=bool)
        excess = np.zeros_like(label, dtype=float)
        label[60:63, 0] = True
        label[70:72, 1] = True
        rows = anomaly_intervals(label, excess, env.tasks, env.asset_weights)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]['onset_s'], rows[0]['end_s']), (300, 315))

    def test_no_event_does_not_get_perfect_detection_score(self):
        env = ActiveInspection(specification())
        metrics, roles, _ = evaluate_episode(env)
        # Initial process has no inspected discoveries; no-event precision or
        # recall cannot be silently substituted by a perfect success value.
        self.assertEqual(metrics['anomaly_segments'], 0)
        self.assertIsNone(metrics['miss_rate'])
        self.assertIsNone(metrics['consequence_weighted_censored_delay_s'])
        self.assertEqual(metrics['confirmed_segments'], 0)

    def test_unconfirmed_segment_contributes_censor_delay(self):
        env = ActiveInspection(specification())
        label = np.zeros((360, len(env.tasks)), dtype=bool)
        label[60:63, 0] = True
        with patch('patrol_rsi.inspection_metrics.task_labels', return_value=(label, label.astype(float))):
            metrics, _, rows = evaluate_episode(env)
        self.assertEqual(metrics['miss_rate'], 1)
        self.assertEqual(metrics['consequence_weighted_censored_delay_s'], 15)
        self.assertIsNone(metrics['detected_only_mean_delay_s'])

    @unittest.skipIf(torch is None, 'PyTorch required by runner')
    def test_test_set_and_training_seed_ranges_are_disjoint(self):
        from patrol_rsi.m11 import test_specs, validation_specs
        rows = test_specs()
        self.assertEqual(len(rows), 1136)
        self.assertEqual(sum(r['group'] == 'gw' for r in rows), 96)
        test_seeds = {r['seed'] for r in rows}
        train_seeds = {1000 + s * 100 + episode for s in range(3) for episode in range(48)}
        self.assertFalse(test_seeds & train_seeds)
        self.assertFalse(test_seeds & {r['seed'] for r in validation_specs('nominal')})


@unittest.skipIf(torch is None, 'PyTorch required')
class TaskPolicyTests(unittest.TestCase):
    def test_masked_actions_and_cpu_update_execute(self):
        torch.manual_seed(11)
        model = TaskPolicy()
        features = torch.zeros(3, 16)
        allowed = torch.tensor([False, True, False])
        self.assertEqual(int(model.distribution(features, allowed).sample()), 1)
        env, rollout = ActiveInspection(specification()), []
        for _ in range(8):
            public = env.value_features()
            with torch.inference_mode():
                value = float(model.values(torch.as_tensor(public)))
            choices, samples, lp, _, _ = policy_decisions(env, model, stochastic=True)
            reward = env.step(choices)
            rollout.append(dict(public_value=public, value=value, samples=samples, logprob=lp, reward=reward))
        old = model.actor[0].weight.detach().clone()
        result = ppo_update(model, torch.optim.Adam(model.parameters(), lr=.0003), rollout)
        self.assertEqual(result['full_rollout_epochs'], 4)
        self.assertFalse(torch.equal(old, model.actor[0].weight))


if __name__ == '__main__':
    unittest.main()
