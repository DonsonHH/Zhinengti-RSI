import unittest

import numpy as np

from patrol_rsi.environment import PatrolEnv, Scenario, greedy_scores, make_scenario


class EnvironmentTests(unittest.TestCase):
    def test_idleness_resets_only_where_observed(self):
        nav = np.ones((5, 5), bool)
        env = PatrolEnv(Scenario("tiny", nav, nav, np.array([[2, 2]]), radius=0, forgetting=0.2), truth=np.zeros((5, 5)))
        env.step(np.array([0]))
        self.assertEqual(env.idleness[3, 2], 0)
        self.assertAlmostEqual(env.idleness[2, 2], 0.2)
        for _ in range(8):
            env.step(np.array([-1]))
        self.assertEqual(env.idleness[0, 0], 1)
        self.assertEqual(env.evaluation()["max_unobserved_steps"], 9)

    def test_policy_does_not_receive_unmeasured_ground_truth(self):
        scenario = make_scenario()
        a = PatrolEnv(scenario, truth=np.zeros(scenario.navigation.shape))
        hidden_b = np.zeros(scenario.navigation.shape)
        hidden_point = tuple(np.argwhere(scenario.targets & ~a.measured)[0])
        hidden_b[hidden_point] = 0.75
        b = PatrolEnv(scenario, truth=hidden_b)
        np.testing.assert_array_equal(a.policy_observations(), b.policy_observations())
        np.testing.assert_array_equal(a.estimator_input(), b.estimator_input())
        self.assertNotEqual(a.evaluation()["reconstruction_mse"], b.evaluation()["reconstruction_mse"])

    def test_fractional_tool_action_is_rejected_without_mutation(self):
        env = PatrolEnv(make_scenario())
        before = env.positions.copy()
        with self.assertRaises(ValueError):
            env.step(np.array([0.7, 1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(env.positions, before)
        self.assertEqual(env.t, 0)

    def test_consensus_avoids_shared_destinations_and_obstacles(self):
        env = PatrolEnv(make_scenario(), seed=12)
        for _ in range(100):
            actions = env.safe_actions(greedy_scores(env))
            env.step(actions)
            self.assertEqual(len(set(map(tuple, env.positions))), 4)
            self.assertTrue(all(env.scenario.navigation[tuple(p)] for p in env.positions))

    def test_overlapping_measurements_do_not_duplicate_team_reward(self):
        nav = np.ones((6, 6), bool)
        scenario = Scenario("tiny", nav, nav, np.array([[1, 1], [1, 3]]), radius=1)
        env = PatrolEnv(scenario, truth=np.zeros((6, 6)))
        env.idleness[:] = 0.5
        env.estimate[:] = 0.4
        masks = np.stack([env.neighborhood(p) for p in env.positions])
        expected = float((0.5 * 0.5 * masks.any(axis=0)).sum())
        event = env.step(np.array([-1, -1]))
        self.assertAlmostEqual(sum(event["rewards"]), expected, places=5)

    def test_reset_replays_measurement_noise_and_actions(self):
        env = PatrolEnv(make_scenario(), seed=5)
        first = [env.step(env.safe_actions(greedy_scores(env))) for _ in range(8)]
        env.reset()
        second = [env.step(env.safe_actions(greedy_scores(env))) for _ in range(8)]
        self.assertEqual(first, second)

    def test_rack_corner_and_crossing_movements_are_rejected(self):
        nav = np.ones((5, 5), bool)
        nav[2, 1] = False
        env = PatrolEnv(Scenario("tiny", nav, nav, np.array([[1, 1]])), truth=np.zeros((5, 5)))
        self.assertFalse(env.valid_move(env.positions[0], 1))
        nav = np.ones((5, 5), bool)
        env = PatrolEnv(Scenario("tiny", nav, nav, np.array([[1, 1], [2, 1]])), truth=np.zeros((5, 5)))
        with self.assertRaises(ValueError):
            env.step(np.array([1, 3]))


if __name__ == "__main__":
    unittest.main()
