import itertools
import unittest

import numpy as np

from patrol_rsi.deadline import measured_mean_estimate
from patrol_rsi.dispatch import DispatchEnv
from patrol_rsi.environment import PatrolEnv
from patrol_rsi.experiment import room
from patrol_rsi.m7 import contract
from patrol_rsi.stress import (PairedNoiseEnv, PairCyclePlanner, StressDispatchEnv,
                               arrival_benefit, best_joint_assignment, pairings,
                               planner_choices, stress_room)


class StressTests(unittest.TestCase):
    def test_development_layouts_connected_and_public_features_compatible(self):
        fingerprints = set()
        for case in contract()['development']:
            wrapper = StressDispatchEnv(case)
            env = wrapper.env
            self.assertEqual(int(env.scenario.targets.sum()), 8)
            self.assertTrue(all(np.isfinite(cost[env.scenario.navigation]).all() for cost in wrapper.costs))
            features = wrapper.prepare()['features']
            self.assertEqual(features.shape, (4, 8, 14))
            fingerprints.add(env.scenario.navigation.tobytes() + env.scenario.targets.tobytes())
        self.assertEqual(len(fingerprints), 4)

    def test_noise_is_the_same_for_cell_and_time_despite_different_history(self):
        scenario = room(30000, .03)
        truth = np.full((24, 24), .5, dtype=np.float32)
        left = PairedNoiseEnv(scenario, 33000000, truth)
        right = PairedNoiseEnv(scenario, 33000000, truth)
        other = scenario.targets.copy()
        other[tuple(np.argwhere(other)[0])] = False
        left.t = 2
        left._observe(other)
        left.t = right.t = 9
        left._observe(scenario.targets)
        right._observe(scenario.targets)
        np.testing.assert_array_equal(left.values[scenario.targets], right.values[scenario.targets])
        previous = left.values.copy()
        left.t = 10
        left._observe(scenario.targets)
        self.assertFalse(np.array_equal(previous[scenario.targets], left.values[scenario.targets]))

    def test_clean_rack_replays_m6_control_exactly(self):
        case = contract()['development'][0]
        old = DispatchEnv(case)
        new = StressDispatchEnv(case)
        for _ in range(32):
            old.env.set_estimate(measured_mean_estimate(old.env))
            new.env.set_estimate(measured_mean_estimate(new.env))
            state = old.prepare()
            choices = planner_choices(state, 'slack')
            reward_a, event_a = old.step(choices, state)
            reward_b, event_b = new.step(choices, new.prepare())
            self.assertEqual(event_a, event_b)
            self.assertEqual(reward_a, reward_b)
            np.testing.assert_array_equal(old.env.values, new.env.values)

    def test_assignment_finds_joint_optimum_when_sequential_greedy_fails(self):
        state = StressDispatchEnv(contract()['development'][0]).prepare()
        state['free'][:] = False
        state['free'][:2] = True
        state['allowed'][:] = False
        state['allowed'][:2, :2] = True
        state['reserved'][:] = False
        utility = np.zeros((4, 8))
        utility[0, :2] = [10, 9]
        utility[1, :2] = [9, 0]
        choices, info = best_joint_assignment(utility, state)
        self.assertEqual(choices.tolist(), [1, 0, -1, -1])
        self.assertEqual(info['proxy_value'], 18)
        self.assertEqual(info['valid_combinations'], 2)

    def test_assignment_matches_independent_permutation_reference(self):
        state = StressDispatchEnv(contract()['development'][0]).prepare()
        state['allowed'][:, 0] = False
        state['free'][1] = False
        utility = np.random.default_rng(42).normal(size=(4, 8))
        selected, info = best_joint_assignment(utility, state)
        actors = [int(i) for i in state['order'] if state['free'][i]]
        scores = []
        for candidate in itertools.permutations(range(8), len(actors)):
            if all(state['allowed'][i, j] and not state['reserved'][j] for i, j in zip(actors, candidate)):
                scores.append(sum(utility[i, j] for i, j in zip(actors, candidate)))
        self.assertAlmostEqual(info['proxy_value'], max(scores))
        self.assertEqual(selected[1], -1)
        self.assertEqual(len(set(selected[selected >= 0])), len(actors))

    def test_planning_reads_public_state_not_hidden_field(self):
        case = contract()['development'][2]
        a, b = StressDispatchEnv(case), StressDispatchEnv(case)
        a.env._truth[:] = 0
        b.env._truth[:] = 1
        a.env.estimate[:] = b.env.estimate[:] = .5
        sa, sb = a.prepare(), b.prepare()
        np.testing.assert_array_equal(arrival_benefit(sa), arrival_benefit(sb))
        np.testing.assert_array_equal(planner_choices(sa, 'matching'), planner_choices(sb, 'matching'))
        self.assertEqual(PairCyclePlanner(a).pairs, PairCyclePlanner(b).pairs)

    def test_pair_partition_and_planner_preserve_legal_commitments(self):
        self.assertEqual(len(list(pairings(range(8)))), 105)
        wrapper = StressDispatchEnv(contract()['development'][2])
        planner = PairCyclePlanner(wrapper)
        self.assertEqual(sorted(point for pair in planner.pairs for point in pair), list(range(8)))
        for _ in range(48):
            state = wrapper.prepare()
            choices = planner_choices(state, 'pair_cycle', planner)
            self.assertTrue((choices[~state['free']] == -1).all())
            wrapper.step(choices, state)
        self.assertEqual(wrapper.duplicate_assignments, 0)

    def test_startup_lower_bound_counts_only_steps_before_possible_observation(self):
        wrapper = StressDispatchEnv(contract()['development'][2])
        bounds = wrapper.initial_access_bounds()
        earliest = bounds['earliest_possible_first_observation_steps']
        count = sum(sum(step > due and step < arrival for step in range(1, 97))
                    for due, arrival in zip(wrapper.deadlines, earliest))
        self.assertEqual(bounds['startup_violation_lower_bound'], count / (96 * 8))
        self.assertIn('not a recurrent feasibility proof', bounds['scope'])


if __name__ == '__main__':
    unittest.main()
