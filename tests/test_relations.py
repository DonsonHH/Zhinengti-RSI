import importlib.util
import unittest

import numpy as np

from patrol_rsi.m8 import VARIANTS, contract, training_case
from patrol_rsi.relations import RelationDispatchEnv
from patrol_rsi.stress import StressDispatchEnv, planner_choices


class RelationStateTests(unittest.TestCase):
    def test_relation_channels_match_public_map_and_neighbor_features(self):
        wrapper = RelationDispatchEnv(contract()['preflight'])
        state = wrapper.prepare()
        features, edges, team = state['features'], state['target_relations'], state['team_relations']
        self.assertEqual(edges.shape, (8, 8, 8))
        self.assertEqual(team.shape, (4, 8, 4, 8))
        for left in range(8):
            for right in range(8):
                self.assertAlmostEqual(float(edges[left, right, 0]), float(wrapper.costs[right][wrapper.targets[left]]) / 48, places=6)
                np.testing.assert_array_equal(edges[left, right, 1:], features[0, right, [0, 1, 2, 3, 6, 7, 8]])
        for actor in range(4):
            np.testing.assert_allclose(team[0, :, actor, 0], features[actor, :, 4] * 22 / 48)
            self.assertEqual(float(team[actor, 0, actor, 7]), 1)

    def test_hidden_truth_does_not_change_any_relation_state(self):
        left, right = RelationDispatchEnv(contract()['preflight']), RelationDispatchEnv(contract()['preflight'])
        left.env._truth[:] = 0
        right.env._truth[:] = 1
        left.env.estimate[:] = right.env.estimate[:] = .5
        a, b = left.prepare(), right.prepare()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    def test_same_control_and_public_reward_as_m7(self):
        case = contract()['preflight']
        left, right = RelationDispatchEnv(case), StressDispatchEnv(case)
        for _ in range(32):
            state_a, state_b = left.prepare(), right.prepare()
            choices = planner_choices(state_a, 'slack')
            reward_a, event_a = left.step(choices, state_a)
            reward_b, event_b = right.step(choices, state_b)
            self.assertEqual(event_a, event_b)
            self.assertEqual(reward_a, reward_b)

    def test_layout_index_profile_and_field_schedules_are_paired(self):
        schedules, families = [], []
        for variant in VARIANTS:
            rng = np.random.default_rng(1)
            rows, names = [], []
            for index in range(128):
                case, row = training_case(contract(), variant, 1, index, rng)
                rows.append(row)
                names.append(case['family'])
            schedules.append(rows)
            families.append(names)
        self.assertTrue(all(rows == schedules[0] for rows in schedules))
        self.assertEqual(set(families[0]), {'rack'})
        self.assertEqual(set(families[2]), {'rack', 'baffle'})


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'Cluster PyTorch required')
class RelationNetworkTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(2)
        self.torch = torch
        self.state = RelationDispatchEnv(contract()['preflight']).prepare()

    def test_matched_parameters_initialization_and_common_critic(self):
        torch = self.torch
        from patrol_rsi.relation_model import RelationActorCritic, tensors
        torch.manual_seed(7)
        local = RelationActorCritic(False)
        torch.manual_seed(7)
        relation = RelationActorCritic(True)
        self.assertEqual(sum(p.numel() for p in local.parameters()), 22594)
        for left, right in zip(local.parameters(), relation.parameters()):
            self.assertTrue(torch.equal(left, right))
        state = tensors([self.state], torch.device('cpu'))
        with torch.no_grad():
            _, _, _, left_value = local.joint(**state, deterministic=True)
            _, _, _, right_value = relation.joint(**state, deterministic=True)
        self.assertTrue(torch.equal(left_value, right_value))

    def test_local_actor_ignores_relations_while_critic_can_use_them(self):
        torch = self.torch
        from patrol_rsi.relation_model import RelationActorCritic, tensors
        model = RelationActorCritic(False)
        state = tensors([self.state], torch.device('cpu'))
        f, edge, team = state['features'][:, 0], state['target_relations'], state['team_relations'][:, 0]
        a = model.target_logits(f, edge, team)
        b = model.target_logits(f, edge + 1, team - 1)
        self.assertTrue(torch.equal(a, b))
        v1 = model.value(state['features'], edge, state['team_relations'])
        v2 = model.value(state['features'], edge + 1, state['team_relations'] - 1)
        self.assertFalse(torch.allclose(v1, v2))

    def test_target_permutation_equivariance_and_team_slot_invariance(self):
        torch = self.torch
        from patrol_rsi.relation_model import RelationActorCritic, tensors
        model = RelationActorCritic(True)
        state = tensors([self.state], torch.device('cpu'))
        f, edge, team = state['features'][:, 0], state['target_relations'], state['team_relations'][:, 0]
        permutation = torch.tensor([3, 1, 7, 0, 5, 2, 4, 6])
        original = model.target_logits(f, edge, team)
        permuted = model.target_logits(f[:, permutation], edge[:, permutation][:, :, permutation], team[:, permutation])
        torch.testing.assert_close(permuted, original[:, permutation], atol=1e-6, rtol=1e-6)
        team_permuted = model.target_logits(f, edge, team[:, :, [2, 0, 3, 1]])
        torch.testing.assert_close(team_permuted, original, atol=1e-6, rtol=1e-6)

    def test_conditional_collection_replay_probability_and_gradients_both_modes(self):
        torch = self.torch
        from patrol_rsi.relation_model import RelationActorCritic, ppo_update, tensors
        for relational in (False, True):
            torch.manual_seed(3)
            model = RelationActorCritic(relational)
            wrapper = RelationDispatchEnv(contract()['preflight'])
            rollout = []
            for step in range(8):
                state = wrapper.prepare()
                args = tensors([state], torch.device('cpu'))
                with torch.no_grad():
                    choice, logp, _, value = model.joint(**args)
                    _, repeated, _, _ = model.joint(**args, choices=choice)
                torch.testing.assert_close(logp, repeated, atol=1e-6, rtol=1e-6)
                reward, _ = wrapper.step(choice[0].numpy(), state)
                rollout.append({'state': state, 'choices': choice[0].numpy(), 'reward': reward,
                                'logp': float(logp[0]), 'value': float(value[0]), 'done': step == 7})
            before = model.actor[-1].weight.detach().clone()
            outcome = ppo_update(model, torch.optim.Adam(model.parameters(), lr=.001), rollout, torch.device('cpu'), epochs=2, batch_size=8)
            self.assertLessEqual(outcome['behavior_logp_max_error'], 2e-5)
            self.assertFalse(torch.equal(before, model.actor[-1].weight))


if __name__ == '__main__':
    unittest.main()
