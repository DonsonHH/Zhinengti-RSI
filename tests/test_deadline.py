import unittest
import numpy as np
from patrol_rsi.environment import PatrolEnv
from patrol_rsi.experiment import room
from patrol_rsi.deadline import measured_mean_estimate,observations,public_potential,reward_components,target_deadlines


class DeadlineTests(unittest.TestCase):
    def test_hidden_truth_does_not_change_inputs_or_public_potential(self):
        s=room(13000);a=PatrolEnv(s,truth=np.zeros((24,24)));b=PatrolEnv(s,truth=np.ones((24,24)))
        a.estimate[:]=.5;b.estimate[:]=.5;d=target_deadlines(13000,'balanced')
        np.testing.assert_array_equal(observations(a,d),observations(b,d))
        self.assertEqual(public_potential(a,d,13000),public_potential(b,d,13000))

    def test_deadline_input_ablation_has_same_shape_and_hides_due_vector(self):
        env=PatrolEnv(room(13001));d=target_deadlines(13001,'tight')
        self.assertEqual(observations(env,d).shape,(4,8,24,24))
        env.raw_age[env.scenario.targets]=15
        a=observations(env,d,False);b=observations(env,d[::-1],False)
        np.testing.assert_array_equal(a,b)
        self.assertTrue((a[:,3]==0).all())
        self.assertFalse(np.array_equal(observations(env,d),observations(env,d[::-1])))

    def test_reward_is_independent_of_learned_vae_predictions(self):
        s=room(13002);a=PatrolEnv(s,seed=7);b=PatrolEnv(s,seed=7)
        a.set_estimate(np.zeros((24,24)));b.set_estimate(np.ones((24,24)))
        d=target_deadlines(13002,'balanced');actions=np.full(4,-1)
        ra,_,pa=reward_components(a,d,actions,13002)
        rb,_,pb=reward_components(b,d,actions,13002)
        np.testing.assert_array_equal(ra,rb);self.assertEqual(pa,pb)

    def test_potential_shaping_matches_fixed_formula(self):
        a=PatrolEnv(room(13003),seed=1);b=PatrolEnv(room(13003),seed=1);d=target_deadlines(13003,'balanced')
        ra,_,parts=reward_components(a,d,np.full(4,-1),13003,True)
        rb,_,_=reward_components(b,d,np.full(4,-1),13003,False)
        self.assertTrue(np.allclose(ra-rb,.99*parts['potential_after']-parts['potential_before']))

    def test_no_vae_baseline_keeps_observed_values(self):
        env=PatrolEnv(room(13004));env.measured[5,5]=True;env.values[5,5]=.8
        x=measured_mean_estimate(env)
        self.assertAlmostEqual(float(x[5,5]),.8,places=6);self.assertEqual(float(x[10,10]),.5)

    def test_terminal_transition_has_zero_successor_potential(self):
        env=PatrolEnv(room(13005));d=target_deadlines(13005,'tight')
        _,_,parts=reward_components(env,d,np.full(4,-1),13005,terminal=True)
        self.assertEqual(parts['potential_after'],0)
        self.assertEqual(parts['shaping_reward'],-parts['potential_before'])
