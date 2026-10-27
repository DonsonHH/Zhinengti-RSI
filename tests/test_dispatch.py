import importlib.util
import unittest
import numpy as np
from patrol_rsi.dispatch import DispatchEnv,heuristic_choices


def case():return {'split':'unit','layout_seed':16000,'field_seed':23000000,'profile':'balanced','noise':0.0}


class DispatchTests(unittest.TestCase):
    def test_joint_assignment_and_low_level_replay_are_identical(self):
        a=DispatchEnv(case());b=DispatchEnv(case())
        for _ in range(32):
            state=a.prepare();choices=heuristic_choices(state,'slack')
            bstate=b.prepare();reward,event=a.step(choices,state)
            other_reward,other_event=b.step(choices,bstate)
            self.assertEqual(event,other_event);self.assertEqual(reward,other_reward)
            self.assertEqual(len(set(a.assigned)),4)

    def test_joint_rejects_duplicate_new_target_but_independent_control_allows_it(self):
        env=DispatchEnv(case());s=env.prepare();j=int(np.flatnonzero(s['allowed'].all(0))[0]);choices=np.full(4,j)
        with self.assertRaises(ValueError):env.assign(choices,s,True)
        np.testing.assert_array_equal(env.assigned,np.full(4,-1))
        self.assertEqual(env.assignments,0);self.assertEqual(env.duplicate_assignments,0)
        env=DispatchEnv(case());env.assign(choices,env.prepare(),False)
        self.assertEqual(env.duplicate_assignments,3)

    def test_late_invalid_choice_has_no_partial_side_effect(self):
        env=DispatchEnv(case());state=env.prepare();choices=heuristic_choices(state,'slack')
        choices[state['order'][-1]]=99
        before=env.assigned.copy()
        with self.assertRaises(ValueError):env.assign(choices,state)
        np.testing.assert_array_equal(env.assigned,before)
        self.assertEqual(env.assignments,0);self.assertEqual(env.env.t,0)

    def test_hidden_truth_not_in_high_level_features_or_unseen_reward(self):
        a=DispatchEnv(case());b=DispatchEnv(case())
        b.env._truth[:]=1;a.env._truth[:]=0
        a.env.estimate[:]=.5;b.env.estimate[:]=.5
        sa=a.prepare();sb=b.prepare();np.testing.assert_array_equal(sa['features'],sb['features'])
        choices=heuristic_choices(sa,'slack')
        # Before any target is measured, the public reward is truth-independent.
        ra,_=a.step(choices,sa);rb,_=b.step(choices,sb);self.assertEqual(ra,rb)

    def test_busy_executors_keep_target_until_observed(self):
        env=DispatchEnv(case());s=env.prepare();choices=heuristic_choices(s,'slack');env.step(choices,s)
        assigned=env.assigned.copy();state=env.prepare()
        for i in np.flatnonzero(~state['free']):self.assertEqual(env.assigned[i],assigned[i])
        with self.assertRaises(ValueError):env.assign(np.zeros(4,dtype=int),state)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'Cluster PyTorch required')
class DispatchNeuralTests(unittest.TestCase):
    def test_joint_behavior_replay_and_gradient(self):
        import torch
        from patrol_rsi.dispatch_model import DispatchActorCritic,tensors,ppo_update
        torch.manual_seed(1);torch.set_num_threads(2);model=DispatchActorCritic();env=DispatchEnv(case());roll=[]
        for t in range(8):
            state=env.prepare();ts=tensors([state],torch.device('cpu'))
            with torch.no_grad():choices,logp,_,value=model.joint(**ts)
            action=choices[0].numpy();reward,_=env.step(action,state)
            roll.append({'state':state,'choices':action,'logp':float(logp[0]),'value':float(value[0]),'reward':reward,'done':t==7})
        before=model.actor[-1].weight.detach().clone()
        out=ppo_update(model,torch.optim.Adam(model.parameters(),lr=.001),roll,torch.device('cpu'),epochs=2,batch_size=8)
        self.assertLess(out['behavior_logp_max_error'],2e-5)
        self.assertFalse(torch.equal(before,model.actor[-1].weight))

    def test_advantage_does_not_leak_across_episode_boundary(self):
        from patrol_rsi.dispatch_model import generalized_advantage
        a,_=generalized_advantage([1,100],[0,0],[True,True])
        self.assertEqual(float(a[0]),1)
