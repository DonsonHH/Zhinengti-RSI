import unittest
import numpy as np
from patrol_rsi.environment import PatrolEnv
from patrol_rsi.experiment import RouteGreedy, room, split_manifest


class ExperimentTests(unittest.TestCase):
    def test_layout_and_field_split_sets_are_disjoint(self):
        m=split_manifest()
        groups=[set(m['train_layout_seeds'])]+[set(x['layout_seed'] for x in m[k]) for k in ['validation','test','development']]
        for i in range(len(groups)):
            for j in range(i):self.assertFalse(groups[i]&groups[j])
        signatures=[]
        for k in ['validation','test','development']:
            for case in m[k]:
                s=room(case['layout_seed']);signatures.append(s.navigation.tobytes()+s.targets.tobytes())
        self.assertEqual(len(signatures),len(set(signatures)))

    def test_known_checkpoints_visible_without_hidden_field(self):
        env=PatrolEnv(room(1000))
        channel=env.policy_observations()[0,1]
        self.assertTrue(np.all(channel[~env.scenario.targets]==0))
        self.assertTrue(np.all(channel[env.scenario.targets]>0))

    def test_route_greedy_reaches_all_checkpoints_in_budget(self):
        env=PatrolEnv(room(2000),seed=4000);controller=RouteGreedy(env)
        for _ in range(96):env.step(env.safe_actions(controller.scores(env)))
        self.assertEqual(env.evaluation()['target_coverage'],1.0)


if __name__=='__main__':unittest.main()
