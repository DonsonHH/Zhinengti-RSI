import importlib.util
import unittest
import numpy as np
from patrol_rsi.m5 import VARIANTS,contract,rank


class M5ContractTests(unittest.TestCase):
    def test_training_validation_and_test_geometries_use_disjoint_seed_ranges(self):
        c=contract();train=set(c['train_layout_seeds']);val={r['layout_seed'] for r in c['validation']};test={r['layout_seed'] for r in c['test']}
        self.assertFalse(train&val or train&test or val&test)
        self.assertEqual(len(train),64);self.assertEqual(len(c['test']),48)
        for layout in test:
            rows=[r for r in c['test'] if r['layout_seed']==layout]
            self.assertEqual({(r['profile'],r['noise']) for r in rows},{('balanced',0.0),('balanced',.03),('tight',0.0),('tight',.03)})

    def test_primary_rank_preserves_coverage_priority(self):
        self.assertLess(rank({'target_coverage':1,'deadline_violation':.2,'weighted_idleness':.5}),
                        rank({'target_coverage':.9,'deadline_violation':0,'weighted_idleness':.1}))


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'Cluster PyTorch required')
class M5NeuralTests(unittest.TestCase):
    def test_same_architecture_can_update_from_eight_channel_replay(self):
        import torch
        from patrol_rsi.m5 import make_policy
        from patrol_rsi.models import PrioritizedReplay,ddqn_update
        torch.manual_seed(3);torch.set_num_threads(2)
        online,target=make_policy(),make_policy();target.load_state_dict(online.state_dict())
        x=np.zeros((8,24,24),np.float32);replay=PrioritizedReplay(seed=3)
        for i in range(16):replay.add(x,i%8,1,x,True,np.ones(8,bool))
        old=online.value.weight.detach().clone()
        loss=ddqn_update(online,target,torch.optim.Adam(online.parameters()),replay,torch.device('cpu'))
        self.assertTrue(np.isfinite(loss));self.assertFalse(torch.equal(old,online.value.weight))
