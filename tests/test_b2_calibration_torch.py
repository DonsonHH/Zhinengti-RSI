import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Real bounded calibration requires Torch')
class CalibrationDiagnosisTests(unittest.TestCase):
    def test_tied_auc_and_balanced_prior_correction(self):
        from patrol_rsi.b2_calibration import class_weights,weighted_auc,sigmoid
        y=np.r_[np.zeros(99),1];w=np.ones(100)
        weights,p=class_weights(y,w,True)
        self.assertAlmostEqual(weights[y==1].sum(),.5);self.assertAlmostEqual(p,.01)
        self.assertAlmostEqual(weighted_auc(y,np.zeros(100),w),.5)
        self.assertAlmostEqual(float(sigmoid(np.log(p/(1-p)))),p)

    def test_ridge_removal_changes_dynamic_range_at_same_frozen_scores(self):
        from patrol_rsi.b2_calibration import fit,sigmoid
        from patrol_rsi.limited_model import fit_calibrator
        s=np.linspace(-3,3,400);y=(s>2.7).astype(int);w=np.full(400,1/400)
        original=fit_calibrator(s,y,w);new=fit(s,y,w,.0001)
        self.assertEqual(new['status'],'completed_within_fixed_optimizer_budget')
        self.assertLessEqual(new['closure_evaluations'],300)
        self.assertGreater(new['a'],original['a'])
        self.assertGreater(np.ptp(sigmoid(new['a']*s+new['b'])),np.ptp(sigmoid(original['a']*s+original['b'])))


if __name__=='__main__':unittest.main()
