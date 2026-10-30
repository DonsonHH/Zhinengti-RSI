import importlib.util
import unittest
import json
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'Torch unavailable in lightweight host')
class LimitedModelTorchTests(unittest.TestCase):
    def data(self,batch):
        from patrol_rsi.limited_prediction import COL
        class Fixture:
            def batches(self,batch_rows=16384):
                for start in range(0,self.n,batch):
                    end=min(self.n,start+batch);yield slice(start,end),self.x[start:end],self.w[start:end],self.y[start:end]
        d=Fixture();d.n=64;d.x=np.zeros((64,98),np.float32)
        d.x[:,COL['root_time_s']]=np.tile([30,60,90,120],16);d.x[:,COL['pod']]=np.arange(64)%16
        d.x[:,COL['temperature_measurement.value']]=np.linspace(40,80,64)
        d.x[:,COL['temperature_measurement.available']]=1
        target=(np.arange(64)>31).astype(np.int8);d.y=np.column_stack((target,target));d.w=np.full(64,1/64)
        return d

    def test_streamed_gradient_fit_matches_single_batch_fit(self):
        from patrol_rsi.limited_model import fit_lr,lr_scores
        a=self.data(8);b=self.data(64)
        ma=fit_lr(a,'public_lr','thermal',1,.1);mb=fit_lr(b,'public_lr','thermal',1,.1)
        self.assertEqual(ma['status'],'completed_within_fixed_optimizer_budget')
        self.assertEqual(mb['status'],'completed_within_fixed_optimizer_budget')
        np.testing.assert_allclose(lr_scores(a,ma),lr_scores(b,mb),atol=2e-6,rtol=2e-6)
        self.assertLessEqual(ma['closure_evaluations'],400)
        self.assertTrue(ma['no_full_float64_design_matrix'])

    def test_calibration_is_monotone_finite_and_bounded_budget(self):
        from patrol_rsi.limited_model import fit_calibrator,calibrated
        scores=np.linspace(-2,2,64);y=(scores>0).astype(np.int8);w=np.ones(64)
        c=fit_calibrator(scores,y,w);self.assertEqual(c['status'],'completed_within_fixed_optimizer_budget')
        p=calibrated(scores,c);self.assertTrue((np.diff(p)>=0).all())
        self.assertTrue(np.isfinite(p).all());self.assertLessEqual(c['closure_evaluations'],300)

    def test_nonfinite_fit_failures_remain_serializable_and_consume_slot(self):
        import torch
        from patrol_rsi.limited_model import fit_lr,fit_calibrator
        def corrupt(opt,closure):
            with torch.no_grad():opt.param_groups[0]['params'][0].fill_(float('nan'))
            return closure()
        with patch.object(torch.optim.LBFGS,'step',corrupt):
            model=fit_lr(self.data(8),'public_lr','thermal',1,.1)
            cal=fit_calibrator(np.linspace(-1,1,8),np.array([0]*4+[1]*4),np.ones(8))
        self.assertEqual(model['status'],'failed_consumes_candidate_slot');self.assertIsNone(model['coefficient'])
        self.assertEqual(cal['status'],'failed_consumes_candidate_slot')
        json.dumps([model,cal],allow_nan=False)

    def test_optimizer_budget_return_requires_real_termination(self):
        import torch
        from patrol_rsi.limited_model import verify_termination,FitBudgetExceeded
        parameter=torch.tensor([1.,1.],dtype=torch.float64,requires_grad=True);parameter.grad=torch.ones(2,dtype=torch.float64)
        state={'n_iter':300,'func_evals':400,'prev_loss':10.,'d':torch.ones(2,dtype=torch.float64),'t':1.}
        with self.assertRaises(FitBudgetExceeded):verify_termination(state,parameter,400,1.,300,400)
        parameter.grad.zero_();self.assertTrue(verify_termination(state,parameter,400,1.,300,400)['termination_tolerance_satisfied'])


if __name__=='__main__':unittest.main()
