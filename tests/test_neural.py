import importlib.util
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from patrol_rsi.models import ConditionalVAEUNet, PrioritizedReplay, SharedDuelingQ, ddqn_update, vae_loss


@unittest.skipUnless(HAS_TORCH, "Neural tests require the cluster PyTorch module")
class NeuralTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        torch.set_num_threads(2)

    def test_prior_prediction_cannot_invoke_ground_truth_posterior(self):
        model = ConditionalVAEUNet().eval()
        def reject(_module, _inputs):
            raise AssertionError("Posterior used during deployment")
        model.posterior.register_forward_pre_hook(reject)
        prediction = model.predict(torch.zeros(1, 2, 24, 24), samples=2)
        self.assertEqual(tuple(prediction.shape), (1, 1, 24, 24))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_vae_loss_has_real_gradient_and_requires_staged_perceptual_model(self):
        model = ConditionalVAEUNet()
        x = torch.rand(2, 2, 24, 24)
        target = torch.rand(2, 1, 24, 24)
        predicted, prior, posterior = model(x, target)
        loss, _ = vae_loss(predicted, target, prior, posterior)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.prior.parameters()))
        with self.assertRaises(ValueError):
            vae_loss(predicted, target, prior, posterior, perceptual_weight=0.1)

    def test_vae_can_learn_spatial_reconstruction_instead_of_constant_output(self):
        # A small, fixed learnable pattern catches saturation that a finite loss
        # and one nonzero gradient cannot detect. No pretrained resources needed.
        model=ConditionalVAEUNet()
        y=torch.zeros(4,1,24,24);y[:,:,6:18,6:18]=.8
        mask=torch.zeros_like(y);mask[:,:,::3,::3]=1
        x=torch.cat([y*mask,mask],dim=1)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
        for _ in range(80):
            out,prior,posterior=model(x,y)
            loss,_=vae_loss(out,y,prior,posterior)
            optimizer.zero_grad();loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():
            estimate=model.predict(x,4)
        self.assertLess(float((estimate-y).square().mean()),float((y-y.mean()).square().mean()))
        self.assertGreater(float(estimate[:,:,8:16,8:16].mean()-estimate[:,:,:4,:4].mean()),.15)

    def test_replay_drives_shared_dueling_double_q_parameter_update(self):
        online, target = SharedDuelingQ(), SharedDuelingQ()
        target.load_state_dict(online.state_dict())
        replay = PrioritizedReplay(seed=4)
        x = np.zeros((5, 24, 24), dtype=np.float32)
        for i in range(16):
            replay.add(x, i % 8, 0.5, x, i % 2 == 0, np.ones(8, bool))
        old = online.value.weight.detach().clone()
        loss = ddqn_update(online, target, torch.optim.Adam(online.parameters()), replay, torch.device("cpu"))
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(old, online.value.weight.detach()))


if __name__ == "__main__":
    unittest.main()
