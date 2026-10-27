import unittest
from unittest.mock import Mock

import numpy as np

from patrol_rsi.estimation import (ObservationState, history_context, bounded_gp,
                                   radial_knn, temperature_metrics)
from patrol_rsi.estimation_data import (case_specs, contract, generate_case,
                                        CompositeFacility, deterministic_digest)
from patrol_rsi.facility import CoupledFacility, FacilityConfig

try:
    import torch
    from patrol_rsi.estimation_model import TemperatureCVAE
except ImportError:
    torch = None


class EstimationDataTests(unittest.TestCase):
    def test_disjoint_case_splits_and_heldout_combinations(self):
        train = case_specs('train')
        test = case_specs('test')
        self.assertEqual(sum(r['split'] == 'train' for r in train), 224)
        self.assertEqual(sum(r['split'] == 'validation' for r in train), 56)
        self.assertEqual(sum(r['split'] == 'test' for r in test), 306)
        self.assertEqual(sum(r['split'] == 'gw' for r in test), 12)
        self.assertFalse({r['seed'] for r in train} & {r['seed'] for r in test})
        self.assertTrue(all('+' not in r['scenario'] for r in train))

    def test_history_padding_and_future_independence(self):
        x = np.arange(6 * 2 * 3).reshape(6, 2, 3)
        before = history_context(x)
        changed = x.copy()
        changed[4:] = -999
        np.testing.assert_equal(before[:4], history_context(changed)[:4])
        np.testing.assert_equal(before[0], np.tile(x[0], (1, 4)))
        np.testing.assert_equal(history_context(x, current_only=True)[3], np.tile(x[3], (1, 4)))

    def test_packet_whitelist_and_future_inspection_guard(self):
        plant = CoupledFacility(FacilityConfig(), seed=901)
        plant.step()
        packet = plant.public_telemetry()
        state = ObservationState(plant.config)
        with self.assertRaises(ValueError):
            state.update(dict(packet, hidden_fault='healthy'), [])
        with self.assertRaises(ValueError):
            state.update(packet, [{'time_s': 10, 'completion': None}])
        state.update(packet, [])
        first = state.features(packet).copy()
        plant._rack_c[:] = -1000
        np.testing.assert_equal(first, state.features(packet))

    def test_composition_preserves_atom_and_adds_second_fault(self):
        config = FacilityConfig()
        atom = CoupledFacility(config, 900, 'tower_derate')
        composite = CompositeFacility(config, 900, 'tower_derate+sensor_bias', 'feedback')
        first, second = atom._faults(500), composite._faults(500)
        self.assertEqual(first['tower'], second['tower'])
        self.assertTrue(np.any(second['sensor_bias'] < 0))
        self.assertTrue(np.all(composite._faults(50)['sensor_bias'] == 0))

    def test_noisy_sensor_conditions_keep_truth_and_inspections_identical(self):
        spec = dict(split='development', scenario='sensor_bias', seed=900,
                    control='feedback', condition='nominal')
        first, meta1 = generate_case(spec, baselines=False)
        second, meta2 = generate_case(dict(spec, condition='sparse'), baselines=False)
        np.testing.assert_equal(first['temperature_c'], second['temperature_c'])
        self.assertEqual(meta1['effective_inspection_events'], meta2['effective_inspection_events'])
        self.assertNotEqual(deterministic_digest(first), deterministic_digest(second))
        self.assertEqual(first['features'].shape, (60, 128, 14))

    def test_gw_budget_is_supply_not_IT_or_FLOPS(self):
        c = contract()
        self.assertEqual(c['gw_racks'] * FacilityConfig().rated_rack_kw / 1000,
                         c['gw_IT_nameplate_MW'])
        self.assertGreater(c['gw_supply_design_budget_MW'], c['gw_IT_nameplate_MW'])
        self.assertIn('not an implemented global power limiter', c['gw_budget_semantics'])


class EstimatorControlTests(unittest.TestCase):
    def test_gp_ignores_samples_outside_frozen_budget(self):
        x = np.column_stack((np.arange(128), np.zeros(128)))
        stamp = np.concatenate((np.zeros(64), np.full(64, -1000.)))
        values = np.linspace(45, 65, 128)
        before = bounded_gp(values, stamp, x, 5, capacity=64)
        values[64:] += 1000
        np.testing.assert_equal(before, bounded_gp(values, stamp, x, 5, capacity=64))
        np.testing.assert_equal(bounded_gp(values, np.full(128, -1), x, 5), np.full(128, 50))

    def test_gp_distributes_simultaneous_sensor_budget_across_facility(self):
        x = np.column_stack((np.arange(128), np.zeros(128)))
        values = np.linspace(50, 70, 128)
        prediction = bounded_gp(values, np.zeros(128), x, 5, capacity=2)
        self.assertLess(abs(prediction[0] - 50), .2)
        self.assertLess(abs(prediction[-1] - 70), .2)

    def test_rknn_reproduces_unique_fresh_measured_point(self):
        x = np.array([[0., 0.], [1., 0.], [2., 0.]])
        prediction = radial_knn(np.array([70., 40., 50.]), np.array([5., -1., -1.]), x, 5)
        np.testing.assert_equal(prediction, np.full(3, 70.))

    def test_metric_units_weights_and_empty_hotspots(self):
        metric = temperature_metrics(np.array([[50., 50.]]), np.array([[51., 53.]]), [1., 3.])
        self.assertAlmostEqual(metric['rmse_c'], np.sqrt(5))
        self.assertAlmostEqual(metric['weighted_rmse_c'], np.sqrt(7))
        self.assertIsNone(metric['precision'])
        self.assertIsNone(metric['recall'])
        self.assertIsNone(metric['r2'])
        hot = temperature_metrics(np.array([[70., 50.]]), np.array([[60., 70.]]), [1, 1])
        self.assertEqual((hot['tp'], hot['fp'], hot['fn']), (0, 1, 1))


@unittest.skipIf(torch is None, 'PyTorch required')
class ConditionalInferenceTests(unittest.TestCase):
    def test_inference_never_calls_target_posterior(self):
        torch.manual_seed(0)
        model = TemperatureCVAE()
        model.posterior.forward = Mock(side_effect=AssertionError('Posterior used in inference'))
        with torch.inference_mode():
            prediction = model.predict(torch.zeros(5, 56))
        self.assertEqual(tuple(prediction.shape), (5, 1))

    def test_same_capacity_and_paired_initialization_for_history_ablation(self):
        torch.manual_seed(3)
        first = TemperatureCVAE()
        torch.manual_seed(3)
        second = TemperatureCVAE()
        for key in first.state_dict():
            torch.testing.assert_close(first.state_dict()[key], second.state_dict()[key], rtol=0, atol=0)

    def test_training_loss_updates_prior_and_decoder(self):
        torch.manual_seed(2)
        model = TemperatureCVAE()
        loss, kl = model(torch.randn(8, 56), torch.randn(8, 1))
        loss.backward()
        self.assertTrue(torch.isfinite(loss) and torch.isfinite(kl))
        self.assertIsNotNone(model.prior[0].weight.grad)
        self.assertIsNotNone(model.decoder[0].weight.grad)


if __name__ == '__main__':
    unittest.main()
