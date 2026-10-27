import unittest
from dataclasses import replace

import numpy as np

from patrol_rsi.facility import CoupledFacility, FacilityConfig, facility_topology, secondary_hydraulics
from patrol_rsi.facility_observation import (FacilityObserver, FixedQualifiedInspection,
                                            InspectionTask, eligible, inspection_tasks)
from patrol_rsi.m9 import run_case


class CoupledPhysicsTests(unittest.TestCase):
    def test_electrical_and_full_control_volume_energy_balance(self):
        config = FacilityConfig(racks=16, horizon_s=600)
        plant = CoupledFacility(config, seed=91, scenario='combined')
        for _ in range(120):
            row = plant.step()
            self.assertLess(abs(row['electric_residual_kw']), 1e-7)
            self.assertLess(abs(row['energy_residual_kj']), 1e-6)
            self.assertTrue(np.isfinite(row['rack_temperature_c']).all())
            self.assertTrue((row['rack_flow_kg_s'] >= 0).all())
            self.assertLessEqual(row['it_power_kw'], row['source_power_kw'])

    def test_parallel_branches_and_pump_curve_residual(self):
        config = FacilityConfig(racks=16)
        valve = np.ones(16)
        valve[0] = .2
        speed = np.asarray([.8, 1.0])
        flow, pressure, power = secondary_hydraulics(config, speed, valve)
        branch_k = config.branch_nominal_dp_kpa / config.nominal_rack_flow_kg_s ** 2 / valve ** 2
        np.testing.assert_allclose(branch_k * flow ** 2, np.repeat(pressure, 8), rtol=1e-12)
        pump_k = (config.pump_shutoff_kpa - config.branch_nominal_dp_kpa) / (8 * config.nominal_rack_flow_kg_s) ** 2
        np.testing.assert_allclose(pressure, config.pump_shutoff_kpa * speed ** 2 - pump_k * flow.reshape(2, 8).sum(1) ** 2, rtol=1e-12)
        self.assertLess(flow[0], flow[1])
        self.assertTrue((power > 0).all())

    def test_upstream_auxiliary_power_failure_reduces_flow_and_heats_racks(self):
        config = FacilityConfig(racks=16, horizon_s=900)
        healthy = CoupledFacility(config, seed=90, scenario='healthy', control='fixed_commands')
        failed = CoupledFacility(config, seed=90, scenario='aux_feeder_loss', control='fixed_commands')
        for _ in range(180):
            baseline, fault = healthy.step(), failed.step()
        self.assertLess(fault['rack_flow_kg_s'][:8].mean(), baseline['rack_flow_kg_s'][:8].mean() * .4)
        self.assertGreater(fault['rack_temperature_c'][:8].mean(), baseline['rack_temperature_c'][:8].mean() + 5)
        self.assertGreater(fault['impact_weighted_severity'], baseline['impact_weighted_severity'])

    def test_power_loss_is_not_misreported_as_low_risk_just_because_it_cools(self):
        config = FacilityConfig(racks=16, horizon_s=900)
        plant = CoupledFacility(config, seed=90, scenario='it_feeder_loss')
        for _ in range(180):
            row = plant.step()
        self.assertGreater(row['service_gap'][:8].mean(), .8)
        self.assertGreater(row['severity_proxy'][:8].mean(), .2)

    def test_local_coldplate_fault_has_largest_temperature_effect_at_target(self):
        config = FacilityConfig(racks=16, horizon_s=900)
        healthy = CoupledFacility(config, seed=90)
        failed = CoupledFacility(config, seed=90, scenario='coldplate_fouling')
        for _ in range(180):
            baseline, fault = healthy.step(), failed.step()
        target = (90 % config.pods) * 8 + (90 * 3) % 8
        difference = fault['rack_temperature_c'] - baseline['rack_temperature_c']
        self.assertEqual(int(difference.argmax()), target)
        self.assertGreater(difference[target], 5)

    def test_dt_halving_is_consistent_on_a_nonfault_development_case(self):
        config = FacilityConfig(racks=16, horizon_s=600)
        coarse, fine = CoupledFacility(config, seed=93), CoupledFacility(replace(config, dt_s=2.5), seed=93)
        for _ in range(120): coarse.step()
        for _ in range(240): fine.step()
        np.testing.assert_allclose(coarse.truth_snapshot()['rack_temperature_c'], fine.truth_snapshot()['rack_temperature_c'], atol=.5, rtol=0)
        self.assertAlmostEqual(coarse.t_s, fine.t_s)

    def test_sparse_topology_has_power_cooling_and_control_routes(self):
        small, large = facility_topology(FacilityConfig(racks=16)), facility_topology(FacilityConfig(racks=32))
        nodes = {node['id'] for node in small['nodes']}
        self.assertTrue(all(edge['source'] in nodes and edge['target'] in nodes for edge in small['edges']))
        self.assertLess(len(large['edges']), len(small['edges']) * 2.1)
        routes = {(edge['source'], edge['target'], edge['relation']) for edge in small['edges']}
        self.assertIn(('aux_feeder:0', 'secondary_pump:0', 'power'), routes)
        self.assertIn(('cdu:0', 'coldplate:0', 'secondary_supply'), routes)
        self.assertIn(('sensor:0', 'controller:0', 'feedback'), routes)
        self.assertIsNone(FacilityConfig(racks=16).description()['compute_performance_FLOPS'])


class ObservationTaskTests(unittest.TestCase):
    def test_telemetry_does_not_export_unmeasured_temperatures_or_fault_names(self):
        plant = CoupledFacility(FacilityConfig(racks=16), seed=92, scenario='combined')
        plant.step()
        packet = plant.public_telemetry()
        self.assertEqual(len(packet['temperature_indices']), 4)
        self.assertEqual(len(packet['temperature_c']), 4)
        self.assertFalse(any('fault' in key or 'truth' in key or 'severity' in key for key in packet))
        observer = FacilityObserver(plant.config)
        result = observer.update(packet)
        self.assertAlmostEqual(result['ever_measured'].mean(), .25)

    def test_role_access_duration_and_actual_completion(self):
        config = FacilityConfig(racks=16)
        plant, observer = CoupledFacility(config), FacilityObserver(config)
        task = inspection_tasks(config)[0]
        self.assertFalse(eligible('control', task))
        forbidden = InspectionTask('test', 'thermal', 0, 'thermal', 10, (0, 0), access='internal_energized')
        self.assertFalse(eligible('thermal', forbidden))
        scheduler = FixedQualifiedInspection(config)
        plant.step(); observer.update(plant.public_telemetry())
        scheduler.advance(plant, observer)
        self.assertEqual(observer.thermal_counts.sum(), 0)
        plant.step(); observer.update(plant.public_telemetry())
        completed = scheduler.advance(plant, observer)
        self.assertEqual(observer.thermal_counts.sum(), 1)
        self.assertTrue(all(event['role'] == event['kind'] and event['calibrated'] for event in completed))

    def test_bpid_reference_cannot_become_temperature_or_finish_measurement(self):
        config = FacilityConfig(racks=16)
        observer = FacilityObserver(config)
        before = observer.last_temperature.copy()
        observer.ingest_reference({'kind': 'apparent_ir_reference', 'calibrated': False, 'pixel_values': [0, 1]})
        np.testing.assert_array_equal(before, observer.last_temperature)
        task = inspection_tasks(config)[0]
        with self.assertRaises(ValueError):
            observer.complete_inspection(task, 'thermal', {'rack_index': 0, 'temperature_c': 80}, 10, calibrated=False)
        with self.assertRaises(ValueError):
            observer.complete_inspection(task, 'control', {'rack_index': 0, 'temperature_c': 80}, 10)
        self.assertEqual(observer.thermal_counts.sum(), 0)

    def test_same_packet_produces_same_estimate_without_simulator_access(self):
        config = FacilityConfig(racks=16)
        plant = CoupledFacility(config, seed=95, scenario='healthy')
        plant.step();packet = plant.public_telemetry()
        left, right = FacilityObserver(config), FacilityObserver(config)
        a, b = left.update(packet), right.update(packet)
        for key in ('predicted_temperature_c', 'hold_temperature_c', 'predicted_severity_proxy'):
            np.testing.assert_array_equal(a[key], b[key])
        self.assertFalse(any(isinstance(value, CoupledFacility) for value in left.__dict__.values()))

    def test_complete_episode_determinism_and_inspection_does_not_repair_process(self):
        config = FacilityConfig(racks=16, horizon_s=100)
        record, arrays = run_case(config, 96, 'healthy', 'feedback')
        repeated, other = run_case(config, 96, 'healthy', 'feedback')
        self.assertEqual(record, repeated)
        for key in arrays: np.testing.assert_array_equal(arrays[key], other[key])
        no_inspection = CoupledFacility(config, 96, 'healthy', 'feedback')
        for step in range(20):
            truth = no_inspection.step()
            np.testing.assert_array_equal(truth['rack_temperature_c'], arrays['rack_temperature_c'][step])


if __name__ == '__main__':
    unittest.main()
