"""M9 mechanism suite; no new neural training or site-control deployment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
try:
    import resource
except ImportError:  # Local Windows inspection does not need POSIX accounting.
    resource = None

from .facility import CoupledFacility, FacilityConfig, SCENARIOS, facility_topology
from .facility_observation import FacilityObserver, FixedQualifiedInspection, rack_positions


def contract():
    return {'configuration': asdict(FacilityConfig()), 'scenarios': list(SCENARIOS),
            'controls': ['feedback', 'fixed_commands'], 'paired_simulation_seeds': list(range(5)),
            'episodes': 150, 'steps_each': 360, 'training_tasks': 0,
            'display_case': {'scenario': 'combined', 'control': 'feedback', 'seed': 0},
            'scale_racks': [32, 128, 512, 2048, 4096], 'scale_repetitions': 3,
            'scope': 'Synthetic reduced-order multi-system verification; no calibrated twin, no FLOPS claim, no new VAE/PPO/RSI inference',
            'risk': 'Uncalibrated impact-weighted severity proxy; thermal/contact/service/flow components retained',
            'observation': 'Sparse remote sensors plus qualified completed inspections; observer has no plant reference or hidden-fault identifier',
            'raw_trace_storage': 'All numeric state arrays retained as private compressed NPZ; public scalar traces, exact state hashes and selected demo arrays',
            'B_PID': 'Explicit uncalibrated apparent-reference adapter; source report inspected, no checkpoint available or executed',
            'standard_scope': '2026 official publication metadata verified; no compliance or unpublished-clause parameter claim'}


def arrays_digest(values):
    digest = hashlib.sha256()
    for key in sorted(values):
        array = np.asarray(values[key])
        digest.update(key.encode() + str(array.shape).encode() + array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def run_case(config, seed, scenario, control):
    plant = CoupledFacility(config, seed, scenario, control)
    observer = FacilityObserver(config)
    inspections = FixedQualifiedInspection(config)
    truth_keys = ('rack_temperature_c', 'secondary_supply_c', 'primary_supply_c', 'contact_temperature_c',
                  'requested_it_kw', 'delivered_it_kw', 'rack_flow_kg_s', 'secondary_dp_kpa',
                  'pump_command', 'pump_speed', 'fan_speed', 'throttle', 'severity_proxy')
    observed_keys = ('hold_temperature_c', 'predicted_temperature_c', 'hold_severity_proxy',
                     'predicted_severity_proxy', 'measurement_age_s', 'inspection_age_s',
                     'thermal_inspection_counts')
    history = {key: [] for key in truth_keys + observed_keys}
    scalar = []
    initial_energy = plant.stored_energy_kj()
    for _ in range(round(config.horizon_s / config.dt_s)):
        truth = plant.step()
        packet = plant.public_telemetry()
        observer.update(packet)
        inspections.advance(plant, observer)
        obs = observer.readout(packet)
        for key in truth_keys: history[key].append(np.asarray(truth[key]).copy())
        for key in observed_keys: history[key].append(np.asarray(obs[key]).copy())
        true_temperature = truth['rack_temperature_c']
        row = {key: float(truth[key]) for key in ('time_s', 'source_power_kw', 'it_power_kw', 'aux_power_kw',
                'conversion_loss_kw', 'contact_loss_kw', 'rejected_heat_kw', 'stored_energy_kj',
                'electric_residual_kw', 'energy_residual_kj', 'unserved_it_kw', 'impact_weighted_severity')}
        for method, key in [('hold', 'hold_temperature_c'), ('nominal_model', 'predicted_temperature_c')]:
            error = obs[key] - true_temperature
            row[f'{method}_mse_c2'] = float(np.mean(error ** 2))
            row[f'{method}_weighted_mse_c2'] = float(np.sum(plant.criticality * error ** 2) / plant.criticality.sum())
            risk_key = 'hold_severity_proxy' if method == 'hold' else 'predicted_severity_proxy'
            row[f'{method}_severity_mse'] = float(np.mean((obs[risk_key] - truth['severity_proxy']) ** 2))
        row.update(max_rack_c=float(true_temperature.max()), mean_rack_c=float(true_temperature.mean()),
                   max_contact_c=float(truth['contact_temperature_c'].max()), primary_c=float(truth['primary_supply_c']),
                   mean_flow_kg_s=float(truth['rack_flow_kg_s'].mean()),
                   thermal_degree_excess_sum_c=float(np.maximum(true_temperature - config.thermal_reference_c, 0).sum()),
                   true_hot_racks=int((true_temperature > config.thermal_reference_c).sum()),
                   measurement_coverage=float(obs['ever_measured'].mean()),
                   mobile_thermal_coverage=float((obs['thermal_inspection_counts'] > 0).mean()),
                   max_mobile_inspection_age_s=float(obs['inspection_age_s'].max()),
                   severity_weighted_unverified=float(np.mean(truth['severity_proxy'] * np.minimum(obs['inspection_age_s'] / config.revisit_reference_s, 4))))
        scalar.append(row)
    arrays = {key: np.stack(value) for key, value in history.items()}
    arrays['time_s'] = np.asarray([row['time_s'] for row in scalar])
    arrays['criticality'] = plant.criticality.copy()
    arrays['rack_positions_m'] = rack_positions(config)
    means = lambda key: float(np.mean([row[key] for row in scalar]))
    metrics = {'mean_impact_weighted_severity': means('impact_weighted_severity'),
               'peak_rack_temperature_c': max(row['max_rack_c'] for row in scalar),
               'peak_contact_temperature_c': max(row['max_contact_c'] for row in scalar),
               'peak_primary_temperature_c': max(row['primary_c'] for row in scalar),
               'unserved_IT_kwh': sum(row['unserved_it_kw'] for row in scalar) * config.dt_s / 3600,
               'input_energy_kwh': sum(row['source_power_kw'] for row in scalar) * config.dt_s / 3600,
               'thermal_degree_hours_per_rack': sum(row['thermal_degree_excess_sum_c'] for row in scalar) * config.dt_s / (3600 * config.racks),
               'max_electric_residual_kw': max(abs(row['electric_residual_kw']) for row in scalar),
               'max_energy_residual_kj': max(abs(row['energy_residual_kj']) for row in scalar),
               'mean_severity_weighted_unverified': means('severity_weighted_unverified'),
               'hold_temperature_rmse_c': np.sqrt(means('hold_mse_c2')),
               'nominal_model_temperature_rmse_c': np.sqrt(means('nominal_model_mse_c2')),
               'hold_temperature_weighted_rmse_c': np.sqrt(means('hold_weighted_mse_c2')),
               'nominal_model_temperature_weighted_rmse_c': np.sqrt(means('nominal_model_weighted_mse_c2')),
               'hold_severity_rmse': np.sqrt(means('hold_severity_mse')),
               'nominal_model_severity_rmse': np.sqrt(means('nominal_model_severity_mse')),
               'final_remote_or_mobile_measurement_coverage': scalar[-1]['measurement_coverage'],
               'final_mobile_thermal_coverage': scalar[-1]['mobile_thermal_coverage'],
               'max_mobile_inspection_age_s': max(row['max_mobile_inspection_age_s'] for row in scalar),
               'effective_inspections': sum(agent['completed'] for agent in inspections.agents.values())}
    fault_rows = [row for row in scalar if config.fault_start_s < row['time_s'] <= config.fault_end_s]
    recovery_rows = [row for row in scalar if row['time_s'] > config.fault_end_s]
    metrics['fault_interval_mean_severity'] = float(np.mean([row['impact_weighted_severity'] for row in fault_rows])) if fault_rows else None
    metrics['recovery_mean_severity'] = float(np.mean([row['impact_weighted_severity'] for row in recovery_rows])) if recovery_rows else None
    record = {'seed': seed, 'scenario': scenario, 'control': control, 'configuration': asdict(config),
              'initial_energy_kj': initial_energy, 'metrics': {key: float(value) if value is not None else None for key, value in metrics.items()},
              'scalar_trace': scalar, 'inspection_events': inspections.events,
              'numeric_arrays_sha256': arrays_digest(arrays), 'numeric_shapes': {key: list(value.shape) for key, value in arrays.items()},
              'risk_scope': 'Uncalibrated severity and information-age proxy; inspection is measurement, not physical repair',
              'baselines': ['measurement_hold', 'nominal_physical_predictor'], 'new_neural_inference': False}
    return record, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'contract.json').write_text(json.dumps(contract(), indent=2))
    config = FacilityConfig(horizon_s=600 if args.smoke else 1800)
    (args.output / 'topology.json').write_text(json.dumps(facility_topology(config), indent=2))
    cases = args.output / 'cases'; cases.mkdir()
    private = args.output / 'private-traces'; private.mkdir()
    scenarios = ('healthy', 'combined') if args.smoke else SCENARIOS
    seeds = (0,) if args.smoke else range(5)
    archive_index = []
    started = time.perf_counter()
    for scenario in scenarios:
        for control in ('feedback', 'fixed_commands'):
            for seed in seeds:
                record, arrays = run_case(config, seed, scenario, control)
                name = f'{scenario}--{control}--seed-{seed}'
                numeric_path = private / f'{name}.npz'
                np.savez_compressed(numeric_path, **arrays)
                numeric_sha = hashlib.sha256(numeric_path.read_bytes()).hexdigest()
                record['private_numeric_file'] = {'path': str(numeric_path), 'bytes': numeric_path.stat().st_size, 'sha256': numeric_sha}
                encoded = (json.dumps(record) + '\n').encode()
                destination = cases / f'{name}.json.gz'
                destination.write_bytes(gzip.compress(encoded, compresslevel=6, mtime=0))
                archive_index.append({'case': name, 'record': str(destination.relative_to(args.output)),
                                      'stored_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                                      'uncompressed_sha256': hashlib.sha256(encoded).hexdigest(),
                                      'numeric_arrays_sha256': record['numeric_arrays_sha256'], 'private_numeric_file': record['private_numeric_file']})
                if scenario == 'combined' and control == 'feedback' and seed == 0:
                    (args.output / 'demo.json.gz').write_bytes(gzip.compress(json.dumps({key: value.tolist() for key, value in arrays.items()}).encode(), mtime=0))
                print(json.dumps({'case': name, 'metrics': record['metrics']}), flush=True)
    (args.output / 'case-index.json').write_text(json.dumps(archive_index, indent=2))
    scales = []
    if not args.smoke:
        for racks in contract()['scale_racks']:
            for repeat in range(3):
                scale_config = FacilityConfig(racks=racks)
                tick = time.perf_counter()
                topology = facility_topology(scale_config)
                graph_seconds = time.perf_counter() - tick
                plant = CoupledFacility(scale_config, seed=repeat, scenario='combined')
                tick = time.perf_counter()
                peak_residual = 0.0
                for _ in range(360):
                    row = plant.step()
                    peak_residual = max(peak_residual, abs(row['energy_residual_kj']))
                scales.append({'racks': racks, 'repeat': repeat, 'pods': scale_config.pods,
                               'nominal_IT_MW': racks * scale_config.rated_rack_kw / 1000,
                               'nodes': len(topology['nodes']), 'edges': len(topology['edges']),
                               'graph_seconds': graph_seconds, 'simulation_360_steps_seconds': time.perf_counter() - tick,
                               'max_energy_residual_kj': peak_residual,
                               'model_numeric_state_bytes': sum(value.nbytes for value in plant.__dict__.values() if isinstance(value, np.ndarray))
                                   + sum(value.nbytes for value in plant._last.values() if isinstance(value, np.ndarray)),
                               'process_peak_RSS_kib_cumulative': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None})
    result = {'smoke': args.smoke, 'case_count': len(archive_index), 'scale_trials': scales,
              'seconds': time.perf_counter() - started, 'training_tasks': 0, 'job_id': os.environ.get('SLURM_JOB_ID'),
              'raw_traces_retained_privately': True, 'scope': contract()['scope']}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2))
    print('M9_SUITE_COMPLETE', len(archive_index), 'cases;', len(scales), 'scale trials', flush=True)


if __name__ == '__main__':
    main()
