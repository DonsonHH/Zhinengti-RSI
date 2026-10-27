"""Frozen M10 splits and causally generated public observations."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .facility import CoupledFacility, FacilityConfig, SCENARIOS
from .facility_observation import FixedQualifiedInspection, eligible, inspection_tasks, rack_positions
from .estimation import ObservationState, FEATURES, radial_knn, bounded_gp
from .m9 import arrays_digest

ATOMS = tuple(s for s in SCENARIOS if s != 'combined')
COMBINATIONS = ('tower_derate+sensor_bias', 'grid_derate+coldplate_fouling',
                'aux_feeder_loss+cdu_fouling')
CONTROLS = ('feedback', 'fixed_commands')


def contract():
    return {'target': 'current rack temperature in degC', 'physical_step_s': 5,
            'sample_every_s': 30, 'frames_per_case': 60, 'history_frames': 4,
            'features': list(FEATURES), 'racks': 128,
            'train_seeds': list(range(100, 108)), 'validation_seeds': [200, 201],
            'test_seeds': [300, 301, 302], 'gw_seeds': [400, 401, 402],
            'train_atoms': list(ATOMS), 'unseen_combinations': list(COMBINATIONS),
            'controls': list(CONTROLS), 'train_cases': 224, 'validation_cases': 56,
            'test_cases': 306, 'gw_cases': 12,
            'sensor_conditions': {'nominal': {'stride': 4, 'noise_c': .2},
                                  'noise': {'stride': 4, 'noise_c': 1.},
                                  'sparse': {'stride': 8, 'noise_c': .2}},
            'inspection_noise_c': .1, 'gw_racks': 3200, 'gw_IT_nameplate_MW': 800,
            'gw_supply_design_budget_MW': 1000,
            'gw_budget_semantics': 'Observed input checked against design budget; not an implemented global power limiter',
            'training_seeds': [0, 1, 2], 'updates_each': 2000, 'batch_size': 1024,
            'validation_every': 200, 'learning_rate': .001, 'KL_weight': .01,
            'methods': ['hold', 'linear', 'nominal_model', 'rknn', 'gp64', 'cvae_current', 'cvae_history'],
            'scope': 'New-domain CVAE and explicit baselines; no RSI, policy optimization, probability calibration or site validation'}


def case_specs(phase):
    if phase == 'development':
        return [{'split': 'development', 'condition': 'nominal', 'scenario': s,
                 'seed': 900, 'control': 'feedback'} for s in ('healthy', COMBINATIONS[-1])]
    rows = []
    if phase == 'train':
        groups = [('train', range(100, 108), ATOMS, ('nominal',)),
                  ('validation', (200, 201), ATOMS, ('nominal',))]
    elif phase == 'test':
        groups = [('test', (300, 301, 302), ATOMS + COMBINATIONS, ('nominal', 'noise', 'sparse')),
                  ('gw', (400, 401, 402), ('healthy', COMBINATIONS[-1]), ('nominal',))]
    else:
        raise ValueError('Unknown data phase')
    for split, seeds, scenarios, conditions in groups:
        for condition in conditions:
            for scenario in scenarios:
                for control in CONTROLS:
                    for seed in seeds:
                        rows.append(dict(split=split, condition=condition, scenario=scenario,
                                         control=control, seed=seed))
    return rows


def case_name(spec):
    return '--'.join(str(spec[key]) for key in ('split', 'condition', 'scenario', 'control', 'seed'))


class CompositeFacility(CoupledFacility):
    """Combine declared independent synthetic fault modifiers inside the plant."""
    def __init__(self, config, seed, scenario, control):
        self.components = scenario.split('+')
        if any(part not in ATOMS for part in self.components):
            raise ValueError('Only frozen atom combinations')
        super().__init__(config, seed, 'healthy', control)

    def _faults(self, t):
        rows = []
        for component in self.components:
            self._scenario = component
            rows.append(super()._faults(t))
        self._scenario = 'healthy'
        merged = {}
        for key in rows[0]:
            if key == 'controller_stuck':
                merged[key] = np.logical_or.reduce([r[key] for r in rows])
            elif key in ('sensor_bias', 'load_addition'):
                merged[key] = sum(r[key] for r in rows)
            else:
                merged[key] = np.prod([r[key] for r in rows], axis=0)
        return merged


class CompletionSink:
    def complete_inspection(self, task, role, measurement, time_s):
        if not eligible(role, task):
            raise ValueError('Unqualified inspection in data generator')


def generate_case(spec, baselines=True):
    config = FacilityConfig(racks=3200 if spec['split'] == 'gw' else 128)
    plant = CompositeFacility(config, spec['seed'], spec['scenario'], spec['control'])
    scheduler, state = FixedQualifiedInspection(config), ObservationState(config)
    tasks = {t.task_id: t for t in inspection_tasks(config)}
    rng = np.random.default_rng(100000 + spec['seed'])
    inspection_rng = np.random.default_rng(200000 + spec['seed'])
    condition = contract()['sensor_conditions'][spec['condition']]
    arrays = {key: [] for key in ('features', 'temperature_c', 'fresh', 'time_s')}
    if baselines:
        for name in contract()['methods'][:5]:
            arrays['pred_' + name] = []
            arrays['seconds_' + name] = []
    common_seconds = []
    events, max_power, max_energy, max_input = [], 0., 0., 0.
    for tick in range(360):
        truth = plant.step()
        packet = plant.public_telemetry()
        keep = packet['temperature_indices'] % condition['stride'] == 0
        packet['temperature_indices'] = packet['temperature_indices'][keep]
        packet['temperature_c'] = (packet['temperature_c'][keep] +
                                    rng.normal(0, condition['noise_c'], keep.sum()))
        raw_events = scheduler.advance(plant, CompletionSink())
        completed = []
        for event in raw_events:
            event = dict(event, measurement=dict(event['measurement']))
            if event['kind'] == 'thermal':
                event['measurement']['temperature_c'] += inspection_rng.normal(0, .1)
            events.append(event)
            completed.append({'time_s': event['time_s'],
                              'completion': (tasks[event['task_id']], event['role'],
                                             event['measurement'], event['time_s'])})
        start = time.perf_counter()
        observation = state.update(packet, completed)
        common_seconds.append(time.perf_counter() - start)
        max_power = max(max_power, abs(truth['electric_residual_kw']))
        max_energy = max(max_energy, abs(truth['energy_residual_kj']))
        max_input = max(max_input, truth['source_power_kw'])
        if (tick + 1) % 6:
            continue
        arrays['features'].append(state.features(packet))
        arrays['temperature_c'].append(truth['rack_temperature_c'])
        arrays['fresh'].append(observation['fresh_measurement'])
        arrays['time_s'].append(truth['time_s'])
        if baselines:
            # Time methods individually on the same CPU and prepared state.
            obs = state.observer
            positions = rack_positions(config)
            fns = {'hold': lambda: obs.last_temperature.copy(),
                   'linear': lambda: obs.last_temperature + state.velocity * np.minimum(
                       np.where(obs.temperature_stamp >= 0, packet['time_s'] - obs.temperature_stamp, 0), 120),
                   'nominal_model': lambda: obs.predicted_temperature.copy(),
                   'rknn': lambda: radial_knn(obs.last_temperature, obs.temperature_stamp, positions, packet['time_s']),
                   'gp64': lambda: bounded_gp(obs.last_temperature, obs.temperature_stamp, positions, packet['time_s'])}
            for name, fn in fns.items():
                start = time.perf_counter()
                arrays['pred_' + name].append(fn())
                arrays['seconds_' + name].append(time.perf_counter() - start)
    result = {key: np.stack(value) for key, value in arrays.items()}
    result['criticality'] = plant.criticality
    result['common_update_seconds'] = np.asarray(common_seconds)
    metadata = dict(spec, configuration=asdict(config), effective_inspection_events=events,
                    maximum_electric_residual_kw=max_power, maximum_energy_residual_kj=max_energy,
                    peak_source_input_MW=max_input / 1000,
                    IT_nameplate_MW=config.racks * config.rated_rack_kw / 1000,
                    global_supply_budget_MW=1000 if spec['split'] == 'gw' else None,
                    budget_limiter_implemented=False)
    return result, metadata


def deterministic_digest(arrays):
    return arrays_digest({k: v for k, v in arrays.items() if 'seconds' not in k})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('development', 'train', 'test'), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'cases').mkdir()
    (args.output / 'contract.json').write_text(json.dumps(contract(), indent=2))
    index = []
    for spec in case_specs(args.phase):
        arrays, metadata = generate_case(spec, baselines=args.phase != 'train')
        name = case_name(spec)
        path = args.output / 'cases' / (name + '.npz')
        np.savez_compressed(path, **arrays)
        item = dict(metadata, case=name, path=str(path.relative_to(args.output)),
                    bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    numeric_digest=deterministic_digest(arrays))
        index.append(item)
        print(json.dumps({'generated': name, 'peak_input_MW': item['peak_source_input_MW']}), flush=True)
    (args.output / 'index.json').write_text(json.dumps(index, indent=2))
    (args.output / 'summary.json').write_text(json.dumps({'phase': args.phase, 'cases': len(index),
                                                        'job_id': os.environ.get('SLURM_JOB_ID')}, indent=2))


if __name__ == '__main__':
    main()
