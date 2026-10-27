"""Re-run every M9 case and check raw arrays plus independent balances."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from patrol_rsi.facility import FacilityConfig, SCENARIOS, facility_topology
from patrol_rsi.m9 import arrays_digest, contract, run_case


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(json.loads((args.results / 'contract.json').read_text()) == contract(), 'Frozen protocol mismatch')
    summary = json.loads((args.results / 'summary.json').read_text())
    require(not summary['smoke'] and summary['case_count'] == 150 and len(summary['scale_trials']) == 15, 'Full suite missing')
    index = json.loads((args.results / 'case-index.json').read_text())
    require(len(index) == 150 and len({row['case'] for row in index}) == 150, 'Unique case coverage')
    expected = {f'{scenario}--{control}--seed-{seed}' for scenario in SCENARIOS for control in ('feedback', 'fixed_commands') for seed in range(5)}
    require({row['case'] for row in index} == expected, 'Frozen scenario/seed set')
    config = FacilityConfig()
    topology = json.loads((args.results / 'topology.json').read_text())
    require(topology == facility_topology(config), 'Sparse topology')
    prefix = {}
    rows = []
    for item in index:
        compressed = (args.results / item['record']).read_bytes()
        original = gzip.decompress(compressed)
        require(hashlib.sha256(compressed).hexdigest() == item['stored_sha256'], 'Stored record bytes')
        require(hashlib.sha256(original).hexdigest() == item['uncompressed_sha256'], 'Original JSON bytes')
        record = json.loads(original)
        require(record['configuration'] == asdict(config), 'Configuration change')
        private = Path(record['private_numeric_file']['path'])
        require(private.resolve().is_relative_to((args.results / 'private-traces').resolve()), 'Private path out of run')
        require(private.stat().st_size == item['private_numeric_file']['bytes'], 'Private trace length')
        require(hashlib.sha256(private.read_bytes()).hexdigest() == item['private_numeric_file']['sha256'], 'Private trace fingerprint')
        repeated, expected_arrays = run_case(config, record['seed'], record['scenario'], record['control'])
        for key in repeated:
            require(repeated[key] == record[key], f'Exact case replay differs: {item["case"]}/{key}')
        with np.load(private, allow_pickle=False) as arrays:
            require(set(arrays.files) == set(expected_arrays), 'Raw state coverage')
            for key, value in expected_arrays.items():
                require(np.array_equal(arrays[key], value), f'Raw numeric state differs: {key}')
        require(arrays_digest(expected_arrays) == item['numeric_arrays_sha256'], 'Exact state digest')
        scalar = record['scalar_trace']
        source = np.asarray([row['source_power_kw'] for row in scalar])
        stored = np.asarray([row['stored_energy_kj'] for row in scalar])
        rejected = np.asarray([row['rejected_heat_kw'] for row in scalar])
        independent_stored = (config.chip_capacity_kj_k * expected_arrays['rack_temperature_c'].sum(axis=1)
            + config.secondary_capacity_kj_k * expected_arrays['secondary_supply_c'].sum(axis=1)
            + config.primary_capacity_per_pod_kj_k * config.pods * expected_arrays['primary_supply_c']
            + config.contact_capacity_kj_k * expected_arrays['contact_temperature_c'].sum(axis=1))
        require(np.allclose(stored, independent_stored, rtol=0, atol=1e-7), 'Thermal storage from raw temperatures')
        residual = stored - np.r_[record['initial_energy_kj'], stored[:-1]] - config.dt_s * (source - rejected)
        require(np.max(np.abs(residual)) < 1e-4, 'Control-volume energy imbalance')
        electrical = source - np.asarray([row['it_power_kw'] + row['aux_power_kw'] + row['conversion_loss_kw'] + row['contact_loss_kw'] for row in scalar])
        require(np.max(np.abs(electrical)) < 1e-6, 'Independent electric balance')
        for event in record['inspection_events']:
            require(event['role'] == event['kind'] and event['calibrated'] and event['access'] == 'external_sensing', 'Task qualification/measurement boundary')
            require(event['time_s'] >= event['service_s'], 'Cannot finish before service time')
        samples_before_fault = round(config.fault_start_s / config.dt_s)
        early = {key: value if key in ('criticality', 'rack_positions_m') else value[:samples_before_fault]
                 for key, value in expected_arrays.items()}
        stamp = arrays_digest(early)
        key = (record['control'], record['seed'])
        prefix.setdefault(key, stamp)
        require(prefix[key] == stamp, 'Counterfactual cases differ before fault begins')
        rows.append({'case': item['case'], 'steps': len(scalar), 'max_independent_electric_residual_kw': float(np.max(np.abs(electrical))),
                     'max_independent_energy_residual_kj': float(np.max(np.abs(residual))), 'raw_state_and_replay_identical': True})
    for size in contract()['scale_racks']:
        trials = [row for row in summary['scale_trials'] if row['racks'] == size]
        require(len(trials) == 3, 'Scale repetitions')
        graph = facility_topology(FacilityConfig(racks=size))
        for row in trials:
            require(row['nodes'] == len(graph['nodes']) and row['edges'] == len(graph['edges']), 'Scale graph counts')
            require(row['nominal_IT_MW'] == size * config.rated_rack_kw / 1000 and row['max_energy_residual_kj'] < 1e-4, 'Scale units/balance')
    result = {'cases_replayed': 150, 'numeric_microsteps_replayed': 150 * 360,
              'full_private_numeric_traces_read': 150, 'paired_pre_fault_state_equivalence': True,
              'independent_electric_and_control_volume_balance_passed': True,
              'scale_trials_verified': 15, 'topology_nodes': len(topology['nodes']), 'topology_edges': len(topology['edges']),
              'maximum_electric_residual_kw': max(row['max_independent_electric_residual_kw'] for row in rows),
              'maximum_energy_residual_kj': max(row['max_independent_energy_residual_kj'] for row in rows),
              'case_audits': rows, 'scope': 'Implementation and internal-consistency audit, not measured-plant calibration or national-standard compliance'}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
    print('M9_AUDIT_VERIFIED', len(rows), 'cases; 54000 steps and all private state arrays', flush=True)


if __name__ == '__main__':
    main()
