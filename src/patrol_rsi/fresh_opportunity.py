"""Frozen B4-B1a inventory and coverage arithmetic; no fitting or selection."""
from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json

import numpy as np

from .causal_observation import CHANNELS
from .opportunity_data import FEATURE_NAMES, HORIZONS, KINDS

PROTOCOL_SHA = 'd049c0d39663ce35824bfe61059a00c7c48f030d543386511e7bda8c65c77e5e'
MANIFEST_SHA = '2580b46da58320e4349e149ab13b08f926d64b49ced57808cfc138cdc274b269'
WEIGHTS_SHA = '2dc8399ee79fed254fb4b12f39b95b65a98d2ec6f942155c158c31d76df21caa'
SPLITS = ('development', 'validation_calibration', 'validation_selection', 'final_test')
ROOTS = tuple(range(30, 1800, 30))
KIND_IDS = np.r_[np.zeros(128, int), np.ones(16, int), np.full(16, 2), np.full(16, 3)]


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def frozen_inputs(protocol_path, manifest_path):
    need(sha(protocol_path) == PROTOCOL_SHA, 'Frozen protocol SHA changed')
    need(sha(manifest_path) == MANIFEST_SHA, 'Frozen manifest SHA changed')
    protocol, manifest = read(protocol_path), read(manifest_path)
    expected = []
    for split, seeds in protocol['new_seed_groups'].items():
        for seed, fixture, start, noise in itertools.product(seeds,
                protocol['new_process_plan']['fixtures'], (335, 705), (.2, 1.)):
            scenario, duration = fixture
            expected.append({'id': f'{split}--{seed}--{scenario}--{duration}--{start}--{noise}',
                'split': split, 'group_seed': seed, 'physical_key': f'{seed}--{scenario}--{duration}--{start}',
                'specification': {'seed': seed, 'scenario': scenario, 'duration_s': duration,
                    'fault_start_s': start, 'noise_c': noise, 'estimator': 'history',
                    'control': 'feedback', 'resources': 'provisioned'}})
    need(manifest['manifest'] == expected and len(expected) == 1280, 'Exact ordered frozen conditions')
    need(protocol['new_process_plan']['roots_s'] == list(ROOTS), 'Frozen root grid')
    need(tuple(protocol['new_seed_groups']) == SPLITS, 'Four frozen splits')
    return protocol, expected


def condition_strata(base, targets, labels, events):
    """Counts keep missing endpoints separate, event IDs only evaluator-side."""
    times = np.asarray(ROOTS)[:, None] + np.asarray(HORIZONS)[None, :]
    ticks = np.minimum(times // 5 - 1, 359)
    result = []
    for k, kind in enumerate(KINDS):
        mask = KIND_IDS == k
        for h, horizon in enumerate(HORIZONS):
            a, ref, b = (targets[name][:, h, mask] for name in ('A', 'B_reference', 'B_action'))
            eid = labels['event_id'][ticks[:, h]][:, mask]
            retained = labels['retained_event_id'][ticks[:, h]][:, mask]
            def ids(values, good):
                return sorted(int(v) for v in np.unique(values[good]) if v > 0)
            own = [v for v in events if v['kind'] == kind]
            availability = {}
            for channel in CHANNELS:
                available = base[:, mask, FEATURE_NAMES.index(channel + '.available')] > 0
                availability[channel] = {'B_valid_available': int((available & (b >= 0)).sum()),
                    'B_valid_missing': int((~available & (b >= 0)).sum()),
                    'all_root_available': int(available.sum()), 'all_root_missing': int((~available).sum())}
            result.append({'kind': kind, 'horizon_s': horizon, 'total_samples': int(a.size),
                'A_valid': int((a >= 0).sum()), 'A_positive': int((a == 1).sum()),
                'A_negative': int((a == 0).sum()), 'B_reference_valid': int((ref >= 0).sum()),
                'B_reference_positive': int((ref == 1).sum()), 'B_action_valid': int((b >= 0).sum()),
                'B_action_positive': int((b == 1).sum()), 'B_action_negative': int((b == 0).sum()),
                'B_action_invalid': int((b < 0).sum()),
                'A_positive_without_legal_B': int(((a == 1) & (b < 0)).sum()),
                'A_positive_B_reference_negative': int(((a == 1) & (ref == 0)).sum()),
                'B_positive_A_negative': int(((b == 1) & (a == 0)).sum()),
                'A_positive_event_ids': ids(eid, a == 1),
                'B_positive_event_ids': ids(eid, b == 1),
                'B_positive_retained_event_ids': ids(retained, b == 1),
                'physical_event_count': len(own),
                'retained_event_count': sum(v['retained_at_least_15s'] for v in own),
                'availability_in_valid_B_samples': availability})
    return result


def electrical_diagnostic(arrays, spec, events, strata):
    pod = spec['seed'] % 16
    contact = arrays['contact_temperature_c'][:, pod]
    start, end = spec['fault_start_s'], spec['fault_start_s'] + spec['duration_s']
    integration_start = np.arange(360) * 5
    during = (integration_start >= start) & (integration_start < end)
    after = integration_start >= end
    own = [v for v in events if v['kind'] == 'electrical']
    return {'faulted_pod': pod, 'threshold_c': 60., 'injection_start_s': start,
        'injection_end_s': end, 'peak_faulted_contact_during_c': float(contact[during].max()),
        'peak_faulted_contact_after_c': float(contact[after].max()) if after.any() else None,
        'peak_all_contacts_c': float(arrays['contact_temperature_c'].max()),
        'physical_positive_ticks_faulted_pod': int((contact > 60).sum()),
        'physical_event_count': len(own), 'retained_event_count': sum(v['retained_at_least_15s'] for v in own),
        'horizons': [v for v in strata if v['kind'] == 'electrical'],
        'time_semantics': 'Fault applied at integration start t; stored state is end t+5. Original label tick convention retained.'}


def coverage_gate(row, minimum):
    actual = {'positive_seed_groups': len(row['B_positive_seed_groups']),
        'positive_numeric_processes': row['B_positive_numeric_processes'],
        'distinct_retained_positive_events': row['distinct_B_positive_retained_events'],
        'valid_negative_seed_groups': len(row['B_negative_seed_groups'])}
    deficits = {key: max(0, required - actual[key]) for key, required in minimum.items()}
    return {'actual': actual, 'minimum': minimum, 'deficits': deficits, 'passed': not any(deficits.values())}


def fingerprint_conflicts(items):
    groups = {}
    for item in items:
        groups.setdefault(item['numeric_process_fingerprint'], {}).setdefault(item['split'], []).append(item['id'])
    return [{'fingerprint': fp, 'splits': values} for fp, values in groups.items() if len(values) > 1]
