"""Causal feature and separate evaluator-label contracts for B4-B0.

No model is trained here. Public feature extraction never receives process
truth, anomaly windows, scenario identities, seeds, or B4-C value labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .causal_observation import CHANNELS, validate_public
from .information_audit import KINDS, effective_confirmability, reference_signals


HORIZONS = (30, 60, 120)
SOURCE_CODES = {name: i + 1 for i, name in enumerate((
    'qualified_reference', 'remote_or_qualified_temperature', 'remote_flow', 'qualified_flow',
    'qualified_contact', 'qualified_sensor_error', 'causal_nominal_model',
    'frozen_cvae_prior_mean', 'prior_mean_plus_completed_measurements'))}
CHANNEL_FIELDS = ('available', 'value', 'time_s', 'age_s', 'context_cutoff_s', 'source_code')
BASE_FEATURES = ('root_time_s', 'thermal', 'electrical', 'cooling', 'control',
    'pod', 'cdu_branch', 'power_branch', 'weight_scaled', 'service_s', 'deadline_s',
    'last_check_available', 'last_check_s', 'inspection_age_s', 'already_measured_alarm',
    'temperature_velocity_c_s', 'public_requested_pod_kw', 'public_delivered_pod_kw',
    'public_pump_command', 'public_secondary_supply_c', 'public_primary_supply_c', 'telemetry_age_s')
HISTORY_FIELDS = ('available', 'value', 'time_s', 'age_s', 'alarm')
FEATURE_NAMES = BASE_FEATURES + tuple(f'{channel}.{name}' for channel in CHANNELS for name in CHANNEL_FIELDS) + tuple(
    f'qualified_history_{i}.{name}' for i in range(3) for name in HISTORY_FIELDS)
ACTION_FEATURE_NAMES = ('horizon_s', 'completion_time_s', 'legal_completion_available',
    'optional_check', 'planned_dispatch_offset_s', 'planned_service_and_travel_s',
    'planned_Tour_target_offset_s', 'complete_return_slack_s', 'legal_completion_count',
    'qualified_executor_available', 'return_slack_available', 'bounded_trend_at_completion_c')


def need(value, message):
    if not value:
        raise ValueError(message)


@dataclass
class FeatureHistory:
    """Only monotonically arriving validated public observations are accepted."""
    last_time: float = 0.
    history: list = field(default_factory=lambda: [[] for _ in range(176)])
    frames: int = 0
    available_channels: np.ndarray = field(default_factory=lambda: np.zeros((4, len(CHANNELS)), dtype=np.int64))
    total_channels: np.ndarray = field(default_factory=lambda: np.zeros((4, len(CHANNELS)), dtype=np.int64))
    timestamp_readings_checked: int = 0

    def append(self, bundle):
        validate_public(bundle)
        now, tasks = bundle['time_s'], bundle['tasks']
        need(now > self.last_time and len(tasks) == 176, 'Ordered original 176-task public histories required')
        self.last_time = now
        self.frames += 1
        result = []
        packet = bundle['telemetry']
        need(0 <= packet['time_s'] <= now, 'Future or negative public telemetry')
        for row in tasks:
            j, kind, pod = row['task'], row['kind'], row['pod']
            role = KINDS.index(kind)
            need(row['required_role'] == kind and 0 <= pod < 16, 'Professional role and Pod membership required')
            q = row['qualified']
            if q['available']:
                history = self.history[j]
                if not history or q['time_s'] > history[0][0]:
                    history.insert(0, (q['time_s'], q['value'], float(row['already_alarm'])))
                    del history[3:]
                else:
                    need(q['time_s'] == history[0][0] and q['value'] == history[0][1],
                         'Qualified observation changed without a newer measurement')
            known = row['last_check_s'] >= 0
            values = [now, *(float(kind == v) for v in KINDS), pod, pod, pod,
                row['weight_scaled'], row['service_s'], row['deadline_s'], float(known),
                row['last_check_s'] if known else 0., now - row['last_check_s'] if known else 0.,
                float(row['already_alarm']), row['velocity_c_s'],
                sum(packet['public_requested_it_kw'][pod * 8:(pod + 1) * 8]),
                sum(packet['public_delivered_it_kw'][pod * 8:(pod + 1) * 8]),
                packet['pod_pump_command'][pod], packet['secondary_supply_c'][pod],
                packet['primary_supply_c'], now - packet['time_s']]
            for channel in CHANNELS:
                reading = row[channel]
                index = CHANNELS.index(channel)
                self.total_channels[role, index] += 1
                self.available_channels[role, index] += int(reading['available'])
                if reading['available']:
                    need(reading['source'] in SOURCE_CODES, 'Unexpected public channel source')
                    need(0 <= reading['context_cutoff_s'] <= reading['time_s'] <= now and
                         abs(reading['age_s'] - now + reading['time_s']) < 1e-8,
                         'Feature reading contains a future or inconsistent timestamp')
                    values.extend((1., reading['value'], reading['time_s'], reading['age_s'],
                                   reading['context_cutoff_s'], SOURCE_CODES[reading['source']]))
                    self.timestamp_readings_checked += 1
                else:
                    values.extend((0.,) * len(CHANNEL_FIELDS))
            for i in range(3):
                if i < len(self.history[j]):
                    when, value, alarm = self.history[j][i]
                    need(when <= now, 'Future qualified history')
                    values.extend((1., value, when, now - when, alarm))
                else:
                    values.extend((0.,) * len(HISTORY_FIELDS))
            result.append(values)
        array = np.asarray(result, dtype=np.float32)
        need(array.shape == (176, len(FEATURE_NAMES)) and np.isfinite(array).all(), 'Finite closed public feature matrix')
        return array


def completion_lookup(geometry, tour_record=None):
    """Exact future completion opportunities, qualified Tour or one extra CHECK.

    Known deterministic timings come from old public plans; measurements and
    outcome fields in the original Tour record are never read by this function.
    """
    rows = []
    for i in range(len(geometry['task'])):
        rows.append({'task': int(geometry['task'][i]), 'agent': int(geometry['agent'][i]),
            'dispatch_s': float(geometry['dispatch_time_s'][i]), 'finish_s': float(geometry['finish_s'][i]),
            'tour_finish_s': float(geometry['tour_target_finish_s'][i]),
            'return_slack_s': float(geometry['return_slack_s'][i]), 'optional': True, 'geometry_index': i})
    if tour_record is not None:
        # Only assignments' public route/service information and inspection
        # completion times/identities. No alarm or measurement values are used.
        finishes = {(v['agent'], v['task'], v['time_s']): v for v in tour_record['inspection_events']}
        by_pair = {}
        for agent, task, finish in finishes:
            by_pair.setdefault((agent, task), []).append(finish)
        for times in by_pair.values():
            times.sort()
        for assignment in tour_record['assignments']:
            dispatch = assignment['time_s']; j = assignment['task']; a = assignment['agent']
            future = by_pair[(a, j)]
            finish = next((v for v in future if v > dispatch), None)
            if finish is None:
                continue
            expected = dispatch + math.ceil((assignment['distance_m'] + assignment['service_s']) / 5 - 1e-10) * 5
            need(finish == expected, 'Saved Tour timing differs from deterministic public route/service cost')
            rows.append({'task': j, 'agent': a, 'dispatch_s': dispatch, 'finish_s': expected,
                'tour_finish_s': expected, 'return_slack_s': 0., 'optional': False, 'geometry_index': -1})
    rows.sort(key=lambda v: (v['finish_s'], v['task'], v['dispatch_s'], v['agent'], v['optional']))
    for row in rows:
        expected = 0 if row['task'] < 128 else 1 if row['task'] < 144 else 2 if row['task'] < 160 else 3
        actual = 0 if row['agent'] < 12 else 1 if row['agent'] < 14 else 2 if row['agent'] < 16 else 3
        need(expected == actual and 0 <= row['agent'] < 18, 'Unqualified future completion')
    return rows


def legal_completions(root, horizon, rows):
    """Choose the earliest publicly planned legal dispatch, no outcome ranking."""
    finish = root + horizon
    selected = [None] * 176
    count = np.zeros(176, dtype=np.int16)
    if finish > 1800:
        return selected, count
    for row in rows:
        if row['finish_s'] == finish and row['dispatch_s'] >= root:
            j = row['task']
            count[j] += 1
            if selected[j] is None:
                selected[j] = row
    return selected, count


def action_features(public_array, root, horizon, selected, count):
    """Root-known completion/qualification features, no future observations."""
    need(public_array.shape == (176, len(FEATURE_NAMES)), 'Closed base feature shape')
    need(np.all(public_array[:, FEATURE_NAMES.index('root_time_s')] == root), 'Feature origin does not match prediction time')
    need(horizon in HORIZONS, 'Frozen prediction horizon required')
    out = np.zeros((176, len(ACTION_FEATURE_NAMES)), dtype=np.float32)
    out[:, 0] = horizon
    out[:, 1] = root + horizon
    columns = {name: i for i, name in enumerate(FEATURE_NAMES)}
    out[:, -1] = np.clip(public_array[:, columns['temperature_measurement.value']] +
        public_array[:, columns['temperature_velocity_c_s']] * np.minimum(120.,
            public_array[:, columns['temperature_measurement.age_s']] + horizon), 35., 110.)
    # Explicit availability remains in base columns; this numeric padding is
    # never interpreted as a real observed normal temperature.
    out[:, -1] *= public_array[:, columns['temperature_measurement.available']]
    for j, row in enumerate(selected):
        if row is None:
            continue
        need(row['finish_s'] == root + horizon and row['dispatch_s'] >= root,
             'Future completion inconsistent with public prediction root')
        out[j, 2:10] = (1., float(row['optional']), row['dispatch_s'] - root,
            row['finish_s'] - row['dispatch_s'], row['tour_finish_s'] - root,
            row['return_slack_s'], count[j], 1.)
        out[j, 10] = float(row['optional'])
    return out


def evaluator_labels(arrays, seed):
    """Offline A/B labels. B_reference assumes a qualified completed measure.

    B_action is masked separately by legal completion availability. Retained
    anomaly segments are independent metadata, never prerequisites for A.
    """
    physical = reference_signals(arrays) > 0
    confirmable = effective_confirmability(arrays, seed)
    need(physical.shape == confirmable.shape == (360, 176), 'Original professional truth dimensions')
    need(not np.any(confirmable & ~physical), 'Confirmation cannot create a true anomaly')
    event_ids = np.zeros((360, 176), dtype=np.int32)
    retained_ids = np.zeros((360, 176), dtype=np.int32)
    events = []
    for task in range(176):
        boundaries = np.diff(np.r_[False, physical[:, task], False].astype(np.int8))
        for start, end in zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)):
            event = len(events) + 1
            retained = end - start >= 3
            event_ids[start:end, task] = event
            if retained:
                retained_ids[start:end, task] = event
            events.append({'id': event, 'task': task, 'kind': KINDS[0 if task < 128 else 1 if task < 144 else 2 if task < 160 else 3],
                'first_tick': int(start), 'last_exclusive_tick': int(end), 'onset_s': int(start) * 5,
                'end_s': int(end) * 5, 'retained_at_least_15s': bool(retained),
                'confirmable_positive_bins': int(confirmable[start:end, task].sum())})
    return {'A_physical': physical, 'B_reference': confirmable,
            'event_id': event_ids, 'retained_event_id': retained_ids}, events


def sample_targets(labels, root, horizon, selected):
    """Return -1 for unobservable endpoints/illegal completions, never false 0."""
    finish = root + horizon
    if finish > 1800:
        missing = np.full(176, -1, dtype=np.int8)
        return missing, missing.copy(), missing.copy()
    tick = int(finish / 5) - 1
    a = labels['A_physical'][tick].astype(np.int8)
    reference = labels['B_reference'][tick].astype(np.int8)
    b = np.where([v is not None for v in selected], reference, -1).astype(np.int8)
    return a, reference, b


def contract():
    return {'schema': 'b4b0-opportunity-v1', 'base_feature_names': list(FEATURE_NAMES),
        'action_feature_names': list(ACTION_FEATURE_NAMES), 'source_codes': SOURCE_CODES,
        'horizons_s': list(HORIZONS), 'root_grid_s': 5, 'roots': list(range(5, 1800, 5)),
        'features_dtype': 'float32; availability masks separate numeric zero padding from actual values',
        'A': 'Future instantaneous physical professional anomaly, defined at any endpoint<=1800.',
        'B_reference': 'A and original calibrated measurement alarm, conditional on qualified completion; not an action feasibility claim.',
        'B_action': 'B_reference only where a qualified protected-Tour or one-extra-CHECK completion is feasible at the exact endpoint; otherwise -1.',
        'C': 'Separate B4-A one-extra-CHECK then Tour whole-mission outcome reference. Never A/B training input.',
        'event_definition': 'Each maximal physical-positive run. Retained>=15s segments separately tagged, same original endpoint convention.',
        'professional_thresholds': {'thermal': '>65degC plus original deterministic0.1degC measurement noise for B',
            'electrical': 'contact>60degC', 'cooling': 'Pod flow<33.6kg/s', 'control': 'abs(reference sensor error)>1degC'},
        'causal_boundary': 'Public extractor receives only validated root bundle and earlier bundles. Future labels, seed/scenario/event/oracle identities remain evaluator metadata.',
        'topology_semantics': 'Existing synthetic same-Pod CDU and power branch membership; no inferred real-site topology.',
        'sampling_dependence': 'Roots/tasks/horizons share physical trajectories; both noise versions share physical event labels. Seeds group scenarios/phases; numeric fingerprint deduplicates identical trajectories.'}
