"""Explicit task/evidence compatibility; service evidence is not root cause."""
from __future__ import annotations

import numpy as np


def capability_matrix():
    return {
        'thermal_reference': {'implemented': True,
            'accepted': ['qualified_mobile_temperature', 'validated_remote_temperature'],
            'current_remote_temperature_validated': False,
            'meaning': 'Calibrated representative node temperature, not apparent IR pixels'},
        'electrical_contact_heat': {'implemented': True,
            'accepted': ['qualified_contact_temperature', 'validated_remote_contact_temperature'],
            'meaning': 'Connection temperature only; not supply availability or voltage'},
        'cooling_flow': {'implemented': True,
            'accepted': ['qualified_flow_measurement', 'validated_remote_flow'],
            'meaning': 'Calibrated pod flow'},
        'control_sensor_bias': {'implemented': True,
            'accepted': ['qualified_reference_comparison'],
            'meaning': 'Reference comparison, not ordinary remote temperature alone'},
        'IT_service_availability': {'implemented': True,
            'accepted': ['calibrated_requested_and_delivered_power'],
            'meaning': 'Observed IT service deficit; cause remains undiagnosed'},
        'ac_voltage': {'implemented': False, 'accepted': ['calibrated_voltage_measurement'],
            'meaning': 'No actual voltage observation channel in this reduced-order model'},
        'branch_current': {'implemented': False, 'accepted': ['calibrated_current_measurement'],
            'meaning': 'A nominal V/power calculation is an estimate, not current measurement'}
    }


def assess_capability(task_kind, evidence_kind, *, calibrated=False, channel_available=False):
    row = capability_matrix()[task_kind]
    if not row['implemented'] or not channel_available:
        return {'completes_task': False, 'reason': 'required_measurement_channel_unavailable'}
    if not calibrated:
        return {'completes_task': False, 'reason': 'calibration_not_established'}
    if evidence_kind not in row['accepted']:
        return {'completes_task': False, 'reason': 'evidence_does_not_answer_this_task'}
    return {'completes_task': True, 'reason': 'declared_measurement_contract_met'}


class ServiceEvidence:
    """Per-pod, three consecutive 5s readings; all methods get the same tool."""
    def __init__(self, pods, *, calibrated=False):
        self.pods, self.calibrated = pods, calibrated
        self.last_time = None
        self.bad = np.zeros(pods, dtype=int)
        self.good = np.zeros(pods, dtype=int)
        self.active = np.zeros(pods, dtype=bool)
        self.last_report = np.full(pods, -np.inf)
        self.events = []

    def update(self, packet):
        t = float(packet['time_s'])
        if self.last_time is not None and abs(t - self.last_time - 5) > 1e-9:
            raise ValueError('Persistence requires consecutive 5s observations')
        requested = np.asarray(packet['public_requested_it_kw'], dtype=float)
        delivered = np.asarray(packet['public_delivered_it_kw'], dtype=float)
        if requested.shape != (self.pods * 8,) or delivered.shape != requested.shape:
            raise ValueError('Power observation schema mismatch')
        if not np.isfinite(requested).all() or not np.isfinite(delivered).all() or (requested < 0).any() or (delivered < 0).any():
            raise ValueError('Invalid power observations')
        self.last_time = t
        permission = assess_capability('IT_service_availability',
            'calibrated_requested_and_delivered_power', calibrated=self.calibrated, channel_available=True)
        req, actual = requested.reshape(self.pods, 8).sum(1), delivered.reshape(self.pods, 8).sum(1)
        ratio = np.divide(np.maximum(req - actual, 0), req, out=np.zeros(self.pods), where=req > 1)
        deficit = (ratio > .1) & (req > 1)
        self.bad = np.where(deficit, self.bad + 1, 0)
        self.good = np.where(~deficit, self.good + 1, 0)
        emitted = []
        if not permission['completes_task']:
            return emitted
        for pod in range(self.pods):
            kind = None
            if self.bad[pod] >= 3 and not self.active[pod]:
                self.active[pod], kind = True, 'new_measured_alert'
            elif self.bad[pod] >= 3 and t - self.last_report[pod] >= 60:
                kind = 'persistent_recheck'
            elif self.good[pod] >= 3 and self.active[pod]:
                self.active[pod], kind = False, 'recovery_confirmation'
            if kind:
                event = {'time_s': t, 'pod': pod, 'kind': kind, 'task': 'IT_service_availability',
                         'requested_kw': float(req[pod]), 'delivered_kw': float(actual[pod]),
                         'service_gap_ratio': float(ratio[pod]), 'calibration_declared_in_simulation': True,
                         'root_cause_diagnosed': False, 'voltage_or_current_measured': False}
                self.last_report[pod] = t
                emitted.append(event)
                self.events.append(event)
        return emitted
