"""Qualified task completion and observation-only facility baselines."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .facility import FacilityConfig


@dataclass(frozen=True)
class InspectionTask:
    task_id: str
    kind: str
    index: int
    required_role: str
    service_s: float
    position_m: tuple[float, float]
    access: str = 'external_sensing'
    requires_calibrated_measurement: bool = True


def eligible(role, task):
    return role == task.required_role and task.access == 'external_sensing'


def rack_positions(config):
    columns = math.ceil(math.sqrt(config.pods))
    result = np.zeros((config.racks, 2))
    for rack in range(config.racks):
        pod, local = divmod(rack, 8)
        result[rack] = ((pod % columns) * 12 + (local % 4) * 1.2,
                        (pod // columns) * 12 + (local // 4) * 1.5)
    return result


def inspection_tasks(config):
    positions = rack_positions(config)
    tasks = [InspectionTask(f'thermal:{rack}', 'thermal', rack, 'thermal', 10.0,
                            tuple(positions[rack])) for rack in range(config.racks)]
    for pod in range(config.pods):
        origin = positions[pod * 8]
        for kind, duration, offset in [('electrical', 20.0, (2, 5)), ('cooling', 20.0, (6, 3)), ('control', 15.0, (6, 6))]:
            tasks.append(InspectionTask(f'{kind}:{pod}', kind, pod, kind, duration, tuple(origin + offset)))
    return tasks


class FacilityObserver:
    """No reference to the simulator object, hidden state or fault descriptor."""
    def __init__(self, config: FacilityConfig):
        self.config = config
        self.last_temperature = np.full(config.racks, 50.0)
        self.predicted_temperature = np.full(config.racks, 50.0)
        self.temperature_stamp = np.full(config.racks, -1.0)
        self.last_contact = np.full(config.pods, 38.0)
        self.last_flow = np.full(config.pods, config.nominal_rack_flow_kg_s * 8 * .85)
        self.flow_stamp = np.full(config.pods, -1.0)
        self.thermal_counts = np.zeros(config.racks, dtype=int)
        self.thermal_inspection_stamp = np.full(config.racks, -1.0)
        self.external_references = []
        self.sensor_bias_verified = np.zeros(config.pods, dtype=bool)

    def ingest_reference(self, metadata):
        if metadata.get('kind') != 'apparent_ir_reference' or metadata.get('calibrated', False):
            raise ValueError('B-PID adapter accepts explicitly uncalibrated apparent references only')
        self.external_references.append({'kind': 'apparent_ir_reference', 'calibrated': False,
                                         'meaning': 'Candidate verification cue only; not temperature or fault probability'})

    def complete_inspection(self, task, role, measurement, time_s, calibrated=True):
        if not eligible(role, task):
            raise ValueError('Inspection role/access mismatch')
        if task.requires_calibrated_measurement and not calibrated:
            raise ValueError('An apparent image reference cannot complete calibrated measurement')
        if task.kind == 'thermal':
            if measurement.get('rack_index') != task.index or not np.isfinite(measurement['temperature_c']):
                raise ValueError('Wrong thermal measurement target')
            self.last_temperature[task.index] = measurement['temperature_c']
            self.predicted_temperature[task.index] = measurement['temperature_c']
            self.temperature_stamp[task.index] = time_s
            self.thermal_counts[task.index] += 1
            self.thermal_inspection_stamp[task.index] = time_s
        elif task.kind == 'electrical':
            self.last_contact[task.index] = measurement['contact_temperature_c']
        elif task.kind == 'cooling':
            self.last_flow[task.index] = measurement['pod_flow_kg_s']
            self.flow_stamp[task.index] = time_s
        elif task.kind == 'control':
            self.sensor_bias_verified[task.index] = abs(measurement['controller_sensor_error_c']) > 1
        else:
            raise ValueError('Unknown inspection task')

    def update(self, packet):
        c = self.config
        t = float(packet['time_s'])
        flow = c.nominal_rack_flow_kg_s * packet['pod_pump_command'] * 8
        indices = packet['flow_pod_indices']
        self.last_flow[indices] = packet['flow_pod_kg_s']
        self.flow_stamp[indices] = t
        known = self.flow_stamp >= 0
        flow[known] = self.last_flow[known]
        per_rack_flow = np.repeat(flow / 8, 8)
        conductance = c.coldplate_ua_kw_k * (np.maximum(0, per_rack_flow) / c.nominal_rack_flow_kg_s) ** .8
        self.predicted_temperature += c.dt_s * (
            packet['public_delivered_it_kw'] - conductance * (self.predicted_temperature - np.repeat(packet['secondary_supply_c'], 8))
            - c.chip_ambient_ua_kw_k * (self.predicted_temperature - c.ambient_c)) / c.chip_capacity_kj_k
        measured = packet['temperature_indices']
        self.last_temperature[measured] = packet['temperature_c']
        self.predicted_temperature[measured] = packet['temperature_c']
        self.temperature_stamp[measured] = t
        return self.readout(packet)

    def readout(self, packet):
        c, t = self.config, float(packet['time_s'])
        flow = c.nominal_rack_flow_kg_s * packet['pod_pump_command'] * 8
        known = self.flow_stamp >= 0
        flow[known] = self.last_flow[known]
        per_rack_flow = np.repeat(flow / 8, 8)
        gap = 1 - packet['public_delivered_it_kw'] / np.maximum(packet['public_requested_it_kw'], 1e-12)
        required_flow = packet['public_delivered_it_kw'] / (c.secondary_cp_kj_kg_k * 10)
        flow_gap = np.maximum(0, 1 - per_rack_flow / np.maximum(required_flow, 1e-12))
        contact = np.repeat(np.clip((self.last_contact - c.contact_reference_c) / (c.contact_severe_c - c.contact_reference_c), 0, 2), 8)
        risks = {}
        for name, temperature in [('hold', self.last_temperature), ('nominal_model', self.predicted_temperature)]:
            thermal = np.clip((temperature - c.thermal_reference_c) / (c.thermal_severe_c - c.thermal_reference_c), 0, 2)
            risks[name] = .45 * thermal + .20 * contact + .25 * gap + .10 * flow_gap
        ages = np.where(self.temperature_stamp >= 0, t - self.temperature_stamp, t)
        return {'time_s': t, 'hold_temperature_c': self.last_temperature.copy(),
                'predicted_temperature_c': self.predicted_temperature.copy(),
                'hold_severity_proxy': risks['hold'], 'predicted_severity_proxy': risks['nominal_model'],
                'measurement_age_s': ages, 'ever_measured': self.temperature_stamp >= 0,
                'inspection_age_s': np.where(self.thermal_inspection_stamp >= 0, t - self.thermal_inspection_stamp, t),
                'thermal_inspection_counts': self.thermal_counts.copy(),
                'fresh_measurement': (self.temperature_stamp >= 0) & (ages <= 10),
                'scope': 'Transparent nonlearned observation baselines; no VAE or B-PID inference'}


class FixedQualifiedInspection:
    def __init__(self, config):
        tasks = inspection_tasks(config)
        self.groups = {role: [task for task in tasks if task.required_role == role]
                       for role in ('thermal', 'electrical', 'cooling', 'control')}
        self.agents = {role: {'cursor': 0, 'position': np.array([0., 0.]), 'remaining': 0., 'task': None,
                               'travel_total_m': 0.0, 'service_total_s': 0.0, 'completed': 0}
                       for role in self.groups}
        self.events = []
        self.config = config

    def advance(self, plant, observer):
        completed = []
        for role, agent in self.agents.items():
            if agent['task'] is None:
                task = self.groups[role][agent['cursor'] % len(self.groups[role])]
                agent['cursor'] += 1
                if not eligible(role, task):
                    raise RuntimeError('Fixed scheduler attempted an ineligible task')
                distance = float(np.linalg.norm(agent['position'] - np.asarray(task.position_m)))
                agent.update(task=task, remaining=distance / 1.0 + task.service_s)
                agent['travel_total_m'] += distance
            agent['remaining'] -= self.config.dt_s
            if agent['remaining'] <= 1e-9:
                task = agent['task']
                measurement = plant.calibrated_inspection(task.kind, task.index)
                observer.complete_inspection(task, role, measurement, plant.t_s)
                agent['position'] = np.asarray(task.position_m)
                agent['service_total_s'] += task.service_s
                agent['completed'] += 1
                event = {'time_s': plant.t_s, 'role': role, 'task_id': task.task_id, 'kind': task.kind,
                         'index': task.index, 'service_s': task.service_s, 'measurement': measurement,
                         'calibrated': True, 'access': task.access}
                completed.append(event)
                self.events.append(event)
                agent['task'] = None
        return completed
