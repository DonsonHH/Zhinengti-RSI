"""M9 reduced-order power / two-loop cooling / local-control co-simulator.

All numeric parameters are synthetic research assumptions, not plant ratings or
standard limits. Hidden faults and process truth are separate from telemetry.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np


SCENARIOS = ('healthy', 'load_surge', 'grid_derate', 'dc_contact_hotspot',
             'it_feeder_loss', 'aux_feeder_loss', 'tower_derate', 'primary_pump_derate',
             'secondary_pump_trip', 'cdu_fouling', 'valve_restriction',
             'coldplate_fouling', 'sensor_bias', 'controller_freeze', 'combined')


@dataclass(frozen=True)
class FacilityConfig:
    racks: int = 128
    racks_per_pod: int = 8
    rated_rack_kw: float = 250.0
    dt_s: float = 5.0
    horizon_s: float = 1800.0
    fault_start_s: float = 300.0
    fault_end_s: float = 1200.0
    medium_voltage_v: float = 10000.0
    low_voltage_ac_v: float = 380.0
    rack_dc_v: float = 54.0
    power_factor: float = .95
    it_efficiency: float = .985 * .97 * .97
    aux_efficiency: float = .97
    contact_resistance_ohm: float = .001
    it_capacity_margin: float = 1.25
    ambient_c: float = 28.0
    wetbulb_c: float = 25.0
    chip_capacity_kj_k: float = 800.0
    secondary_capacity_kj_k: float = 8000.0
    primary_capacity_per_pod_kj_k: float = 20000.0
    contact_capacity_kj_k: float = 1000.0
    chip_ambient_ua_kw_k: float = .12
    contact_ambient_ua_kw_k: float = .8
    coldplate_ua_kw_k: float = 12.0
    nominal_rack_flow_kg_s: float = 6.0
    secondary_cp_kj_kg_k: float = 3.8
    primary_cp_kj_kg_k: float = 4.18
    pump_shutoff_kpa: float = 250.0
    branch_nominal_dp_kpa: float = 100.0
    pump_efficiency: float = .70
    cdu_effectiveness: float = .85
    tower_ua_per_pod_kw_k: float = 180.0
    fan_rated_per_pod_kw: float = 10.0
    thermal_reference_c: float = 65.0
    thermal_severe_c: float = 85.0
    hardware_trip_c: float = 95.0
    contact_reference_c: float = 60.0
    contact_severe_c: float = 90.0
    revisit_reference_s: float = 300.0

    def __post_init__(self):
        if self.racks < 8 or self.racks_per_pod != 8 or self.racks % self.racks_per_pod:
            raise ValueError('Use complete pods of eight racks')
        if not (0 < self.dt_s <= 5) or self.horizon_s <= 0:
            raise ValueError('Integration step must be in (0,5] seconds')
        if not 0 < self.it_efficiency <= 1 or not 0 < self.aux_efficiency <= 1:
            raise ValueError('Bad conversion efficiency')

    @property
    def pods(self):
        return self.racks // self.racks_per_pod

    def description(self):
        return {'parameters': asdict(self), 'units': {'power': 'kW', 'energy': 'kJ', 'temperature': 'degC',
                'flow': 'kg/s', 'pressure': 'kPa', 'time': 's', 'thermal_capacity': 'kJ/K'},
                'provenance': 'Synthetic reduced-order research parameters; not national-standard clauses or plant calibration',
                'compute_performance_FLOPS': None}


def secondary_hydraulics(config, speed, valve_open):
    """Solve a quadratic pump curve feeding parallel quadratic branches."""
    branch_k = config.branch_nominal_dp_kpa / config.nominal_rack_flow_kg_s ** 2
    conductance = valve_open.reshape(config.pods, 8) / np.sqrt(branch_k)
    total_conductance = conductance.sum(axis=1)
    nominal_total_flow = config.nominal_rack_flow_kg_s * 8
    pump_k = (config.pump_shutoff_kpa - config.branch_nominal_dp_kpa) / nominal_total_flow ** 2
    pressure = config.pump_shutoff_kpa * speed ** 2 / (1 + pump_k * total_conductance ** 2)
    flow = (conductance * np.sqrt(pressure)[:, None]).reshape(-1)
    power = flow.reshape(config.pods, 8).sum(axis=1) * pressure / (1000 * config.pump_efficiency)
    return flow, pressure, power


class CoupledFacility:
    def __init__(self, config=FacilityConfig(), seed=0, scenario='healthy', control='feedback'):
        if scenario not in SCENARIOS or control not in ('feedback', 'fixed_commands'):
            raise ValueError('Unknown scenario or control mode')
        self.config, self.seed, self._scenario, self.control = config, int(seed), scenario, control
        rng = np.random.default_rng(seed)
        self.rated_kw = config.rated_rack_kw * rng.uniform(.90, 1.10, config.racks)
        self.criticality = rng.choice([1.0, 1.5, 2.0], config.racks)
        self._phase = rng.uniform(0, 2 * np.pi, config.racks)
        self.pod_index = np.arange(config.racks) // 8
        self.t_s = 0.0
        self._secondary_c = np.full(config.pods, 35.0)
        self._primary_c = 32.0
        self._rack_c = 35.0 + self.rated_kw * .65 / config.coldplate_ua_kw_k
        self._contact_c = np.full(config.pods, 38.0)
        self._pump_command = np.full(config.pods, .85)
        self._fan_command = .85
        self._throttle = np.ones(config.racks)
        self._last = None

    def _faults(self, t):
        c = self.config
        pod = self.seed % c.pods
        rack = pod * 8 + (self.seed * 3) % 8
        f = {'grid': 1.0, 'it_feed': np.ones(c.pods), 'aux_feed': np.ones(c.pods),
             'contact_r': np.ones(c.pods), 'tower': 1.0, 'primary_pump': 1.0,
             'secondary_pump': np.ones(c.pods), 'cdu': np.ones(c.pods),
             'valve': np.ones(c.racks), 'coldplate': np.ones(c.racks),
             'sensor_bias': np.zeros(c.racks), 'controller_stuck': np.zeros(c.pods, dtype=bool),
             'load_addition': 0.0}
        if not c.fault_start_s <= t < c.fault_end_s:
            return f
        scenario = self._scenario
        if scenario == 'load_surge': f['load_addition'] = .25
        elif scenario == 'grid_derate': f['grid'] = .5
        elif scenario == 'dc_contact_hotspot': f['contact_r'][pod] = 12
        elif scenario == 'it_feeder_loss': f['it_feed'][pod] = .05
        elif scenario == 'aux_feeder_loss': f['aux_feed'][pod] = .05
        elif scenario == 'tower_derate': f['tower'] = .15
        elif scenario == 'primary_pump_derate': f['primary_pump'] = .25
        elif scenario == 'secondary_pump_trip': f['secondary_pump'][pod] = .10
        elif scenario == 'cdu_fouling': f['cdu'][pod] = .20
        elif scenario == 'valve_restriction': f['valve'][rack] = .12
        elif scenario == 'coldplate_fouling': f['coldplate'][rack] = .10
        elif scenario == 'sensor_bias': f['sensor_bias'][pod * 8:(pod + 1) * 8] = -18
        elif scenario == 'controller_freeze': f['controller_stuck'][pod] = True
        elif scenario == 'combined':
            f['aux_feed'][pod] = .15
            f['cdu'][pod] = .45
            f['contact_r'][pod] = 3
            f['sensor_bias'][pod * 8:(pod + 1) * 8] = -10
        return f

    def requested_power(self, t=None):
        t = self.t_s if t is None else t
        fault = self._faults(t)
        load = .65 + .10 * np.sin(t / 180 + self._phase) + .05 * np.sin(t / 41 + self._phase * 1.7)
        return self.rated_kw * np.clip(load + fault['load_addition'], .30, 1.0)

    def step(self):
        c = self.config
        dt, time_before = c.dt_s, self.t_s
        f = self._faults(time_before)
        demand = self.requested_power(time_before)
        sensed = self._rack_c + f['sensor_bias']
        if self.control == 'feedback':
            desired_pump = np.clip(.85 + .035 * (sensed.reshape(c.pods, 8).max(axis=1) - 58), .75, 1.2)
            desired_fan = float(np.clip(.85 + .10 * (self._primary_c - 32), .25, 1.2))
            desired_throttle = np.clip(1 - .035 * (sensed - 65), .20, 1.0)
            alpha = min(1.0, dt / 20)
            self._pump_command += alpha * (desired_pump - self._pump_command)
            self._fan_command += alpha * (desired_fan - self._fan_command)
            self._throttle += alpha * (desired_throttle - self._throttle)
        else:
            self._pump_command[:] = .85
            self._fan_command = .85
            self._throttle[:] = 1
        effective_command = self._pump_command.copy()
        effective_command[f['controller_stuck']] = .25
        throttle = self._throttle.copy()
        throttle[np.repeat(f['controller_stuck'], 8)] = 1
        # A declared independent local hardware trip is a simulator protection,
        # not truth made available to a task scheduler or estimator.
        throttle[self._rack_c >= c.hardware_trip_c] = 0
        pump_speed = effective_command * np.sqrt(f['grid'] * f['aux_feed']) * f['secondary_pump']
        flow, pressure, secondary_pump_kw = secondary_hydraulics(c, pump_speed, f['valve'])
        desired_it = demand * throttle
        preline_requested = desired_it.reshape(c.pods, 8).sum(axis=1) / c.it_efficiency
        k = 1000 * c.contact_resistance_ohm * f['contact_r'] / (c.low_voltage_ac_v ** 2 * c.power_factor ** 2)
        capacity = self.rated_kw.reshape(c.pods, 8).sum(axis=1) * c.it_capacity_margin * f['grid'] * f['it_feed']
        max_preline = 2 * capacity / (1 + np.sqrt(1 + 4 * k * capacity))
        availability = np.minimum(1, max_preline / np.maximum(preline_requested, 1e-12))
        delivered_it = desired_it * np.repeat(availability, 8)
        preline = delivered_it.reshape(c.pods, 8).sum(axis=1) / c.it_efficiency
        contact_heat = k * preline ** 2
        it_conversion_loss = delivered_it.sum() * (1 / c.it_efficiency - 1)
        primary_speed = .95 * np.sqrt(f['grid']) * f['primary_pump']
        primary_flow_per_pod = c.nominal_rack_flow_kg_s * 8 * primary_speed
        primary_pump_kw = primary_flow_per_pod * c.pods * (80 * primary_speed ** 2) / (1000 * .75)
        fan_speed = self._fan_command * np.sqrt(f['grid'])
        fan_kw = c.fan_rated_per_pod_kw * c.pods * fan_speed ** 3
        aux_kw = secondary_pump_kw.sum() + primary_pump_kw + fan_kw
        aux_conversion_loss = aux_kw * (1 / c.aux_efficiency - 1)
        source_kw = preline.sum() + contact_heat.sum() + aux_kw / c.aux_efficiency
        electric_residual = source_kw - (delivered_it.sum() + aux_kw + it_conversion_loss + aux_conversion_loss + contact_heat.sum())

        conductance = c.coldplate_ua_kw_k * (flow / c.nominal_rack_flow_kg_s) ** .8 * f['coldplate']
        coldplate_heat = conductance * (self._rack_c - np.repeat(self._secondary_c, 8))
        secondary_heat = coldplate_heat.reshape(c.pods, 8).sum(axis=1)
        secondary_capacity_rate = flow.reshape(c.pods, 8).sum(axis=1) * c.secondary_cp_kj_kg_k
        return_rise = np.divide(secondary_heat, secondary_capacity_rate, out=np.zeros(c.pods), where=secondary_capacity_rate > 1e-12)
        secondary_return = self._secondary_c + return_rise
        primary_capacity_rate = primary_flow_per_pod * c.primary_cp_kj_kg_k
        cdu_heat = c.cdu_effectiveness * f['cdu'] * np.minimum(secondary_capacity_rate, primary_capacity_rate) * (secondary_return - self._primary_c)
        tower_heat = c.tower_ua_per_pod_kw_k * c.pods * fan_speed * f['tower'] * max(0, self._primary_c - c.wetbulb_c)
        rack_ambient_heat = c.chip_ambient_ua_kw_k * (self._rack_c - c.ambient_c)
        contact_ambient_heat = c.contact_ambient_ua_kw_k * (self._contact_c - c.ambient_c)
        energy_before = self.stored_energy_kj()
        self._rack_c += dt * (delivered_it - coldplate_heat - rack_ambient_heat) / c.chip_capacity_kj_k
        self._secondary_c += dt * (secondary_heat + secondary_pump_kw - cdu_heat) / c.secondary_capacity_kj_k
        self._primary_c += dt * (cdu_heat.sum() + primary_pump_kw - tower_heat) / (c.primary_capacity_per_pod_kj_k * c.pods)
        self._contact_c += dt * (contact_heat - contact_ambient_heat) / c.contact_capacity_kj_k
        rejected_heat = tower_heat + rack_ambient_heat.sum() + contact_ambient_heat.sum() + fan_kw + it_conversion_loss + aux_conversion_loss
        energy_residual = self.stored_energy_kj() - energy_before - dt * (source_kw - rejected_heat)
        self.t_s += dt
        flow_required = delivered_it / (c.secondary_cp_kj_kg_k * 10)
        thermal_severity = np.clip((self._rack_c - c.thermal_reference_c) / (c.thermal_severe_c - c.thermal_reference_c), 0, 2)
        contact_severity = np.repeat(np.clip((self._contact_c - c.contact_reference_c) / (c.contact_severe_c - c.contact_reference_c), 0, 2), 8)
        service_gap = 1 - delivered_it / np.maximum(demand, 1e-12)
        flow_gap = np.maximum(0, 1 - flow / np.maximum(flow_required, 1e-12))
        severity = .45 * thermal_severity + .20 * contact_severity + .25 * service_gap + .10 * flow_gap
        self._last = {'time_s': self.t_s, 'interval_start_s': time_before,
            'rack_temperature_c': self._rack_c.copy(), 'secondary_supply_c': self._secondary_c.copy(),
            'primary_supply_c': self._primary_c, 'contact_temperature_c': self._contact_c.copy(),
            'requested_it_kw': demand, 'delivered_it_kw': delivered_it, 'rack_flow_kg_s': flow,
            'secondary_dp_kpa': pressure, 'pump_command': effective_command, 'pump_speed': pump_speed,
            'primary_pump_speed': primary_speed, 'fan_speed': fan_speed, 'throttle': throttle,
            'it_power_availability': availability, 'public_sensor_temperature_c': self._rack_c + f['sensor_bias'],
            'thermal_severity': thermal_severity, 'contact_severity': contact_severity,
            'service_gap': service_gap, 'flow_gap': flow_gap, 'severity_proxy': severity,
            'source_power_kw': float(source_kw), 'it_power_kw': float(delivered_it.sum()), 'aux_power_kw': float(aux_kw),
            'conversion_loss_kw': float(it_conversion_loss + aux_conversion_loss), 'contact_loss_kw': float(contact_heat.sum()),
            'rejected_heat_kw': float(rejected_heat), 'stored_energy_kj': self.stored_energy_kj(),
            'electric_residual_kw': float(electric_residual), 'energy_residual_kj': float(energy_residual),
            'unserved_it_kw': float((demand - delivered_it).sum()),
            'impact_weighted_severity': float(np.sum(severity * self.criticality) / self.criticality.sum())}
        return self.truth_snapshot()

    def stored_energy_kj(self):
        c = self.config
        return float(c.chip_capacity_kj_k * self._rack_c.sum() + c.secondary_capacity_kj_k * self._secondary_c.sum()
                     + c.primary_capacity_per_pod_kj_k * c.pods * self._primary_c + c.contact_capacity_kj_k * self._contact_c.sum())

    def truth_snapshot(self):
        if self._last is None:
            raise RuntimeError('Step the model before requesting a snapshot')
        return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in self._last.items()}

    def public_telemetry(self):
        if self._last is None:
            raise RuntimeError('No telemetry before first step')
        s, c = self._last, self.config
        mask = np.arange(c.racks) % 4 == 0
        pod_flow_mask = np.arange(c.pods) % 2 == 0
        # The recipient never gets unmeasured rack temperatures or fault labels.
        return {'time_s': self.t_s, 'temperature_indices': np.flatnonzero(mask),
                'temperature_c': s['public_sensor_temperature_c'][mask].copy(),
                'public_requested_it_kw': s['requested_it_kw'].copy(), 'public_delivered_it_kw': s['delivered_it_kw'].copy(),
                'pod_pump_command': s['pump_command'].copy(), 'primary_supply_c': s['primary_supply_c'],
                'secondary_supply_c': s['secondary_supply_c'].copy(),
                'flow_pod_indices': np.flatnonzero(pod_flow_mask),
                'flow_pod_kg_s': s['rack_flow_kg_s'].reshape(c.pods, 8).sum(axis=1)[pod_flow_mask].copy()}

    def calibrated_inspection(self, kind, index):
        """Measurements generated only at a valid task-completion boundary."""
        if kind == 'thermal':
            return {'rack_index': index, 'temperature_c': float(self._rack_c[index])}
        if kind == 'electrical':
            return {'pod_index': index, 'contact_temperature_c': float(self._contact_c[index])}
        if kind == 'cooling':
            return {'pod_index': index, 'pod_flow_kg_s': float(self._last['rack_flow_kg_s'].reshape(self.config.pods, 8).sum(axis=1)[index])}
        if kind == 'control':
            return {'pod_index': index, 'controller_sensor_error_c': float(np.mean(self._last['public_sensor_temperature_c'].reshape(self.config.pods, 8)[index] - self._rack_c.reshape(self.config.pods, 8)[index]))}
        raise ValueError('Unknown inspection channel')


def facility_topology(config):
    """Sparse dependency graph, not a reconstructed site piping diagram."""
    nodes = [{'id': 'grid', 'system': 'power', 'kind': 'medium_voltage_inlet', 'voltage_v': config.medium_voltage_v},
             {'id': 'tower', 'system': 'cooling', 'kind': 'cooling_tower'},
             {'id': 'primary_pump', 'system': 'cooling', 'kind': 'facility_primary_pump'},
             {'id': 'primary_loop', 'system': 'cooling', 'kind': 'facility_primary_loop'}]
    edges = [{'source': 'grid', 'target': 'primary_pump', 'relation': 'power'},
             {'source': 'primary_pump', 'target': 'primary_loop', 'relation': 'circulation'},
             {'source': 'primary_loop', 'target': 'tower', 'relation': 'heat_rejection'},
             {'source': 'tower', 'target': 'primary_loop', 'relation': 'cooled_supply'}]
    for pod in range(config.pods):
        for kind, system in [('transformer', 'power'), ('ac_panel', 'power'), ('ups', 'power'), ('dc_rectifier', 'power'),
                             ('aux_feeder', 'power'), ('secondary_pump', 'cooling'), ('cdu', 'cooling'), ('controller', 'control')]:
            item = {'id': f'{kind}:{pod}', 'system': system, 'kind': kind, 'pod': pod}
            if kind == 'ac_panel': item['voltage_v'] = config.low_voltage_ac_v
            if kind == 'dc_rectifier': item['voltage_v'] = config.rack_dc_v
            nodes.append(item)
        for source, target, relation in [('grid', f'transformer:{pod}', 'power'),
                (f'transformer:{pod}', f'ac_panel:{pod}', 'voltage_conversion'),
                (f'ac_panel:{pod}', f'ups:{pod}', 'power_conditioning'),
                (f'ups:{pod}', f'dc_rectifier:{pod}', 'ac_dc_conversion'),
                (f'ac_panel:{pod}', f'aux_feeder:{pod}', 'auxiliary_power'),
                (f'aux_feeder:{pod}', f'secondary_pump:{pod}', 'power'),
                (f'secondary_pump:{pod}', f'cdu:{pod}', 'circulation'),
                (f'cdu:{pod}', 'primary_loop', 'heat_exchange'),
                ('primary_loop', f'cdu:{pod}', 'facility_supply'),
                (f'controller:{pod}', f'secondary_pump:{pod}', 'speed_command')]:
            edges.append({'source': source, 'target': target, 'relation': relation})
        for rack in range(pod * 8, pod * 8 + 8):
            for kind, system in [('rack', 'IT'), ('coldplate', 'cooling'), ('sensor', 'control')]:
                nodes.append({'id': f'{kind}:{rack}', 'system': system, 'kind': kind, 'rack': rack, 'pod': pod})
            for source, target, relation in [(f'dc_rectifier:{pod}', f'rack:{rack}', 'power'),
                    (f'rack:{rack}', f'coldplate:{rack}', 'heat'),
                    (f'coldplate:{rack}', f'cdu:{pod}', 'secondary_return'),
                    (f'cdu:{pod}', f'coldplate:{rack}', 'secondary_supply'),
                    (f'coldplate:{rack}', f'rack:{rack}', 'cooling_effect'),
                    (f'rack:{rack}', f'sensor:{rack}', 'sensing'),
                    (f'sensor:{rack}', f'controller:{pod}', 'feedback'),
                    (f'controller:{pod}', f'rack:{rack}', 'load_limit')]:
                edges.append({'source': source, 'target': target, 'relation': relation})
    return {'nodes': nodes, 'edges': edges, 'nominal_IT_MW': config.racks * config.rated_rack_kw / 1000,
            'scope': 'Synthetic hierarchical dependency graph; no site calibration or FLOPS inference'}
