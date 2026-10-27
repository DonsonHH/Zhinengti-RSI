"""M11 causal, qualified, aisle-based asynchronous inspection environment."""
from __future__ import annotations

from functools import lru_cache
import math
import time

import numpy as np

from .aisles import AisleGraph, DEADLINES, ROLES, build_agents
from .estimation import ObservationState
from .estimation_data import CompositeFacility
from .facility import FacilityConfig
from .facility_observation import eligible
from .m9 import arrays_digest

CANDIDATE_FEATURES = ('age_ratio', 'eta_ratio', 'public_severity', 'never_inspected',
                      'measurement_age_ratio', 'model_disagreement', 'asset_weight',
                      'service_ratio', 'travel_ratio', 'thermal', 'electrical',
                      'cooling', 'control', 'clock_fraction', 'local_fraction',
                      'team_unverified_fraction')


@lru_cache(maxsize=4)
def graph_for(racks, variant):
    return AisleGraph(FacilityConfig(racks=racks), variant)


class ProcessTrace:
    """Process evolves independently of inspection; labels never enter actor."""
    def __init__(self, racks, seed, scenario, control):
        self.config = FacilityConfig(racks=racks)
        plant = CompositeFacility(self.config, seed, scenario, control)
        self.seed, self.scenario, self.control = seed, scenario, control
        packets, truth = [], []
        maximum_energy, maximum_electric, peak_input = 0., 0., 0.
        keys = ('rack_temperature_c', 'secondary_supply_c', 'primary_supply_c',
                'contact_temperature_c', 'rack_flow_kg_s',
                'public_sensor_temperature_c', 'severity_proxy')
        arrays = {key: [] for key in keys}
        initial_energy = plant.stored_energy_kj()
        for _ in range(360):
            row = plant.step()
            packets.append(plant.public_telemetry())
            for key in keys:
                arrays[key].append(row[key])
            truth.append([row[k] for k in ('time_s', 'unserved_it_kw', 'source_power_kw',
                         'stored_energy_kj', 'rejected_heat_kw', 'it_power_kw', 'aux_power_kw',
                         'conversion_loss_kw', 'contact_loss_kw')])
            maximum_energy = max(maximum_energy, abs(row['energy_residual_kj']))
            maximum_electric = max(maximum_electric, abs(row['electric_residual_kw']))
            peak_input = max(peak_input, row['source_power_kw'])
        self.arrays = {key: np.stack(value) for key, value in arrays.items()}
        self.arrays['process_scalars'] = np.asarray(truth)
        self.arrays['asset_weights'] = plant.criticality.copy()
        self.packets = packets
        self.fingerprint = arrays_digest(self.arrays)
        self.metadata = {'racks': racks, 'seed': seed, 'scenario': scenario, 'control': control,
                         'numeric_digest': self.fingerprint, 'peak_source_input_MW': peak_input / 1000,
                         'IT_nameplate_MW': racks * 250 / 1000,
                         'GW_supply_design_budget_MW': 1000 if racks == 3200 else None,
                         'maximum_energy_residual_kj': maximum_energy,
                         'maximum_electric_residual_kw': maximum_electric,
                         'initial_energy_kj': initial_energy,
                         'unserved_IT_kwh': sum(r[1] for r in truth) * 5 / 3600}

    def measurement(self, task, tick):
        a, c = self.arrays, self.config
        if task.kind == 'thermal':
            noise = np.random.default_rng(3000000 + self.seed * 10000 + task.index * 361 + tick).normal(0, .1)
            return {'rack_index': task.index, 'temperature_c': float(a['rack_temperature_c'][tick, task.index] + noise)}
        if task.kind == 'electrical':
            return {'pod_index': task.index, 'contact_temperature_c': float(a['contact_temperature_c'][tick, task.index])}
        if task.kind == 'cooling':
            return {'pod_index': task.index, 'pod_flow_kg_s': float(a['rack_flow_kg_s'][tick].reshape(c.pods, 8)[task.index].sum())}
        error = a['public_sensor_temperature_c'][tick] - a['rack_temperature_c'][tick]
        return {'pod_index': task.index, 'controller_sensor_error_c': float(error.reshape(c.pods, 8)[task.index].mean())}


@lru_cache(maxsize=4)
def trace_for(racks, seed, scenario, control):
    return ProcessTrace(racks, seed, scenario, control)


def measured_anomaly(task, measurement):
    if task.kind == 'thermal':
        return measurement['temperature_c'] > 65
    if task.kind == 'electrical':
        return measurement['contact_temperature_c'] > 60
    if task.kind == 'cooling':
        return measurement['pod_flow_kg_s'] < 33.6
    return abs(measurement['controller_sensor_error_c']) > 1


class ActiveInspection:
    def __init__(self, spec, predictor=None):
        self.spec = dict(spec)
        self.graph = graph_for(spec.get('racks', 128), spec.get('variant', 'open'))
        self.config = self.graph.config
        self.trace = trace_for(self.config.racks, spec['seed'], spec['scenario'], spec['control'])
        self.tasks = self.graph.tasks
        self.agents = build_agents(self.graph, spec['resources'])
        self.mode = spec['estimator']
        if self.mode not in ('nominal', 'history') or self.mode == 'history' and predictor is None:
            raise ValueError('Frozen history predictor required')
        self.predictor = predictor
        self.state = ObservationState(self.config)
        self.last_inspection = np.full(len(self.tasks), -1.)
        self.ever = np.zeros(len(self.tasks), dtype=bool)
        asset = self.trace.arrays['asset_weights']
        self.asset_weights = np.array([asset[t.index] if t.kind == 'thermal'
                                      else asset.reshape(self.config.pods, 8)[t.index].mean() for t in self.tasks])
        self.deadlines = np.array([DEADLINES[t.kind] for t in self.tasks])
        self.events, self.assignments, self.frames, self.scalar_trace = [], [], [], []
        self.tick, self.clock = 0, 5.
        self.current_history = np.full(self.config.racks, 50.)
        self.inference_seconds = 0.
        self._update([])
        self.scalar_trace.append(self.public_metrics())

    @property
    def done(self):
        return self.tick >= 359

    def packet(self):
        packet = self.trace.packets[self.tick]
        result = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in packet.items()}
        rng = np.random.default_rng(4000000 + self.spec['seed'] * 1000 + self.tick)
        result['temperature_c'] += rng.normal(0, .2, len(result['temperature_c']))
        return result

    def _update(self, completed):
        self._risk_cache = None
        self.current_packet = self.packet()
        self.observation = self.state.update(self.current_packet, completed)
        if round(self.clock) % 30 == 0:
            self.frames.append(self.state.features(self.current_packet))
            self.frames = self.frames[-4:]
            if self.mode == 'history':
                frames = [self.frames[0]] * (4 - len(self.frames)) + self.frames
                context = np.concatenate(frames, axis=-1)
                start = time.perf_counter()
                self.current_history = np.asarray(self.predictor(context), dtype=float)
                self.inference_seconds += time.perf_counter() - start
                if self.current_history.shape != (self.config.racks,) or not np.isfinite(self.current_history).all():
                    raise ValueError('Invalid observation-only estimate')
        fresh = self.state.observer.temperature_stamp == self.clock
        self.current_history[fresh] = self.state.observer.last_temperature[fresh]

    def public_risk(self):
        if self._risk_cache is not None:
            return self._risk_cache
        obs, c = self.state.observer, self.config
        temperature = obs.predicted_temperature if self.mode == 'nominal' else self.current_history
        disagreement = np.zeros(c.racks) if self.mode == 'nominal' else abs(temperature - obs.predicted_temperature) / 20
        thermal = np.maximum(temperature - 65, 0) / 20
        contact = np.maximum(obs.last_contact - 60, 0) / 30
        flow = np.maximum(1 - obs.last_flow / 33.6, 0)
        gap = 1 - self.current_packet['public_delivered_it_kw'] / np.maximum(self.current_packet['public_requested_it_kw'], 1e-12)
        pod_thermal = thermal.reshape(c.pods, 8).max(1)
        pod_gap = gap.reshape(c.pods, 8).mean(1)
        rows = []
        for task in self.tasks:
            if task.kind == 'thermal':
                rows.append(thermal[task.index] + .25 * gap[task.index])
            elif task.kind == 'electrical':
                rows.append(contact[task.index] + pod_gap[task.index] + .1 * pod_thermal[task.index])
            elif task.kind == 'cooling':
                rows.append(flow[task.index] + .25 * pod_thermal[task.index])
            else:
                rows.append(float(obs.sensor_bias_verified[task.index]) + .25 * pod_thermal[task.index])
        self._risk_cache = np.minimum(np.asarray(rows), 4), disagreement
        return self._risk_cache

    def public_metrics(self):
        ages = np.where(self.last_inspection >= 0, self.clock - self.last_inspection, self.clock)
        risk, _ = self.public_risk()
        ratio = ages / self.deadlines
        return {'time_s': self.clock, 'mean_age_ratio': float(np.minimum(ratio, 6).mean()),
                'overdue_fraction': float((ratio > 1).mean()), 'maximum_age_s': float(ages.max()),
                'coverage': float(self.ever.mean()), 'public_risk_age': float(np.mean(risk * np.minimum(ratio, 4))),
                'per_role_overdue': {role: float((ratio[[t.kind == role for t in self.tasks]] > 1).mean()) for role in ROLES},
                'per_role_maximum_age_s': {role: float(ages[[t.kind == role for t in self.tasks]].max()) for role in ROLES}}

    def value_features(self):
        metrics = self.public_metrics()
        risk, _ = self.public_risk()
        free = np.mean([a['task'] is None for a in self.agents])
        gap = 1 - self.current_packet['public_delivered_it_kw'] / np.maximum(self.current_packet['public_requested_it_kw'], 1e-12)
        return np.asarray([metrics['mean_age_ratio'], metrics['overdue_fraction'], risk.mean(), risk.max(),
                           1 - metrics['coverage'], free, self.clock / 1800, gap.mean()], dtype=np.float32)

    def reserved(self):
        return {a['task'] for a in self.agents if a['task'] is not None}

    def options(self, agent_index, reserved, guard=False):
        agent = self.agents[agent_index]
        if agent['task'] is not None:
            raise ValueError('Busy inspector cannot choose a new task')
        risk, disagreement = self.public_risk()
        obs = self.state.observer
        ids, features = [], []
        unverified = 1 - self.ever.mean()
        for j in agent['allowed']:
            task = self.tasks[j]
            if j in reserved or not eligible(agent['role'], task):
                continue
            distance = self.graph.path(agent['position'], task.position_m)[0]
            eta = math.ceil((distance + task.service_s) / 5 - 1e-10) * 5
            if self.clock + eta > 1800 + 1e-9:
                continue
            deadline = self.deadlines[j]
            age = self.clock - self.last_inspection[j] if self.last_inspection[j] >= 0 else self.clock
            rack = task.index if task.kind == 'thermal' else task.index * 8
            stamp = obs.temperature_stamp[rack]
            measured_age = self.clock - stamp if stamp >= 0 else self.clock
            role_flags = [float(task.kind == role) for role in ROLES]
            features.append([min(age / deadline, 6), eta / deadline, risk[j], float(not self.ever[j]),
                             min(measured_age / 300, 6), min(disagreement[rack], 4), self.asset_weights[j] / 2,
                             task.service_s / deadline, distance / deadline, *role_flags,
                             self.clock / 1800, (task.index % 8) / 7, unverified])
            ids.append(j)
        values = np.asarray(features, dtype=np.float32).reshape(-1, len(CANDIDATE_FEATURES))
        allowed = np.ones(len(ids), dtype=bool)
        if guard and len(ids):
            urgency = values[:, 0] + values[:, 1]
            if urgency.max() >= .9:
                allowed[:] = False
                allowed[int(np.argmax(urgency))] = True
        return np.asarray(ids, dtype=int), values, allowed

    def step(self, choices):
        if self.done:
            raise ValueError('Episode ended')
        free = {a['id'] for a in self.agents if a['task'] is None}
        if set(choices) != free:
            raise ValueError('One decision per free inspector required')
        reserved = self.reserved()
        proposed = []
        # Validate all choices before mutating any inspector.
        for i, j in choices.items():
            agent = self.agents[i]
            ids, _, _ = self.options(i, reserved)
            if j is None:
                if len(ids):
                    raise ValueError('Cannot idle while an eligible completable task is available')
                continue
            if j not in ids:
                raise ValueError('Invalid profession, area, reservation or horizon choice')
            task = self.tasks[j]
            distance, route = self.graph.path(agent['position'], task.position_m)
            proposed.append((i, j, distance, route))
            reserved.add(j)
        before_coverage = self.ever.mean()
        distance_before = sum(a['travel_m'] for a in self.agents)
        for i, j, distance, route in proposed:
            agent, task = self.agents[i], self.tasks[j]
            agent.update(task=j, remaining_travel_s=distance, remaining_service_s=task.service_s)
            self.assignments.append({'time_s': self.clock, 'agent': i, 'task': j,
                                     'source': list(agent['position']), 'target': list(task.position_m),
                                     'distance_m': distance, 'route': [list(p) for p in route],
                                     'service_s': task.service_s})
        self.tick += 1
        self.clock = (self.tick + 1) * 5.
        completed = []
        alarms = 0
        for agent in self.agents:
            budget = 5.
            if agent['task'] is None:
                agent['idle_s'] += budget
                continue
            travel = min(budget, agent['remaining_travel_s'])
            agent['remaining_travel_s'] -= travel
            agent['travel_m'] += travel
            budget -= travel
            service = min(budget, agent['remaining_service_s'])
            agent['remaining_service_s'] -= service
            agent['service_s'] += service
            budget -= service
            if agent['remaining_travel_s'] <= 1e-8 and agent['remaining_service_s'] <= 1e-8:
                j = agent['task']
                task = self.tasks[j]
                measurement = self.trace.measurement(task, self.tick)
                completed.append({'time_s': self.clock, 'completion': (task, agent['role'], measurement, self.clock)})
                alarm = measured_anomaly(task, measurement)
                alarms += alarm
                self.last_inspection[j], self.ever[j] = self.clock, True
                agent['position'], agent['task'] = task.position_m, None
                agent['completed'] += 1
                agent['idle_s'] += budget
                self.events.append({'time_s': self.clock, 'agent': agent['id'], 'task': j,
                                    'task_id': task.task_id, 'kind': task.kind, 'index': task.index,
                                    'role': agent['role'], 'service_s': task.service_s, 'measurement': measurement,
                                    'calibrated': True, 'access': task.access, 'alarm': bool(alarm)})
        self._update(completed)
        metrics = self.public_metrics()
        gain = float(self.ever.mean() - before_coverage)
        distance = sum(a['travel_m'] for a in self.agents) - distance_before
        reward = (2 * gain - .1 * metrics['overdue_fraction'] - .02 * metrics['mean_age_ratio']
                  - .02 * metrics['public_risk_age'] + .02 * alarms / max(1, len(self.agents))
                  - .0001 * distance / max(1, len(self.agents)))
        metrics.update(public_reward=float(reward), new_coverage=gain, completed_alarm_count=alarms,
                       travel_increment_m=float(distance))
        self.scalar_trace.append(metrics)
        return float(reward)


def heuristic_decisions(env, method):
    if method not in ('cycle', 'urgency', 'risk', 'rolling'):
        raise ValueError('Unknown rule')
    choices, reserved = {}, env.reserved()
    order = np.roll(np.arange(len(env.agents)), -(env.tick % len(env.agents)))
    for i in order:
        agent = env.agents[i]
        if agent['task'] is not None:
            continue
        ids, features, allowed = env.options(int(i), reserved)
        if not len(ids):
            choices[int(i)] = None
            continue
        if method == 'cycle':
            tour = agent['tour']
            for offset in range(len(tour)):
                cursor = (agent['cursor'] + offset) % len(tour)
                if tour[cursor] in ids:
                    chosen = tour[cursor]
                    agent['cursor'] = (cursor + 1) % len(tour)
                    break
            else:
                # Near episode end another task can fit before the tour's
                # next item; choosing it is explicit, not a false completion.
                chosen = int(ids[np.argmin(features[:, 1])])
        elif method == 'urgency':
            chosen = int(ids[np.argmax(features[:, 0] + features[:, 1])])
        elif method == 'risk':
            chosen = int(ids[np.argmax((.01 + features[:, 2]) * features[:, 6] / (.03 + features[:, 1]))])
        else:
            # Bounded two-step receding plan over eight immediate candidates.
            first = ((features[:, 0] + .1 + features[:, 2]) * features[:, 6] /
                     (.1 + features[:, 1]))
            top = np.argsort(-first, kind='stable')[:8]
            scores = []
            for index in top:
                target = env.tasks[int(ids[index])].position_m
                second = []
                for other in top:
                    if other == index:
                        continue
                    task = env.tasks[int(ids[other])]
                    eta = (env.graph.path(target, task.position_m)[0] + task.service_s) / env.deadlines[ids[other]]
                    second.append((features[other, 0] + .1 + features[other, 2]) * features[other, 6] /
                                  (.1 + features[index, 1] + eta))
                scores.append(first[index] + .5 * max(second, default=0))
            chosen = int(ids[int(top[int(np.argmax(scores))])])
        choices[int(i)] = chosen
        reserved.add(chosen)
    return choices
