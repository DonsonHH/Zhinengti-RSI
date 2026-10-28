"""M11.1 prototype: fixed ownership, full cyclic fallback, causal sensing."""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import time

import numpy as np

from .active_inspection import ActiveInspection, ProcessTrace, graph_for
from .aisles import build_agents, DEADLINES
from .coverage_plan import ReturnPlan, InfeasibleCoverage, exact_task_gaps
from .estimation import ObservationState
from .estimation_data import CompositeFacility
from .facility import FacilityConfig
from .inspection_contracts import ServiceEvidence
from .inspection_metrics import evaluate_episode
from .m9 import arrays_digest

SELECTORS = ('cycle', 'risk_free', 'risk_certified', 'untrained_certified')


class DurationTrace(ProcessTrace):
    def __init__(self, seed, scenario, duration, control='feedback'):
        if duration not in (60, 180, 900):
            raise ValueError('Frozen fault duration required')
        self.config = FacilityConfig(fault_start_s=300, fault_end_s=300 + duration)
        self.seed, self.scenario, self.control = seed, scenario, control
        plant = CompositeFacility(self.config, seed, scenario, control)
        keys = ('rack_temperature_c', 'secondary_supply_c', 'primary_supply_c',
                'contact_temperature_c', 'rack_flow_kg_s',
                'public_sensor_temperature_c', 'severity_proxy')
        arrays = {key: [] for key in keys}
        scalar, self.packets = [], []
        initial = plant.stored_energy_kj()
        max_energy, max_electric = 0., 0.
        for _ in range(360):
            row = plant.step()
            self.packets.append(plant.public_telemetry())
            for key in keys:
                arrays[key].append(row[key])
            scalar.append([row[k] for k in ('time_s', 'unserved_it_kw', 'source_power_kw',
                'stored_energy_kj', 'rejected_heat_kw', 'it_power_kw', 'aux_power_kw',
                'conversion_loss_kw', 'contact_loss_kw')])
            max_energy = max(max_energy, abs(row['energy_residual_kj']))
            max_electric = max(max_electric, abs(row['electric_residual_kw']))
        self.arrays = {key: np.stack(value) for key, value in arrays.items()}
        self.arrays['process_scalars'] = np.asarray(scalar)
        self.arrays['asset_weights'] = plant.criticality.copy()
        self.fingerprint = arrays_digest(self.arrays)
        self.metadata = {'racks': 128, 'seed': seed, 'scenario': scenario, 'control': control,
            'fault_duration_s': duration, 'configuration': asdict(self.config),
            'numeric_digest': self.fingerprint, 'initial_energy_kj': initial,
            'maximum_energy_residual_kj': max_energy, 'maximum_electric_residual_kw': max_electric,
            'peak_source_input_MW': max(r[2] for r in scalar) / 1000,
            'IT_nameplate_MW': 32., 'unserved_IT_kwh': sum(r[1] for r in scalar) * 5 / 3600}


@lru_cache(maxsize=4)
def duration_trace(seed, scenario, duration):
    return DurationTrace(seed, scenario, duration)


class CoveredInspection(ActiveInspection):
    def __init__(self, spec, predictor=None):
        self.spec = dict(spec)
        if spec.get('control', 'feedback') != 'feedback':
            raise ValueError('Startup prototype fixes feedback control')
        self.selector = spec['selector']
        if self.selector not in SELECTORS:
            raise ValueError('Unknown prototype selector')
        self.protected = self.selector != 'risk_free'
        self.graph = graph_for(128, spec.get('variant', 'open'))
        self.tasks = self.graph.tasks
        self.agents = build_agents(self.graph, spec.get('resources', 'provisioned'))
        self.deadlines = np.array([DEADLINES[t.kind] for t in self.tasks])
        for agent in self.agents:
            agent['allowed'] = list(agent['tour'])
        self.plans = [ReturnPlan(self.graph, self.tasks, a['tour'], self.deadlines) for a in self.agents]
        self.protected_cursor = np.zeros(len(self.agents), dtype=int)
        self.pending_cursor, self.selected_certificates, self.certificate_cache = {}, [], {}
        self.last_inspection = np.full(len(self.tasks), -1.)
        self.ever = np.zeros(len(self.tasks), dtype=bool)
        self.initial_certificates = [plan.check(5, a['position'], self.last_inspection, 0)
                                     for plan, a in zip(self.plans, self.agents)]
        if not all(c['ok'] for c in self.initial_certificates):
            raise InfeasibleCoverage('Initial uninspected backlog has no declared fallback')
        self.trace = duration_trace(spec['seed'], spec['scenario'], spec['duration_s'])
        self.config = self.trace.config
        self.mode, self.predictor = spec['estimator'], predictor
        if self.mode not in ('nominal', 'history') or self.mode == 'history' and predictor is None:
            raise ValueError('Fixed observation-only predictor required')
        asset = self.trace.arrays['asset_weights']
        self.asset_weights = np.array([asset[t.index] if t.kind == 'thermal'
            else asset.reshape(self.config.pods, 8)[t.index].mean() for t in self.tasks])
        self.state = ObservationState(self.config)
        self.events, self.assignments, self.frames, self.scalar_trace = [], [], [], []
        self.tick, self.clock = 0, 5.
        self.current_history = np.full(128, 50.)
        self.inference_seconds = 0.
        self.measured_alert = {}
        self.supply_monitor = ServiceEvidence(self.config.pods, calibrated=True)
        self._update([])
        self.supply_monitor.update(self.current_packet)
        self.scalar_trace.append(self.public_metrics())
        self.certificate_evaluations, self.certificate_rejections = 0, 0

    def candidate_certificate(self, i, j):
        key = (i, int(j), int(self.protected_cursor[i]))
        if key not in self.certificate_cache:
            result = self.plans[i].check(self.clock, self.agents[i]['position'],
                self.last_inspection, int(self.protected_cursor[i]), int(j))
            self.certificate_cache[key] = result
            self.certificate_evaluations += 1
            self.certificate_rejections += not result['ok']
        return self.certificate_cache[key]

    def options(self, agent_index, reserved, guard=False):
        ids, features, _ = super().options(agent_index, reserved, guard=False)
        if self.selector == 'cycle':
            expected = self.plans[agent_index].tour[self.protected_cursor[agent_index]]
            keep = np.asarray([j == expected for j in ids])
        elif self.protected:
            keep = np.asarray([self.candidate_certificate(agent_index, j)['ok'] for j in ids])
        else:
            keep = np.ones(len(ids), dtype=bool)
        if not len(ids):
            return ids, features, np.ones(0, dtype=bool)
        return ids[keep], features[keep], np.ones(int(keep.sum()), dtype=bool)

    def step(self, choices):
        prepared = {}
        for i, j in choices.items():
            if j is None:
                if self.protected:
                    due = min(max(0., self.last_inspection[k]) + self.deadlines[k] for k in self.plans[i].tour)
                    if due < min(1800., self.clock + 5) - 1e-9:
                        raise InfeasibleCoverage('Protected obligation expires during idle')
                continue
            cert = self.candidate_certificate(i, j)
            if self.protected and not cert['ok']:
                raise InfeasibleCoverage('Unsafe candidate cannot be submitted')
            prepared[i] = {'task': int(j), 'cursor_after': cert.get('cursor_after', int(self.protected_cursor[i])),
                'agent': i, 'clock_s': self.clock, 'position': list(self.agents[i]['position']),
                'baseline_next': self.plans[i].tour[self.protected_cursor[i]],
                'protection_enabled': self.protected, 'certificate': cert}
        count = len(self.events)
        reward = super().step(choices)
        # Commit cursor effects only after atomic physical-dispatch validation.
        for i, row in prepared.items():
            self.pending_cursor[i] = row
            self.selected_certificates.append(row)
        for event in self.events[count:]:
            i, j = event['agent'], event['task']
            pending = self.pending_cursor.pop(i)
            if pending['task'] != j:
                raise RuntimeError('Completion does not match pending owned task')
            if j == pending['baseline_next']:
                self.protected_cursor[i] = (self.protected_cursor[i] + 1) % len(self.plans[i].tour)
            previous = self.measured_alert.get(j)
            current = bool(event['alarm'])
            event['measurement_transition'] = ('new_measured_alert' if current and not previous else
                'persistent_recheck' if current else 'recovery_confirmation' if previous else 'normal_check')
            self.measured_alert[j] = current
        self.certificate_cache.clear()
        self.supply_monitor.update(self.current_packet)
        return reward


def select_owned(env, score_model=None):
    choices, reserved = {}, env.reserved()
    for i in np.roll(np.arange(len(env.agents)), -(env.tick % len(env.agents))):
        i = int(i)
        if env.agents[i]['task'] is not None:
            continue
        ids, features, _ = env.options(i, reserved)
        if not len(ids):
            choices[i] = None
            continue
        if env.selector == 'cycle':
            selected = int(ids[0])
        elif env.selector == 'untrained_certified':
            scores = np.asarray(score_model(features))
            selected = int(ids[int(np.argmax(scores))])
        else:
            scores = (.01 + features[:, 2]) * features[:, 6] / (.03 + features[:, 1])
            selected = int(ids[int(np.argmax(scores))])
        choices[i] = selected
        reserved.add(selected)
    return choices


def run_covered(spec, predictor=None, score_model=None):
    env = CoveredInspection(spec, predictor)
    decisions, decision_seconds = [], 0.
    while not env.done:
        start = time.perf_counter()
        choices = select_owned(env, score_model)
        decision_seconds += time.perf_counter() - start
        decisions.append({'time_s': env.clock, 'choices': {str(i): j for i, j in choices.items()}})
        env.step(choices)
    metrics, by_role, intervals = evaluate_episode(env)
    maxima, gaps = exact_task_gaps(env.tasks, env.events)
    metrics.update(maximum_exact_gap_s=float(maxima.max()),
        maximum_exact_gap_ratio=float(np.max(maxima / env.deadlines)),
        tasks_violating_exact_gap=int((maxima > env.deadlines + 1e-9).sum()),
        decision_CPU_seconds=decision_seconds,
        certificate_candidate_evaluations=env.certificate_evaluations,
        certificate_candidate_rejections=env.certificate_rejections,
        selected_detours=sum(row['task'] != row['baseline_next'] for row in env.selected_certificates))
    if env.protected and metrics['tasks_violating_exact_gap']:
        raise InfeasibleCoverage('Accepted certificate did not preserve episode coverage')
    record = {'specification': spec, 'metrics': metrics, 'by_role': by_role,
        'anomaly_intervals': intervals, 'exact_task_gaps': gaps,
        'initial_certificates': env.initial_certificates,
        'selected_certificates': env.selected_certificates, 'action_log': decisions,
        'assignments': env.assignments, 'inspection_events': env.events,
        'service_evidence_events': env.supply_monitor.events,
        'scalar_trace': env.scalar_trace, 'process': env.trace.metadata,
        'team': {'count': len(env.agents), 'per_agent': [
            {key: a[key] for key in ('id', 'role', 'zone', 'travel_m', 'service_s', 'idle_s', 'completed')}
            for a in env.agents]},
        'scope': 'Prototype with fixed owner permissions; no new policy training, voltage/current readings or fault root-cause claim'}
    return record, env
