"""M12-B task layer: typed signals, explicit return and phase-varied traces."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import time

import numpy as np

from .active_inspection import ProcessTrace, graph_for
from .aisles import DEADLINES, build_agents
from .covered_inspection import CoveredInspection
from .coverage_plan import InfeasibleCoverage, ReturnPlan, exact_task_gaps
from .estimation import ObservationState
from .estimation_data import CompositeFacility
from .facility import FacilityConfig
from .facility_workflow import WorkflowInspection, public_outcome
from .inspection_contracts import ServiceEvidence
from .inspection_metrics import evaluate_episode
from .m9 import arrays_digest

SIGNAL_ROLES = {'age': 'all', 'thermal_excess': 'thermal', 'thermal_disagreement': 'thermal',
    'contact_heat': 'electrical', 'flow_deficit': 'cooling', 'verified_sensor_bias': 'control'}
DOMAINS = {'age': (.25, .5, .75), 'thermal_excess': (.05, .25, .5),
    'thermal_disagreement': (.05, .25, .5), 'contact_heat': (.05, .25, .5),
    'flow_deficit': (.05, .25, .5), 'verified_sensor_bias': (.5,)}
ACTIONS = ('signal', 'urgency', 'ppo', 'return_tour')


@dataclass(frozen=True)
class TypedRule:
    signal: str
    threshold: float
    action: str

    def __post_init__(self):
        if self.signal not in DOMAINS or self.action not in ACTIONS:
            raise ValueError('Unknown typed signal/action')
        if type(self.threshold) not in (int, float) or self.threshold not in DOMAINS[self.signal]:
            raise ValueError('Numeric threshold outside typed domain')


@dataclass(frozen=True)
class PatrolProgram:
    refresh_s: int = 30
    cooldown_s: int = 60
    fallback: str = 'tour'
    rules: tuple[TypedRule, ...] = ()

    def __post_init__(self):
        if type(self.refresh_s) is not int or self.refresh_s not in (30, 60, 120):
            raise ValueError('Invalid refresh interval')
        if type(self.cooldown_s) is not int or self.cooldown_s not in (0, 30, 60, 120):
            raise ValueError('Invalid cooldown')
        if self.fallback not in ('tour', 'ppo'):
            raise ValueError('Fallback is tour or ppo')
        if not isinstance(self.rules, tuple) or len(self.rules) > 3 or not all(isinstance(r, TypedRule) for r in self.rules):
            raise ValueError('At most three typed rules')

    def to_dict(self):
        return {'refresh_s': self.refresh_s, 'cooldown_s': self.cooldown_s,
            'fallback': self.fallback, 'rules': [asdict(r) for r in self.rules]}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {'refresh_s', 'cooldown_s', 'fallback', 'rules'}:
            raise ValueError('Program keys mismatch')
        if not isinstance(value['rules'], list):
            raise ValueError('Rules must be a list')
        if any(not isinstance(r, dict) or set(r) != {'signal', 'threshold', 'action'} for r in value['rules']):
            raise ValueError('Typed rule fields mismatch')
        return cls(value['refresh_s'], value['cooldown_s'], value['fallback'], tuple(TypedRule(**r) for r in value['rules']))

    @property
    def version_id(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


class PhasedTrace(ProcessTrace):
    def __init__(self, seed, scenario, duration, start):
        if duration not in (60, 180, 900) or start not in (150, 335, 520, 705):
            raise ValueError('Frozen injection duration and phase required')
        self.config = FacilityConfig(fault_start_s=start, fault_end_s=start + duration)
        self.seed, self.scenario, self.control = seed, scenario, 'feedback'
        plant = CompositeFacility(self.config, seed, scenario, 'feedback')
        keys = ('rack_temperature_c', 'secondary_supply_c', 'primary_supply_c',
            'contact_temperature_c', 'rack_flow_kg_s', 'public_sensor_temperature_c', 'severity_proxy')
        values = {k: [] for k in keys}
        scalar, self.packets = [], []
        initial = plant.stored_energy_kj()
        max_energy, max_electric = 0., 0.
        for _ in range(360):
            row = plant.step()
            self.packets.append(plant.public_telemetry())
            for key in keys:
                values[key].append(row[key])
            scalar.append([row[k] for k in ('time_s', 'unserved_it_kw', 'source_power_kw',
                'stored_energy_kj', 'rejected_heat_kw', 'it_power_kw', 'aux_power_kw',
                'conversion_loss_kw', 'contact_loss_kw')])
            max_energy = max(max_energy, abs(row['energy_residual_kj']))
            max_electric = max(max_electric, abs(row['electric_residual_kw']))
        self.arrays = {k: np.stack(v) for k, v in values.items()}
        self.arrays['process_scalars'] = np.asarray(scalar)
        self.arrays['asset_weights'] = plant.criticality.copy()
        self.fingerprint = arrays_digest(self.arrays)
        self.metadata = {'racks': 128, 'seed': seed, 'scenario': scenario, 'control': 'feedback',
            'fault_duration_s': duration, 'fault_start_s': start, 'configuration': asdict(self.config),
            'numeric_digest': self.fingerprint, 'initial_energy_kj': initial,
            'maximum_energy_residual_kj': max_energy, 'maximum_electric_residual_kw': max_electric,
            'peak_source_input_MW': max(r[2] for r in scalar) / 1000, 'IT_nameplate_MW': 32.,
            'unserved_IT_kwh': sum(r[1] for r in scalar) * 5 / 3600}


@lru_cache(maxsize=32)
def phased_trace(seed, scenario, duration, start):
    return PhasedTrace(seed, scenario, duration, start)


class ImproverTask(WorkflowInspection):
    """Keep old classes untouched; initialize the new frozen task contract."""
    def __init__(self, spec, program, predictor=None):
        self.spec, self.workflow = dict(spec), program
        if spec.get('control', 'feedback') != 'feedback' or spec.get('resources', 'provisioned') != 'provisioned':
            raise ValueError('This study fixes plant control and staffing')
        if spec['noise_c'] not in (.2, 1.):
            raise ValueError('Frozen observation noise required')
        self.selector, self.protected = 'risk_certified', True
        self.graph = graph_for(128, 'open')
        self.tasks = self.graph.tasks
        self.agents = build_agents(self.graph, 'provisioned')
        self.deadlines = np.asarray([DEADLINES[t.kind] for t in self.tasks])
        for agent in self.agents:
            agent['allowed'] = list(agent['tour'])
        self.plans = [ReturnPlan(self.graph, self.tasks, a['tour'], self.deadlines) for a in self.agents]
        self.protected_cursor = np.zeros(len(self.agents), dtype=int)
        self.pending_cursor, self.selected_certificates, self.certificate_cache = {}, [], {}
        self.last_inspection = np.full(len(self.tasks), -1.)
        self.ever = np.zeros(len(self.tasks), dtype=bool)
        self.initial_certificates = [p.check(5., a['position'], self.last_inspection, 0) for p, a in zip(self.plans, self.agents)]
        if not all(c['ok'] for c in self.initial_certificates):
            raise InfeasibleCoverage('Initial backlog cannot satisfy protected tour')
        self.trace = phased_trace(spec['seed'], spec['scenario'], spec['duration_s'], spec['fault_start_s'])
        self.config = self.trace.config
        self.mode, self.predictor = spec['estimator'], predictor
        if self.mode not in ('nominal', 'history') or self.mode == 'history' and predictor is None:
            raise ValueError('Frozen observation-only estimator required')
        asset = self.trace.arrays['asset_weights']
        self.asset_weights = np.asarray([asset[t.index] if t.kind == 'thermal'
            else asset.reshape(16, 8)[t.index].mean() for t in self.tasks])
        self.state = ObservationState(self.config)
        self.events, self.assignments, self.frames, self.scalar_trace = [], [], [], []
        self.tick, self.clock = 0, 5.
        self.current_history = np.full(128, 50.)
        self.inference_seconds, self.estimator_calls = 0., 0
        self.workflow_decisions = []
        self.measured_alert = {}
        self.supply_monitor = ServiceEvidence(16, calibrated=True)
        self._update([])
        self.supply_monitor.update(self.current_packet)
        self.scalar_trace.append(self.public_metrics())
        self.certificate_evaluations, self.certificate_rejections = 0, 0
        self.detours_used = np.zeros(len(self.agents), dtype=int)
        self.baseline_completed = np.zeros(len(self.agents), dtype=int)

    def packet(self):
        original = self.trace.packets[self.tick]
        result = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in original.items()}
        rng = np.random.default_rng(4000000 + self.spec['seed'] * 1000 + self.tick)
        result['temperature_c'] += rng.normal(0, self.spec['noise_c'], len(result['temperature_c']))
        return result

    def options(self, agent_index, reserved, guard=False):
        ids, features, masks = super().options(agent_index, reserved, guard)
        baseline = self.plans[agent_index].tour[self.protected_cursor[agent_index]]
        if baseline not in ids:
            # Common mission-tail policy: no detour after the next tour item
            # can no longer finish. Idle still undergoes the deadline check.
            return ids[:0], features[:0], masks[:0]
        return ids, features, masks


def typed_signal(env, ids, features, signal):
    role = SIGNAL_ROLES[signal]
    valid = np.asarray([role == 'all' or env.tasks[int(j)].kind == role for j in ids])
    if signal == 'age':
        return features[:, 0].astype(float), valid
    obs = env.state.observer
    temperature = obs.predicted_temperature if env.mode == 'nominal' else env.current_history
    values = []
    for j in ids:
        task = env.tasks[int(j)]
        if task.kind != role:
            values.append(0.)
        elif signal == 'thermal_excess':
            values.append(max(temperature[task.index] - 65, 0) / 20)
        elif signal == 'thermal_disagreement':
            values.append(abs(temperature[task.index] - obs.predicted_temperature[task.index]) / 20)
        elif signal == 'contact_heat':
            values.append(max(obs.last_contact[task.index] - 60, 0) / 30)
        elif signal == 'flow_deficit':
            values.append(max(1 - obs.last_flow[task.index] / 33.6, 0))
        else:
            values.append(float(obs.sensor_bias_verified[task.index]))
    return np.asarray(values), valid


def choose_program(env, scorer, actor_tool='ppo'):
    if actor_tool not in ('ppo', 'urgency'):
        raise ValueError('Unknown actor intervention')
    choices, logs, reserved = {}, [], env.reserved()
    for raw_i in np.roll(np.arange(len(env.agents)), -(env.tick % len(env.agents))):
        i = int(raw_i)
        if env.agents[i]['task'] is not None:
            continue
        ids, f, _ = env.options(i, reserved)
        if not len(ids):
            choices[i] = None
            continue
        scores = np.asarray(scorer(f))
        if scores.shape != (len(ids),) or not np.isfinite(scores).all():
            raise ValueError('Invalid frozen actor scores')
        actor = int(ids[int(np.argmax(scores))])
        baseline = env.plans[i].tour[env.protected_cursor[i]]
        selected, tool, used = None, None, None
        for k, rule in enumerate(env.workflow.rules):
            signal, valid = typed_signal(env, ids, f, rule.signal)
            flags = valid & (signal >= rule.threshold)
            if not flags.any():
                continue
            candidates = np.flatnonzero(flags)
            if rule.action == 'return_tour':
                selected = int(baseline)
            elif rule.action == 'ppo' and actor_tool == 'ppo':
                selected = int(ids[candidates[int(np.argmax(scores[candidates]))]])
            else:
                values = f[:, 0] + f[:, 1] if rule.action in ('urgency', 'ppo') else (.01 + signal) * f[:, 6] / (.03 + f[:, 1])
                selected = int(ids[candidates[int(np.argmax(values[candidates]))]])
            tool, used = rule.action if rule.action != 'ppo' or actor_tool == 'ppo' else 'urgency', k
            break
        if selected is None:
            if env.workflow.fallback == 'tour':
                selected, tool = int(baseline), 'return_tour'
            elif actor_tool == 'ppo':
                selected, tool = actor, 'ppo'
            else:
                selected, tool = int(ids[int(np.argmax(f[:, 0] + f[:, 1]))]), 'urgency'
        choices[i] = selected
        reserved.add(selected)
        logs.append({'time_s': env.clock, 'agent': i, 'task': selected, 'baseline_next': int(baseline),
            'actor_task': actor, 'tool': tool, 'rule_index': used, 'legal_candidates': len(ids),
            'detours_used_before': int(env.detours_used[i]), 'baseline_completed_before': int(env.baseline_completed[i])})
    return choices, logs


def run_program(spec, program, predictor, scorer, actor_tool='ppo'):
    env = ImproverTask(spec, program, predictor)
    action_log, tool_observations, elapsed = [], [], 0.
    while not env.done:
        if int(env.clock) % 30 == 0:
            estimate = env.current_history if env.mode == 'history' else env.state.observer.predicted_temperature
            difference = abs(estimate - env.state.observer.predicted_temperature)
            tool_observations.append({'time_s': env.clock, 'mean_abs_difference_degC': float(difference.mean()),
                'peak_abs_difference_degC': float(difference.max())})
        start = time.perf_counter()
        choices, logs = choose_program(env, scorer, actor_tool)
        elapsed += time.perf_counter() - start
        action_log.append({'time_s': env.clock, 'choices': {str(i): j for i, j in choices.items()}})
        env.step(choices)
        env.workflow_decisions.extend(logs)
    metrics, by_role, intervals = evaluate_episode(env)
    maxima, gaps = exact_task_gaps(env.tasks, env.events)
    metrics.update(maximum_exact_gap_s=float(maxima.max()), maximum_exact_gap_ratio=float(np.max(maxima / env.deadlines)),
        tasks_violating_exact_gap=int((maxima > env.deadlines + 1e-9).sum()), decision_CPU_seconds=elapsed,
        estimator_calls=env.estimator_calls, selected_detours=sum(r['task'] != r['baseline_next'] for r in env.selected_certificates))
    if metrics['tasks_violating_exact_gap']:
        raise InfeasibleCoverage('Typed program violated exact protected revisit')
    record = {'specification': dict(spec), 'workflow': program.to_dict(), 'version_id': program.version_id,
        'deployment': {'actor_tool': actor_tool}, 'metrics': metrics, 'by_role': by_role,
        'anomaly_intervals': intervals, 'exact_task_gaps': gaps, 'initial_certificates': env.initial_certificates,
        'selected_certificates': env.selected_certificates, 'action_log': action_log, 'assignments': env.assignments,
        'inspection_events': env.events, 'service_evidence_events': env.supply_monitor.events,
        'scalar_trace': env.scalar_trace, 'workflow_decisions': env.workflow_decisions, 'process': env.trace.metadata,
        'thermal_tool_observations': tool_observations,
        'workflow_contract': {'extra_checks_per_owned_round': 2, 'task_weights': env.asset_weights.tolist(),
            'deadlines_s': env.deadlines.tolist(), 'tail': 'idle if baseline item cannot finish; all methods share this'},
        'team': {'count': len(env.agents), 'per_agent': [
            {key: a[key] for key in ('id', 'role', 'zone', 'travel_m', 'service_s', 'idle_s', 'completed')}
            for a in env.agents]}}
    record['public_selection'] = public_outcome(record, env.asset_weights)
    return record, env
