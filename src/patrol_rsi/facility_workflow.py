"""Bounded, executable workflow changes above frozen sensing and coverage."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import time

import numpy as np

from .covered_inspection import CoveredInspection
from .coverage_plan import InfeasibleCoverage, exact_task_gaps
from .inspection_metrics import evaluate_episode

ROLES = ('all', 'thermal', 'electrical', 'cooling', 'control')
SIGNALS = ('risk', 'uncertainty', 'age')
RANKERS = ('risk', 'uncertainty', 'urgency')
THRESHOLDS = {'risk': (.05, .25, .5), 'uncertainty': (.05, .25, .5), 'age': (.25, .5, .75)}
REFRESH = (30, 60, 120)
COOLDOWN = (0, 30, 60, 120)
DETOUR_LIMIT = 2


@dataclass(frozen=True)
class Rule:
    role: str
    signal: str
    threshold: float
    ranker: str

    def __post_init__(self):
        if self.role not in ROLES or self.signal not in SIGNALS or self.ranker not in RANKERS:
            raise ValueError('Unknown workflow tool or condition')
        if isinstance(self.threshold, bool) or self.threshold not in THRESHOLDS[self.signal]:
            raise ValueError('Threshold outside frozen grammar')


@dataclass(frozen=True)
class FacilityWorkflow:
    refresh_s: int = 30
    cooldown_s: int = 60
    rules: tuple[Rule, ...] = ()

    def __post_init__(self):
        if type(self.refresh_s) is not int or self.refresh_s not in REFRESH:
            raise ValueError('Invalid estimator refresh interval')
        if type(self.cooldown_s) is not int or self.cooldown_s not in COOLDOWN:
            raise ValueError('Invalid extra-check cooldown')
        if not isinstance(self.rules, tuple) or len(self.rules) > 3 or not all(isinstance(r, Rule) for r in self.rules):
            raise ValueError('At most three typed ordered rules required')

    def to_dict(self):
        return {'refresh_s': self.refresh_s, 'cooldown_s': self.cooldown_s,
                'rules': [asdict(r) for r in self.rules]}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {'refresh_s', 'cooldown_s', 'rules'}:
            raise ValueError('Workflow keys must match the frozen grammar')
        if not isinstance(value['rules'], list) or len(value['rules']) > 3:
            raise ValueError('Rules must be a list of length 0..3')
        for rule in value['rules']:
            if not isinstance(rule, dict) or set(rule) != {'role', 'signal', 'threshold', 'ranker'}:
                raise ValueError('Invalid rule fields')
        return cls(value['refresh_s'], value['cooldown_s'], tuple(Rule(**r) for r in value['rules']))

    @property
    def version_id(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


def rule_scores(features, ranker):
    if ranker == 'risk':
        return (.01 + features[:, 2]) * features[:, 6] / (.03 + features[:, 1])
    if ranker == 'uncertainty':
        return (.01 + features[:, 5]) * features[:, 6] / (.03 + features[:, 1])
    if ranker == 'urgency':
        return features[:, 0] + features[:, 1]
    raise ValueError('Unknown ranker')


def rule_match(rule, features, role):
    if rule.role not in ('all', role):
        return np.zeros(len(features), dtype=bool)
    index = {'risk': 2, 'uncertainty': 5, 'age': 0}[rule.signal]
    return features[:, index] >= rule.threshold


class WorkflowInspection(CoveredInspection):
    def __init__(self, spec, workflow, predictor=None):
        self.workflow = workflow
        self.estimator_calls = 0
        self.workflow_decisions = []
        super().__init__(dict(spec, selector='risk_certified'), predictor)
        self.detours_used = np.zeros(len(self.agents), dtype=int)
        self.baseline_completed = np.zeros(len(self.agents), dtype=int)

    def _update(self, completed):
        self._risk_cache = None
        self.current_packet = self.packet()
        self.observation = self.state.update(self.current_packet, completed)
        if round(self.clock) % 30 == 0:
            self.frames.append(self.state.features(self.current_packet))
            self.frames = self.frames[-4:]
            if self.mode == 'history' and round(self.clock) % self.workflow.refresh_s == 0:
                frames = [self.frames[0]] * (4 - len(self.frames)) + self.frames
                start = time.perf_counter()
                self.current_history = np.asarray(self.predictor(np.concatenate(frames, axis=-1)), dtype=float)
                self.inference_seconds += time.perf_counter() - start
                self.estimator_calls += 1
                if self.current_history.shape != (128,) or not np.isfinite(self.current_history).all():
                    raise ValueError('Invalid observation-only history estimate')
        fresh = self.state.observer.temperature_stamp == self.clock
        self.current_history[fresh] = self.state.observer.last_temperature[fresh]

    def options(self, agent_index, reserved, guard=False):
        ids, features, _ = super().options(agent_index, reserved, guard)
        if not len(ids):
            return ids, features, np.ones(0, dtype=bool)
        baseline = self.plans[agent_index].tour[self.protected_cursor[agent_index]]
        # Mandatory tour obligations are never suppressed by a recheck cooldown.
        recent = np.where(self.last_inspection[ids] >= 0,
                          self.clock - self.last_inspection[ids], np.inf)
        discretionary = (self.detours_used[agent_index] < DETOUR_LIMIT) & (recent >= self.workflow.cooldown_s)
        keep = (ids == baseline) | discretionary
        return ids[keep], features[keep], np.ones(int(keep.sum()), dtype=bool)

    def step(self, choices):
        previous_events, previous_plans = len(self.events), len(self.selected_certificates)
        expected = [p.tour[self.protected_cursor[i]] for i, p in enumerate(self.plans)]
        reward = super().step(choices)
        for row in self.selected_certificates[previous_plans:]:
            if row['task'] != row['baseline_next']:
                self.detours_used[row['agent']] += 1
                if self.detours_used[row['agent']] > DETOUR_LIMIT:
                    raise RuntimeError('Extra-check quota exceeded')
        for event in self.events[previous_events:]:
            i = event['agent']
            if event['task'] == expected[i]:
                self.baseline_completed[i] += 1
                if self.baseline_completed[i] == len(self.plans[i].tour):
                    self.baseline_completed[i] = 0
                    self.detours_used[i] = 0
        return reward


def workflow_choices(env, scorer, *, control='workflow', actor_fallback='ppo', signals_permuted=False):
    if control not in ('workflow', 'cycle') or actor_fallback not in ('ppo', 'urgency'):
        raise ValueError('Unknown deployment intervention')
    choices, reserved, logs = {}, env.reserved(), []
    for raw_i in np.roll(np.arange(len(env.agents)), -(env.tick % len(env.agents))):
        i = int(raw_i)
        if env.agents[i]['task'] is not None:
            continue
        ids, features, _ = env.options(i, reserved)
        if not len(ids):
            choices[i] = None
            continue
        scores = np.asarray(scorer(features))
        if scores.shape != (len(ids),) or not np.isfinite(scores).all():
            raise ValueError('Frozen actor must score the current legal candidate set')
        actor_task = int(ids[int(np.argmax(scores))])
        baseline = env.plans[i].tour[env.protected_cursor[i]]
        selected, tool, rule_index = None, None, None
        signals = features.copy()
        if signals_permuted:
            permutation = np.random.default_rng(8800000 + env.spec['seed'] * 10000 + env.tick * 32 + i).permutation(len(ids))
            signals[:, [2, 5]] = features[permutation][:, [2, 5]]
        if control == 'cycle':
            if baseline not in ids:
                # The workflow option mask must agree with an intentional tail idle.
                raise RuntimeError('Cycle control must restrict options before dispatch')
            selected, tool = int(baseline), 'cycle'
        else:
            for k, rule in enumerate(env.workflow.rules):
                flags = rule_match(rule, signals, env.agents[i]['role'])
                if flags.any():
                    candidates = np.flatnonzero(flags)
                    selected = int(ids[candidates[int(np.argmax(rule_scores(signals[flags], rule.ranker)))]] )
                    tool, rule_index = rule.ranker, k
                    break
            if selected is None:
                selected = actor_task if actor_fallback == 'ppo' else int(ids[int(np.argmax(rule_scores(features, 'urgency')))])
                tool = 'ppo' if actor_fallback == 'ppo' else 'urgency_fallback'
        choices[i] = selected
        reserved.add(selected)
        logs.append({'time_s': env.clock, 'agent': i, 'task': selected, 'baseline_next': int(baseline),
            'actor_task': actor_task, 'tool': tool, 'rule_index': rule_index,
            'legal_candidates': len(ids), 'detours_used_before': int(env.detours_used[i]),
            'baseline_completed_before': int(env.baseline_completed[i])})
    return choices, logs


def public_outcome(record, task_weights):
    """Selection is computed only from public completion/measurement evidence."""
    events, count = record['inspection_events'], len(task_weights)
    last = np.zeros(count)
    weights = np.asarray(task_weights)
    deadlines = np.asarray(record['workflow_contract']['deadlines_s'])
    stamps, used, ages = {}, set(), []
    transitions = {kind: 0 for kind in ('new_measured_alert', 'persistent_recheck', 'recovery_confirmation', 'normal_check')}
    for event in events:
        stamps.setdefault(event['time_s'], []).append(event)
        transitions[event['measurement_transition']] += 1
        if event['alarm']:
            used.add(event['task'])
    for t in range(5, 1801, 5):
        for event in stamps.get(t, []):
            last[event['task']] = t
        ages.append(float(np.average((t - last) / deadlines, weights=weights)))
    return {'violating_tasks': record['metrics']['tasks_violating_exact_gap'],
        'coverage': record['metrics']['final_professional_coverage'],
        'first_alert_tasks': len(used), 'first_alert_task_weight': float(weights[list(sorted(used))].sum()),
        'weighted_age_ratio': float(np.mean(ages)), 'travel_m': record['metrics']['travel_m'],
        'service_s': record['metrics']['service_s'], 'measurement_transitions': transitions,
        'estimator_calls': record['metrics']['estimator_calls'],
        'dispatches': len(record['assignments']), 'extra_checks': record['metrics']['selected_detours'],
        'rule_dispatches': sum(r['rule_index'] is not None for r in record['workflow_decisions']),
        'ppo_dispatches': sum(r['tool'] == 'ppo' for r in record['workflow_decisions']),
        'forced_single_candidate': sum(r['legal_candidates'] == 1 for r in record['workflow_decisions']),
        'different_from_actor': sum(r['task'] != r['actor_task'] for r in record['workflow_decisions'])}


def selection_rank(summary):
    return tuple(round(float(v), 8) for v in (summary['violating_tasks'], -summary['coverage'],
        -summary['first_alert_task_weight'], summary['weighted_age_ratio'], summary['travel_m']))


def run_workflow(spec, workflow, predictor, scorer, *, control='workflow', actor_fallback='ppo', signals_permuted=False):
    env = WorkflowInspection(spec, workflow, predictor)
    if control == 'cycle':
        # CoveredInspection's existing cycle mask, followed by the same quota and cooldown.
        env.selector = 'cycle'
    decisions, elapsed = [], 0.
    while not env.done:
        start = time.perf_counter()
        choices, logs = workflow_choices(env, scorer, control=control,
            actor_fallback=actor_fallback, signals_permuted=signals_permuted)
        elapsed += time.perf_counter() - start
        decisions.append({'time_s': env.clock, 'choices': {str(i): j for i, j in choices.items()}})
        env.step(choices)
        env.workflow_decisions.extend(logs)
    metrics, by_role, intervals = evaluate_episode(env)
    maxima, gaps = exact_task_gaps(env.tasks, env.events)
    metrics.update(maximum_exact_gap_s=float(maxima.max()), maximum_exact_gap_ratio=float(np.max(maxima / env.deadlines)),
        tasks_violating_exact_gap=int((maxima > env.deadlines + 1e-9).sum()), decision_CPU_seconds=elapsed,
        selected_detours=sum(row['task'] != row['baseline_next'] for row in env.selected_certificates),
        estimator_calls=env.estimator_calls)
    if metrics['tasks_violating_exact_gap']:
        raise InfeasibleCoverage('Workflow violated protected coverage')
    record = {'specification': dict(spec), 'workflow': workflow.to_dict(), 'version_id': workflow.version_id,
        'deployment': {'control': control, 'actor_fallback': actor_fallback, 'signals_permuted': signals_permuted},
        'metrics': metrics, 'by_role': by_role, 'anomaly_intervals': intervals,
        'exact_task_gaps': gaps, 'initial_certificates': env.initial_certificates,
        'selected_certificates': env.selected_certificates, 'action_log': decisions,
        'assignments': env.assignments, 'inspection_events': env.events,
        'service_evidence_events': env.supply_monitor.events, 'scalar_trace': env.scalar_trace,
        'workflow_decisions': env.workflow_decisions, 'process': env.trace.metadata,
        'workflow_contract': {'extra_checks_per_owned_round': DETOUR_LIMIT,
            'task_weights': env.asset_weights.tolist(), 'deadlines_s': env.deadlines.tolist()},
        'team': {'count': len(env.agents), 'per_agent': [
            {key: a[key] for key in ('id', 'role', 'zone', 'travel_m', 'service_s', 'idle_s', 'completed')}
            for a in env.agents]}}
    record['public_selection'] = public_outcome(record, env.asset_weights)
    return record, env
