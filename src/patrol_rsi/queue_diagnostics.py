"""Public deterministic queue comparison; no process, model or event labels."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .coverage_plan import ReturnPlan


@dataclass(frozen=True)
class Measurement:
    time_s: float
    value: float
    unit: str


@dataclass(frozen=True)
class TaskEvidence:
    task: int
    latest: Measurement | None = None
    previous: Measurement | None = None
    alarm: bool = False


@dataclass(frozen=True)
class QueueState:
    time_s: float
    agent: int
    role: str
    position_m: tuple[float, float]
    cursor: int
    last_checks_s: tuple[float, ...]
    task_weights: tuple[float, ...]
    evidence: tuple[TaskEvidence, ...]
    detours_used: int = 0
    baseline_completed: int = 0
    global_used: bool = False


@dataclass(frozen=True)
class ProxyConfig:
    """One transparent development prototype, not a fitted risk model."""
    age_coefficient: float = .05
    topology_coefficient: float = .25
    execution_coefficient: float = .05

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in (
                self.age_coefficient, self.topology_coefficient, self.execution_coefficient)):
            raise ValueError('Nonnegative finite prototype coefficients required')


def _public_features(task, evidence, now, finish):
    latest, previous = evidence.latest, evidence.previous
    if latest is None:
        return {'last_qualified_measurement_s': None, 'measurement_age_s': None,
            'measured_value': None, 'unit': None, 'trend_per_s': None,
            'projected_value': None, 'professional_signal': 0., 'rising_trend': 0.,
            'information_status': 'no_qualified_reading; zero signal is not proof of normality'}
    trend = (latest.value - previous.value) / (latest.time_s - previous.time_s) if previous else 0.
    trend = max(-.2, min(.2, trend)) if task.kind == 'thermal' else 0.
    projected = latest.value + trend * min(120., max(0., finish - latest.time_s))
    if task.kind == 'thermal':
        projected = max(35., min(110., projected))
        signal = max(max(latest.value, projected) - 65., 0.) / 20
    elif task.kind == 'electrical':
        signal = max(latest.value - 60., 0.) / 30
    elif task.kind == 'cooling':
        signal = max(1 - latest.value / 33.6, 0.)
    else:
        signal = float(abs(latest.value) > 1.)
    return {'last_qualified_measurement_s': latest.time_s, 'measurement_age_s': now - latest.time_s,
        'measured_value': latest.value, 'unit': latest.unit, 'trend_per_s': trend,
        'projected_value': projected, 'professional_signal': signal,
        'rising_trend': max(trend, 0.) / .2,
        'information_status': 'qualified_history_only; no archived CVAE/telemetry snapshot'}


def collateral_proxy(plan: ReturnPlan, state: QueueState, diagnostic, *, config=ProxyConfig()):
    """A whole-mission delay-exposure proxy; dimensionless, uncalibrated.

    Costs sum weight * mission_delay_exposure/300 * (signal + c_age*age/deadline
    + c_topology*linked signal/trend/alarm in the same public pod). This is neither
    a calibrated failure probability nor the true all-episode value function.
    """
    if not diagnostic['legal']:
        raise ValueError('An illegal action has no public proxy score')
    target = diagnostic['target']
    # Rebuild the public comparison: callers cannot insert future outcomes or
    # alter predicted queue times in the supplied diagnostic.
    if diagnostic != diagnose_queue(plan, state, target):
        raise ValueError('Diagnostic must match the exact public state')
    evidence = {r.task:r for r in state.evidence}
    finishes = {r['task']:r['detour_tour_completion_s'] for r in diagnostic['all_normal_tour_comparisons']}
    features = {j:_public_features(plan.tasks[j], evidence.get(j, TaskEvidence(j)), state.time_s,
        finishes.get(j, state.time_s)) for j in range(len(plan.tasks))}
    pod = lambda j: plan.tasks[j].index // 8 if plan.tasks[j].kind == 'thermal' else plan.tasks[j].index
    costs = []
    for row in diagnostic['affected_tasks']:
        j = row['task'];f = features[j]
        related = [k for k in range(len(plan.tasks)) if k != j and pod(k) == pod(j)]
        topology = max((max(features[k]['rising_trend'], features[k]['professional_signal'],
            float(evidence[k].alarm) if k in evidence else 0.) for k in related), default=0.)
        age = state.time_s - max(0., state.last_checks_s[j])
        weight = state.task_weights[j] / 2
        risk = f['professional_signal'] + config.age_coefficient * age / plan.deadlines[j] + config.topology_coefficient * topology
        first_cost = weight * row['delay_s'] / 300 * risk
        cost = weight * row['mission_delay_exposure_s'] / 300 * risk
        costs.append(dict(f, task=j, task_id=plan.tasks[j].task_id, pod=pod(j),
            weight_scaled=weight, deadline_remaining_s=max(0., state.last_checks_s[j]) + plan.deadlines[j] - state.time_s,
            inspection_age_s=age, delay_s=row['delay_s'], linked_public_tasks=related,
            mission_delay_exposure_s=row['mission_delay_exposure_s'],
            linked_signal_trend_or_alarm=topology, risk_proxy=risk,
            first_round_loss_increment=first_cost, estimated_loss_increment=cost))
    target_features = _public_features(plan.tasks[target], evidence.get(target, TaskEvidence(target)),
        state.time_s, diagnostic['target_detour_completion_s'])
    benefit = state.task_weights[target] / 2 * target_features['professional_signal'] * max(diagnostic['target_lead_s'], 0.) / 300
    execution = config.execution_coefficient * max(diagnostic['extra_full_round_elapsed_s'], 0.) / 300
    collateral = sum(r['estimated_loss_increment'] for r in costs)
    net = benefit - execution - collateral
    return {'view': 'public_proxy_not_future_loss', 'normalization_s': 300.,
        'scope': 'Delay exposure through 1800s with current risk held; repeated exposure is not a probability of distinct future misses',
        'target_features': target_features, 'target_benefit_proxy': benefit,
        'collateral_cost': collateral, 'execution_cost': execution, 'net_check_vs_tour_proxy': net,
        'task_costs': costs,
        'tour_slack_veto_development_candidate': {'tour_soon_s': 60.,
            'veto': diagnostic['target_tour_eta_s'] <= 60 and net <= 0,
            'selected_or_validated_threshold': False},
        'wait_value_scope': 'Only TOUR continuation valued here; future adaptive WAIT value remains unimplemented'}


def _route(plan, state, target, horizon_s):
    order = plan.tour[state.cursor:] + plan.tour[:state.cursor]
    sequence = ([target] if target is not None else []) + list(order)
    now, position, first, timeline = state.time_s, state.position_m, {}, []
    travel, service = 0., 0.
    for index, task in enumerate(sequence):
        distance = float(plan.graph.path(position, plan.tasks[task].position_m)[0])
        duration = plan.travel_service(position, task)
        now += duration
        travel += distance
        service += plan.tasks[task].service_s
        optional = target is not None and index == 0
        if not optional:
            first[task] = now
        timeline.append({'task': int(task), 'time_s': float(now), 'optional': optional,
            'within_horizon': now <= horizon_s})
        position = plan.tasks[task].position_m
    # First full round is always shown, even if it extends past the mission.
    full_round = list(timeline)
    end_of_round = now
    while now < horizon_s:
        for task in order:
            now += plan.travel_service(position, task)
            if now > horizon_s:
                break
            timeline.append({'task': int(task), 'time_s': float(now), 'optional': False, 'within_horizon': True})
            position = plan.tasks[task].position_m
    return {'first_tour_completion_s': first, 'first_round': full_round,
        'mission_completions': [r for r in timeline if r['within_horizon']],
        'first_round_end_s': float(end_of_round), 'first_round_travel_m': travel,
        'first_round_service_s': float(service)}


def diagnose_queue(plan: ReturnPlan, state: QueueState, target: int, *, horizon_s=1800.):
    """Compare DETOUR + unchanged tour against TOUR at one free dispatch.

    The static ReturnPlan supplies graph, ownership, qualifications and costs.
    QueueState contains only already obtained readings and schedule state.
    No argument admits a simulator object, scenario or future anomaly window.
    """
    if type(state) is not QueueState:
        raise TypeError('An explicit public QueueState is required')
    if not math.isfinite(plan.period) or plan.period <= 0:
        raise ValueError('Queue plans must make positive finite progress')
    if not math.isfinite(state.time_s) or state.time_s < 5 or state.time_s % 5:
        raise ValueError('Dispatch time must lie on the existing 5s grid')
    if horizon_s != 1800. or state.time_s >= horizon_s:
        raise ValueError('This interface keeps the frozen 1800s mission')
    if not 0 <= state.cursor < len(plan.tour):
        raise ValueError('Invalid owned queue cursor')
    if len(state.last_checks_s) != len(plan.tasks) or len(state.task_weights) != len(plan.tasks):
        raise ValueError('Public task vectors do not match the static plan')
    if any(not math.isfinite(t) or t > state.time_s for t in state.last_checks_s):
        raise ValueError('Future or invalid inspection time')
    if any(not math.isfinite(w) or w < 0 for w in state.task_weights):
        raise ValueError('Finite nonnegative public weights required')
    if type(state.global_used) is not bool or any(type(v) is not int or v < 0
            for v in (state.agent, state.detours_used, state.baseline_completed)):
        raise ValueError('Invalid public budget or agent counters')
    if len(state.position_m) != 2 or any(not math.isfinite(v) for v in state.position_m):
        raise ValueError('Invalid public dispatch position')
    seen = set()
    for evidence in state.evidence:
        if type(evidence) is not TaskEvidence or type(evidence.task) is not int or not 0 <= evidence.task < len(plan.tasks):
            raise ValueError('Typed evidence for a known task required')
        if evidence.task in seen:
            raise ValueError('Duplicate public evidence')
        seen.add(evidence.task)
        if evidence.previous is not None and evidence.latest is None:
            raise ValueError('Previous reading without latest reading')
        for reading in (evidence.latest, evidence.previous):
            if reading is not None and (type(reading) is not Measurement or
                    not math.isfinite(reading.time_s) or not 0 <= reading.time_s <= state.time_s or
                    not math.isfinite(reading.value) or
                    reading.unit != ('kg/s' if plan.tasks[evidence.task].kind == 'cooling' else 'degC')):
                raise ValueError('Only causal measurements with explicit correct units are accepted')
        if evidence.previous is not None and evidence.previous.time_s >= evidence.latest.time_s:
            raise ValueError('Public measurement timestamps must increase')
        if evidence.latest is not None and evidence.latest.time_s != state.last_checks_s[evidence.task]:
            raise ValueError('Reference reading and last qualified check must agree')
    if type(target) is not int or target not in plan.tour:
        return {'legal': False, 'reason': 'target_outside_owned_tour'}
    task = plan.tasks[target]
    if state.role != task.required_role or task.access != 'external_sensing':
        return {'legal': False, 'reason': 'qualification_or_access_mismatch'}
    if target == plan.tour[state.cursor]:
        return {'legal': False, 'reason': 'not_an_extra_check'}
    if state.global_used or state.detours_used >= 2:
        return {'legal': False, 'reason': 'extra_check_budget_used'}
    if state.last_checks_s[target] >= 0 and state.time_s - state.last_checks_s[target] < 60:
        return {'legal': False, 'reason': 'extra_check_cooldown'}
    if any(r.task == target and r.alarm for r in state.evidence):
        return {'legal': False, 'reason': 'already_measured_alarm'}
    certificate = plan.check(state.time_s, state.position_m, state.last_checks_s, state.cursor, target)
    baseline = plan.tour[state.cursor]
    base_certificate = plan.check(state.time_s, state.position_m, state.last_checks_s, state.cursor, baseline)
    if state.time_s + plan.travel_service(state.position_m, baseline) > horizon_s:
        return {'legal': False, 'reason': 'baseline_finishes_after_mission'}
    if not certificate['ok'] or not base_certificate['ok']:
        return {'legal': False, 'reason': 'whole_return_plan_infeasible',
            'detour_certificate': certificate, 'tour_certificate': base_certificate}
    if certificate['candidate_finish_s'] > horizon_s:
        return {'legal': False, 'reason': 'candidate_finishes_after_mission'}
    tour = _route(plan, state, None, horizon_s)
    detour = _route(plan, state, target, horizon_s)
    rows = []
    for j in plan.tour[state.cursor:] + plan.tour[:state.cursor]:
        before, after = tour['first_tour_completion_s'][j], detour['first_tour_completion_s'][j]
        visits = []
        old, new = before, after
        while old <= horizon_s:
            visits.append({'tour_s': old, 'detour_s': new,
                'additional_unchecked_s_within_mission': max(0., min(new, horizon_s) - old)})
            old += plan.period
            new += plan.period
        rows.append({'task': int(j), 'task_id': plan.tasks[j].task_id,
            'tour_completion_s': before, 'detour_tour_completion_s': after,
            'delay_s': float(after - before), 'deadline_s': float(plan.deadlines[j]),
            'delayed_normal_visits': visits,
            'mission_delay_exposure_s': sum(r['additional_unchecked_s_within_mission'] for r in visits),
            'last_check_s': state.last_checks_s[j],
            'original_due_s': float(max(0., state.last_checks_s[j]) + plan.deadlines[j])})
    return {'legal': True, 'view': 'public_schedule_only', 'time_s': state.time_s,
        'agent': state.agent, 'target': target,
        'target_tour_completion_s': tour['first_tour_completion_s'][target],
        'target_detour_completion_s': certificate['candidate_finish_s'],
        'target_lead_s': tour['first_tour_completion_s'][target] - certificate['candidate_finish_s'],
        'target_tour_eta_s': tour['first_tour_completion_s'][target] - state.time_s,
        'affected_tasks': [r for r in rows if r['task'] != target and r['delay_s'] > 0],
        'all_normal_tour_comparisons': rows, 'tour_plan': tour, 'detour_plan': detour,
        'extra_full_round_travel_m': detour['first_round_travel_m'] - tour['first_round_travel_m'],
        'extra_full_round_service_s': detour['first_round_service_s'] - tour['first_round_service_s'],
        'extra_full_round_elapsed_s': detour['first_round_end_s'] - tour['first_round_end_s'],
        'tour_certificate': base_certificate, 'detour_certificate': certificate,
        'qualifications_checked': True, 'global_optional_budget': 1}
