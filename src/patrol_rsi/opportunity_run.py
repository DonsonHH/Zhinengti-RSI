"""Public controller runs and separately marked evaluator counterfactuals."""
from __future__ import annotations

import time

import numpy as np

from .coverage_plan import InfeasibleCoverage, exact_task_gaps
from .facility_workflow import public_outcome
from .improver_task import ImproverTask, PatrolProgram
from .inspection_metrics import evaluate_episode
from .public_dispatch import choose, snapshot, METHODS
from .opportunity_probe import OneDetourProbe


def _run(spec, predictor, scorer, method, probe_enabled, intervention):
    env = ImproverTask(spec, PatrolProgram(), predictor)
    probe = OneDetourProbe(env) if probe_enabled else None
    decisions, decision_seconds, injection_count = [], 0., 0
    for _ in range(359):
        start = time.perf_counter()
        groups = snapshot(env, scorer)
        choices, logs = choose(groups, method)
        decision_seconds += time.perf_counter() - start
        if probe is not None:
            probe.observe(env, groups)
        if intervention is not None and env.clock == intervention['dispatch_time_s']:
            agent, target = intervention['agent'], intervention['task']
            group = next((g for g in groups if g['agent'] == agent), None)
            if group is None or not any(r['task'] == target for r in group['candidates']):
                raise RuntimeError('Counterfactual target not admissible at original decision state')
            choices[agent] = target
            row = next(r for r in logs if r['agent'] == agent)
            candidate = next(r for r in group['candidates'] if r['task'] == target)
            row.update(task=target, tool='evaluator_one_detour', rule_index=None,
                cue_strength=candidate['cue_strength'], known_measured_alarm=candidate['known_measured_alarm'])
            injection_count += 1
        decisions.append({'time_s': env.clock, 'choices': {str(i): j for i, j in choices.items()}})
        env.step(choices)
        env.workflow_decisions.extend(logs)
    if intervention is not None and injection_count != 1:
        raise RuntimeError('Exactly one evaluator intervention must execute')
    metrics, by_role, intervals = evaluate_episode(env)
    maxima, gaps = exact_task_gaps(env.tasks, env.events)
    metrics.update(maximum_exact_gap_s=float(maxima.max()), maximum_exact_gap_ratio=float(np.max(maxima / env.deadlines)),
        tasks_violating_exact_gap=int((maxima > env.deadlines + 1e-9).sum()), decision_CPU_seconds=decision_seconds,
        estimator_calls=env.estimator_calls, selected_detours=sum(r['task'] != r['baseline_next'] for r in env.selected_certificates))
    if metrics['tasks_violating_exact_gap']:
        raise InfeasibleCoverage('Public rule or evaluator branch violated protected coverage')
    record = {'specification': dict(spec), 'workflow': PatrolProgram().to_dict(),
        'version_id': PatrolProgram().version_id,
        'deployment': {'public_controller': method, 'probe_enabled': probe_enabled, 'evaluator_intervention': intervention},
        'metrics': metrics, 'by_role': by_role, 'anomaly_intervals': intervals, 'exact_task_gaps': gaps,
        'initial_certificates': env.initial_certificates, 'selected_certificates': env.selected_certificates,
        'action_log': decisions, 'assignments': env.assignments, 'inspection_events': env.events,
        'service_evidence_events': env.supply_monitor.events, 'scalar_trace': env.scalar_trace,
        'workflow_decisions': env.workflow_decisions, 'process': env.trace.metadata,
        'workflow_contract': {'extra_checks_per_owned_round': 2, 'task_weights': env.asset_weights.tolist(),
            'deadlines_s': env.deadlines.tolist(), 'cue_threshold': .05},
        'team': {'count': len(env.agents), 'per_agent': [
            {k: a[k] for k in ('id', 'role', 'zone', 'travel_m', 'service_s', 'idle_s', 'completed')} for a in env.agents]}}
    record['public_selection'] = public_outcome(record, env.asset_weights)
    record['evaluator_probe'] = probe.finish(intervals) if probe is not None else None
    return record, env


def run_public(spec, predictor, scorer, method, *, probe_enabled=False):
    if method not in METHODS or probe_enabled and method != 'tour':
        raise ValueError('Probe is limited to the declared tour comparator')
    return _run(spec, predictor, scorer, method, probe_enabled, None)


def run_evaluator_branch(spec, predictor, scorer, intervention):
    """Not a deployable controller; future event labels selected this one action."""
    if set(intervention) != {'dispatch_time_s', 'agent', 'task'}:
        raise ValueError('One bounded evaluator override required')
    return _run(spec, predictor, scorer, 'tour', False, intervention)
