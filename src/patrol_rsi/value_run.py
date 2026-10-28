"""One optional global action, then protected tour. Observer is evaluator-only."""
from __future__ import annotations

import time

import numpy as np

from .coverage_plan import InfeasibleCoverage, exact_task_gaps
from .facility_workflow import public_outcome
from .improver_task import ImproverTask, PatrolProgram
from .inspection_metrics import evaluate_episode
from .inspection_value import choose
from .public_dispatch import snapshot
from .opportunity_probe import OneDetourProbe


def run(spec, predictor, actor_score, method='tour', *, score=None, threshold=0., observer=None,
        probe_enabled=False, intervention=None, model_id=None):
    if (observer is not None or probe_enabled or intervention is not None) and method != 'tour':
        raise ValueError('Evaluator hooks are limited to baseline/label generation')
    env = ImproverTask(spec, PatrolProgram(), predictor)
    probe = OneDetourProbe(env) if probe_enabled else None
    decisions, decision_seconds, used, injected = [], 0., False, 0
    for _ in range(359):
        start = time.perf_counter()
        groups = snapshot(env, actor_score)
        choices, logs, consumed = choose(groups, method, used=used, threshold=threshold, score=score)
        decision_seconds += time.perf_counter() - start
        used = used or consumed
        if probe is not None:
            probe.observe(env, groups)
        if observer is not None:
            observer.observe(env, groups)
        if intervention is not None and env.clock == intervention['dispatch_time_s']:
            group = next((g for g in groups if g['agent'] == intervention['agent']), None)
            candidate = next((r for r in group['candidates'] if r['task'] == intervention['task'] and r['is_optional']), None) if group else None
            if candidate is None:
                raise RuntimeError('Evaluator override is not a legal optional action')
            choices[group['agent']] = candidate['task']
            log = next(r for r in logs if r['agent'] == group['agent'])
            log.update(task=candidate['task'], tool='evaluator_one_detour', rule_index=None,
                cue_strength=candidate['cue_strength'], known_measured_alarm=candidate['known_measured_alarm'])
            injected += 1
        decisions.append({'time_s': env.clock, 'choices': {str(i): j for i, j in choices.items()}})
        env.step(choices)
        env.workflow_decisions.extend(logs)
    if intervention is not None and injected != 1:
        raise RuntimeError('Exactly one evaluator intervention required')
    metrics, by_role, intervals = evaluate_episode(env)
    maxima, gaps = exact_task_gaps(env.tasks, env.events)
    metrics.update(maximum_exact_gap_s=float(maxima.max()), maximum_exact_gap_ratio=float(np.max(maxima / env.deadlines)),
        tasks_violating_exact_gap=int((maxima > env.deadlines + 1e-9).sum()), decision_CPU_seconds=decision_seconds,
        estimator_calls=env.estimator_calls, selected_detours=sum(r['task'] != r['baseline_next'] for r in env.selected_certificates))
    if metrics['tasks_violating_exact_gap'] or metrics['selected_detours'] > 1:
        raise InfeasibleCoverage('One-check budget or exact protected coverage failed')
    record = {'specification': dict(spec), 'workflow': PatrolProgram().to_dict(), 'version_id': PatrolProgram().version_id,
        'deployment': {'public_controller': method, 'threshold': threshold, 'model_id': model_id,
            'probe_enabled': probe_enabled, 'evaluator_intervention': intervention, 'global_optional_budget': 1},
        'metrics': metrics, 'by_role': by_role, 'anomaly_intervals': intervals, 'exact_task_gaps': gaps,
        'initial_certificates': env.initial_certificates, 'selected_certificates': env.selected_certificates,
        'action_log': decisions, 'assignments': env.assignments, 'inspection_events': env.events,
        'service_evidence_events': env.supply_monitor.events, 'scalar_trace': env.scalar_trace,
        'workflow_decisions': env.workflow_decisions, 'process': env.trace.metadata,
        'workflow_contract': {'extra_checks_per_owned_round': 2, 'global_optional_budget': 1,
            'task_weights': env.asset_weights.tolist(), 'deadlines_s': env.deadlines.tolist(), 'cue_threshold': .05},
        'team': {'count': len(env.agents), 'per_agent': [
            {k: a[k] for k in ('id', 'role', 'zone', 'travel_m', 'service_s', 'idle_s', 'completed')} for a in env.agents]}}
    record['public_selection'] = public_outcome(record, env.asset_weights)
    record['evaluator_probe'] = probe.finish(intervals) if probe is not None else None
    return record, env
