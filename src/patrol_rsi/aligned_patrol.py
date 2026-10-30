"""Paired patrol adapter with one common optional-check gate and tool receipts."""
from __future__ import annotations

import time

import numpy as np

from .coverage_plan import exact_task_gaps,InfeasibleCoverage
from .improver_task import ImproverTask,PatrolProgram,TypedRule,choose_program
from .inspection_metrics import evaluate_episode


def arm_programs():
    return {'Tour':PatrolProgram(),'frozen_PPO':PatrolProgram(fallback='ppo'),
        'human_professional':PatrolProgram(rules=(TypedRule('thermal_excess',.05,'signal'),TypedRule('age',.75,'return_tour'))),
        'fixed_improver_retained':PatrolProgram(rules=(TypedRule('flow_deficit',.25,'signal'),TypedRule('thermal_excess',.05,'urgency'),TypedRule('contact_heat',.05,'return_tour'))),
        'revised_PPO_evidence':PatrolProgram(fallback='ppo',rules=(TypedRule('flow_deficit',.05,'signal'),TypedRule('thermal_excess',.05,'signal'),TypedRule('verified_sensor_bias',.5,'signal')))}


def common_one_check(choices,logs,used):
    """First in shared rotating dispatch order; no arm gets extra quota."""
    result=dict(choices);receipts=[];consumed=False
    for original in logs:
        row=dict(original);optional=row['task']!=row['baseline_next']
        row.update(proposed_task=row['task'],proposed_tool=row['tool'],one_extra_already_used=bool(used or consumed))
        if used or consumed:
            result[row['agent']]=row['baseline_next'];row.update(task=row['baseline_next'],tool='common_budget_return_tour')
        elif optional:consumed=True
        receipts.append(row)
    return result,receipts,consumed


def run_episode(spec,program,predictor,scorer,*,environment_factory=ImproverTask):
    """No LLM/model update. Formal callers must enforce their reviewed split gate."""
    env=environment_factory(spec,program,predictor);used=False;actions=[];decisions=[];wall=0.
    while not env.done:
        started=time.perf_counter();proposed,logs=choose_program(env,scorer)
        choices,receipts,consumed=common_one_check(proposed,logs,used)
        wall+=time.perf_counter()-started
        actions.append({'time_s':env.clock,'choices':{str(i):j for i,j in choices.items()}})
        env.step(choices);env.workflow_decisions.extend(receipts);decisions.extend(receipts);used|=consumed
    metrics,roles,intervals=evaluate_episode(env);maxima,gaps=exact_task_gaps(env.tasks,env.events)
    extras=sum(v['task']!=v['baseline_next'] for v in env.selected_certificates)
    if extras>1 or np.any(maxima>env.deadlines+1e-9):raise InfeasibleCoverage('Common oneCHECK/revisit contract violated')
    metrics.update(selected_detours=extras,decision_CPU_seconds=wall,estimator_calls=env.estimator_calls,
        tasks_violating_exact_gap=int(np.sum(maxima>env.deadlines+1e-9)))
    counts={name:sum(v['tool']==name for v in decisions) for name in sorted({v['tool'] for v in decisions})}
    return {'specification':dict(spec),'workflow':program.to_dict(),'version_id':program.version_id,'metrics':metrics,'by_role':roles,
        'anomaly_intervals':intervals,'action_log':actions,'inspection_events':env.events,'exact_task_gaps':gaps,
        'selected_certificates':env.selected_certificates,'workflow_decisions':decisions,'assignments':env.assignments,
        'usage':{'dispatches':len(decisions),'tool_counts':counts,'actual_PPO_dispatches':counts.get('ppo',0),
            'PPO_multichoice_dispatches':sum(v['tool']=='ppo' and v['legal_candidates']>1 for v in decisions),
            'total_actor_scoring_calls':len(decisions),'exclusive_shadow_actor_scoring_calls':len(decisions)-counts.get('ppo',0),
            'global_optional_checks':extras,'estimator_calls':env.estimator_calls},
        'process_fingerprint':env.trace.fingerprint,'new_neural_training':0}
