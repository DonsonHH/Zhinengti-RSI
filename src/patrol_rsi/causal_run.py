"""Full-episode deployment of the new public stopping rule, frozen task."""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from .causal_observation import ObservedTask,capture_public
from .coverage_plan import InfeasibleCoverage,exact_task_gaps
from .inspection_metrics import evaluate_episode
from .inspection_timing import public_packet
from .improver_task import PatrolProgram
from .public_dispatch import snapshot,choose as tour_choose
from .public_stopping import StopConfig,decision
from .robust_timing import RobustTimingPolicy


METHODS=('tour','evidence','lookahead','conservative','revised')


def run(spec,predictor,actor_score,method,config=StopConfig(),*,capture_times=()):
    if method not in METHODS:raise ValueError('Five fixed methods')
    env=ObservedTask(spec,PatrolProgram(),predictor)
    old=RobustTimingPolicy({'method':'lookahead','conservative':True} if method=='conservative' else
        {'method':method} if method in ('tour','evidence','lookahead') else {'method':'tour'})
    used=False;decisions=[];actions=[];bundles=[];stamps=[];wall=0.
    for _ in range(359):
        started=time.perf_counter();groups=snapshot(env,actor_score)
        bundle=capture_public(env) if not used and method=='revised' or env.clock in capture_times else None
        if bundle is not None:
            stamps.append({'time_s':env.clock,'sha256':hashlib.sha256(json.dumps(bundle,sort_keys=True).encode()).hexdigest(),
                'latest_context_cutoff_s':max((r[k]['context_cutoff_s'] for r in bundle['tasks']
                    for k in ('nominal','cvae_raw','history_assimilated') if r[k]['available']),default=None)})
        if env.clock in capture_times:bundles.append(bundle)
        if method=='revised':
            value=decision(bundle,used,config) if not used else {'action':'TOUR','selected':None}
            choices,logs=tour_choose(groups,'tour');selected=value['selected'];consumed=False
            if selected:
                group=next(g for g in groups if g['agent']==selected['agent'])
                if not any(r['task']==selected['task'] and r['is_optional'] for r in group['candidates']):
                    raise RuntimeError('Public forecast changed current legality')
                choices[group['agent']]=selected['task'];consumed=True
                log=next(r for r in logs if r['agent']==group['agent'])
                log.update(task=selected['task'],tool='causal_stopping',skill_value=selected['net_score'])
            decisions.append(dict(value,time_s=env.clock))
        else:
            packet=public_packet(env) if old.needs_observations(used) else {'time_s':env.clock}
            choices,logs,consumed,value=old.choose(groups,packet,used)
            decisions.append(value)
        wall+=time.perf_counter()-started;used=used or consumed
        actions.append({'time_s':env.clock,'choices':{str(i):j for i,j in choices.items()}})
        env.step(choices);env.workflow_decisions.extend(logs)
    metrics,roles,windows=evaluate_episode(env)
    maxima,gaps=exact_task_gaps(env.tasks,env.events)
    metrics.update(tasks_violating_exact_gap=int(np.sum(maxima>env.deadlines+1e-9)),
        selected_detours=sum(r['task']!=r['baseline_next'] for r in env.selected_certificates),
        decision_wall_s=wall,estimator_calls=env.estimator_calls)
    if metrics['tasks_violating_exact_gap'] or metrics['selected_detours']>1:
        raise InfeasibleCoverage('Exact revisit/global optional budget violation')
    return {'specification':dict(spec),'method':method,'metrics':metrics,'by_role':roles,
        'anomaly_intervals':windows,'action_log':actions,'inspection_events':env.events,
        'selected_certificates':env.selected_certificates,'assignments':env.assignments,
        'exact_task_gaps':gaps,'workflow_decisions':env.workflow_decisions,'stopping_decisions':decisions,
        'public_snapshots':bundles,'causal_snapshot_stamps':stamps,'trace_fingerprint':env.trace.fingerprint},env
