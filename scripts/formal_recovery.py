"""Recover a source-proven JSON-only audit failure without extra replays.

This deliberately supports only the observed development failure in code-v3.
It is not a general 'skip N' or an unaudited checkpoint override.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import numpy as np

from patrol_rsi.active_inspection import ProcessTrace,graph_for
from patrol_rsi.aisles import build_agents,DEADLINES
from patrol_rsi.coverage_plan import ReturnPlan
from patrol_rsi.queue_history import history_state
from patrol_rsi.queue_diagnostics import diagnose_queue


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def recovery_proof(folder,rows,job):
    root=folder.parent;frozen=json.loads((folder/'phase-frozen.json').read_bytes());chunk=frozen['chunk']
    if frozen['phase']!='development' or str(job)!='130534':raise ValueError('Recovery restricted to the recorded JSON failure')
    source=root/'code-v3/scripts/audit_formal_study.py';manifest=json.loads((root/'code-v3-manifest.json').read_bytes())
    if digest(source)!=manifest['files']['scripts/audit_formal_study.py']['sha256']:raise ValueError('Old audit source changed')
    old=source.read_text();message='Fresh policy/weight replay differs: public_queue_comparison'
    binding_path=root/f'development-{chunk}-historical-binding.json'
    binding=json.loads(binding_path.read_bytes())
    if binding['job']['JobID']!=str(job)+'_'+str(chunk) or binding['job']['State']!='FAILED' or binding['job']['ExitCode']!='1:0':
        raise ValueError('Historical allocation binding differs')
    for row in binding['files']+binding['logs']:
        path=Path(row['path']);info=path.lstat()
        if (not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_size!=row['size'] or
                info.st_ctime_ns!=row['ctime_ns'] or info.st_mtime_ns!=row['mtime_ns'] or digest(path)!=row['sha256']):
            raise ValueError('Source/log/input changed after historical binding')
    if ('verify_case_index(row,old,fresh)' not in old or
            "'selected_public_snapshots','public_queue_comparison','initial_certificates'" not in old or
            "if not same(old[key],fresh[key])" not in old):raise ValueError('Unexpected old failure order')
    stdout=root/f'jobs/development-audit-{job}_{chunk}.out';stderr=root/f'jobs/development-audit-{job}_{chunk}.err'
    text=stderr.read_text()
    if not text.rstrip().endswith('RuntimeError: '+message) or str(source) not in text:raise ValueError('Failure is not the known serialization comparison')
    first=next((i for i,row in enumerate(rows) if row.get('public_queue_comparison') is not None),None)
    if first!=20 or any(row['definition']['method']!='tour' for row in rows[:16]):raise ValueError('Unexpected first nonempty queue location')
    if f'FORMAL_AUDIT development {chunk} 16' not in stdout.read_text() or f'FORMAL_AUDIT development {chunk} 24' in stdout.read_text():
        raise ValueError('Saved progress does not match proven failure position')
    processes=json.loads((folder/'process-index.json').read_bytes())
    if {row['process_key'] for row in rows[:first]}!=set(processes):raise ValueError('Prior successful prefix did not verify all arrays')
    return {'old_job':str(job)+'_'+str(chunk),'fully_audited_prefix':first,'core_replayed_failure_case':first+1,
        'recovered_weight_replays':first+1,'old_audit_source_sha256':digest(source),
        'old_source_manifest_sha256':digest(root/'code-v3-manifest.json'),
        'case_index_sha256':digest(folder/'case-index.json'),'model_manifest_sha256':digest(folder/'model-inputs.json'),
        'phase_freeze_sha256':digest(folder/'phase-frozen.json'),'stdout_sha256':digest(stdout),'stderr_sha256':digest(stderr),
        'historical_binding_sha256':digest(binding_path),'historical_binding_scope':binding['scope'],
        'proof':'First nonempty integer-key queue is case21. Prior source checks all deterministic metrics, action/event/decision/source/public snapshot fields before that failure. All eight arrays passed in first20. Only queue JSON representation and case21 initial certificate follow the failure site.'}


def saved_environment(record,folder,process):
    """No constructor simulation, step or predictor: read frozen arrays only."""
    graph=graph_for(128,'open');tasks=graph.tasks
    trace=ProcessTrace.__new__(ProcessTrace)
    with np.load(folder/process['path']) as data:trace.arrays={k:data[k].copy() for k in data.files}
    trace.config=graph.config;trace.seed=record['specification']['seed']
    limits=np.asarray([DEADLINES[t.kind] for t in tasks])
    agents=build_agents(graph,'provisioned');plans=[ReturnPlan(graph,tasks,a['tour'],limits) for a in agents]
    initial=[p.check(5.,a['position'],np.full(len(tasks),-1.),0) for p,a in zip(plans,agents)]
    return SimpleNamespace(graph=graph,tasks=tasks,spec=record['specification'],trace=trace,
        deadlines=limits,protected=True,plans=plans,initial_certificates=initial)


def recovered_queue(record,env):
    action=next((r for r in record['workflow_decisions'] if r['task']!=r['baseline_next']),None)
    if action is None:return None
    now=action['time_s'];agent=build_agents(env.graph,'provisioned')[action['agent']]
    # These are the original public weights, not future event severity.
    asset=env.trace.arrays['asset_weights']
    weights=[asset[t.index] if t.kind=='thermal' else asset.reshape(16,8)[t.index].mean() for t in env.tasks]
    state=history_state(env.plans[agent['id']],now=now,agent=agent,
        assignments=[r for r in record['assignments'] if r['time_s']<now],
        events=[r for r in record['inspection_events'] if r['time_s']<=now],weights=weights,global_used=False)
    return diagnose_queue(env.plans[agent['id']],state,action['task'])
