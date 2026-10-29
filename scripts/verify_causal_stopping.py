"""Portable source, protocol, public regression and complete-pilot checks."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile

from patrol_rsi.causal_observation import task_signal,validate_public
from patrol_rsi.public_stopping import StopConfig,decision,public_frontier


def read(path):
    raw=path.read_bytes();return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)


def equal(left,right):
    if isinstance(left,dict):return set(left)==set(right) and all(equal(v,right[k]) for k,v in left.items())
    if isinstance(left,list):return len(left)==len(right) and all(equal(a,b) for a,b in zip(left,right))
    if isinstance(left,float):return isinstance(right,(int,float)) and math.isclose(left,right,rel_tol=1e-10,abs_tol=1e-11)
    return left==right


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.workspace.resolve()
    base=root/'research/2026-10-29-m12c1b31-interface';ev=base/'evidence'
    protected=read(ev/'protected-baseline.json')
    for name,row in protected['files'].items():
        raw=(root/name).read_bytes()
        if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:
            raise RuntimeError('Protected file changed '+name)
    manifest=read(base/'code-v6-manifest.json')
    archive=base/'source/code-v6.tar.gz'
    if archive.exists():
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=manifest['sha256']:
            raise RuntimeError('Actual-Torch source archive fingerprint changed')
        with tarfile.open(archive) as tar:
            members={r.name:r for r in tar.getmembers() if r.isfile()}
            if set(members)!=set(manifest['files']):raise RuntimeError('Archived source members differ')
            for name,row in manifest['files'].items():
                raw=tar.extractfile(members[name]).read()
                if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:
                    raise RuntimeError('Archived source bytes differ')
                if hashlib.sha256((root/name).read_bytes()).hexdigest()!=row['sha256']:
                    raise RuntimeError('Current actual-Torch-tested source drift '+name)
    protocol_path=base/'正式实验冻结方案.json';protocol=read(protocol_path)
    frozen=read(ev/'experiment-freeze.json')
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest()!=frozen['protocol_sha256']:
        raise RuntimeError('Formal configuration changed after freeze')
    pilot=ev/'cluster-reviewed/pilot-v3';final=ev/'cluster-final/audit-v4'
    receipt=read(pilot/'protocol-receipt.json')
    if receipt['protocol']!=protocol or receipt['sha256']!=frozen['protocol_sha256']:
        raise RuntimeError('Pilot did not use the final frozen design')
    tests=read(final/'full-tests.json');audit=read(final/'pilot-audit.json')
    if not tests['success'] or tests['skipped'] or tests['failures'] or tests['errors'] or tests['tests']!=215:
        raise RuntimeError('Actual final Torch suite incomplete')
    if audit['fresh_weight_replayed_cases']!=10 or audit['formal_test_accessed']:
        raise RuntimeError('Final replay scope differs')
    regression=read(pilot/'old-public-regressions.json')['cases']
    for case in regression:
        bundle=validate_public(case['public_state']);when=bundle['time_s'];target=case['target']
        source=task_signal(bundle['tasks'][target],45. if target==156 else 25.)
        if source!=(case['target_signal'],case['signal_source']):raise RuntimeError('Public signal drift')
        frontier=public_frontier(bundle)
        current=next(r for r in frontier if r['task']==target and r['dispatch_time_s']==when)
        if not equal(current,case['current_target_value']) or not equal(decision(bundle),case['stopping_decision']):
            raise RuntimeError('Public CHECK/WAIT regression drift')
        if current['target_benefit']<=0:raise RuntimeError('Old positive public evidence omitted')
    flow=regression[0]['public_state']['tasks'][156]
    if flow['flow']['time_s']!=370 or flow['qualified']['time_s']!=130 or flow['flow']['value']>=33.6:
        raise RuntimeError('Cooling156 freshness regression')
    rows=read(pilot/'pilot-index.json');counts={};measurements=0
    for row in rows:
        raw=(pilot/row['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise RuntimeError('Pilot original bytes')
        record=json.loads(gzip.decompress(raw));methods=counts.setdefault(row['specification']['scenario'],set());methods.add(row['method'])
        if len(record['action_log'])!=359 or len(record['stopping_decisions'])!=359:raise RuntimeError('Truncated full episode')
        if row['metrics']!=record['metrics'] or row['wall_s']>30:raise RuntimeError('Cost or saved metrics mismatch')
        bytask={j:[] for j in range(176)}
        deadlines=[300.]*128+[600.]*48
        for event in record['inspection_events']:bytask[event['task']].append(event['time_s'])
        for j,times in bytask.items():
            times=[0.]+sorted(times)+[1800.]
            if max(b-a for a,b in zip(times,times[1:]))>deadlines[j]+1e-9:raise RuntimeError('Independent exact gap')
        if sum(c['task']!=c['baseline_next'] for c in record['selected_certificates'])>1:raise RuntimeError('Budget')
        measurements+=len(record['inspection_events'])
        for snap in record['public_snapshots']:validate_public(snap)
    if len(rows)!=10 or any(v!=set(protocol['main_methods']) for v in counts.values()) or len(counts)!=2:
        raise RuntimeError('Five-method paired pilot incomplete')
    if measurements!=audit['independent_route_measurements']:raise RuntimeError('Audit measurement scope')
    ledger=list(csv.DictReader((ev/'cluster-final/jobs/slurm-final.psv').open(encoding='utf-8'),delimiter='|'))
    ledger={r['JobID']:r for r in ledger}
    ids=[str(j) for j in read(ev/'job-identities.json')['all_jobs']]
    if ledger['130509']['State']!='FAILED':raise RuntimeError('Original packaging failure hidden')
    for job in ids:
        if job!='130509' and (ledger[job]['State']!='COMPLETED' or ledger[job]['ExitCode']!='0:0'):
            raise RuntimeError('Nonterminal expected run '+job)
    peak=[float(ledger[j+'.batch']['MaxRSS'].removesuffix('K'))/1024 for j in ids if ledger[j+'.batch']['MaxRSS'].endswith('K')]
    if not peak or max(peak)>2048:raise RuntimeError('Frozen peak RSS cap exceeded')
    if any('gres/gpu' in ledger[j]['AllocTRES'] for j in ids):raise RuntimeError('Unexpected GPU allocation')
    if (ev/'run-manifest.json').exists():
        for name,r in read(ev/'run-manifest.json')['files'].items():
            data=(root/name).read_bytes()
            if len(data)!=r['bytes'] or hashlib.sha256(data).hexdigest()!=r['sha256']:raise RuntimeError('Final manifest drift '+name)
    result={'baseline':protected['baseline'],'all_protected_files_unchanged':len(protected['files']),
        'actual_Torch_tests_no_skips':tests['tests'],'pilot_full_cases':len(rows),'old_causal_regressions':len(regression),
        'independent_full_measurements':measurements,'peak_RSS_MB':max(peak),
        'all_public_values_and_CHECK_WAIT_decisions_recomputed':True,'frozen_protocol_unchanged':True,
        'float_tolerance':{'relative':1e-10,'absolute':1e-11,'actions_and_counts':'exact'},
        'final_test_not_opened':True,'source_scope':'Portable audit binds actual Torch and full path/weight replays; not a local Torch rerun.'}
    with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
