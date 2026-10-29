"""Portable complete split/selection/statistics/provenance verification."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import tarfile

from patrol_rsi.formal_study import conditions,load_protocol,method_definitions
from explain_formal_study import explain_examples,resource_ledger


def bytes_original(path):
    if path.exists():return path.read_bytes()
    return gzip.decompress(path.with_name(path.name+'.gz').read_bytes())


def read(path):
    data=bytes_original(path)
    return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)


def same(a,b):
    if isinstance(a,dict):return set(a)==set(b) and all(same(v,b[k]) for k,v in a.items())
    if isinstance(a,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8)
    return a==b


def aggregate(rows):
    weight=sum(r['event_weight_sum'] for r in rows)
    return {'cases':len(rows),'segments':sum(r['metrics']['anomaly_segments'] for r in rows),
        'missed':sum(r['metrics']['missed_segments'] for r in rows),
        'weighted_delay_sum':sum(r['weighted_delay_sum'] for r in rows),'event_weight_sum':weight,
        'weighted_wait_s':sum(r['weighted_delay_sum'] for r in rows)/weight if weight else None,
        'travel_m':sum(r['metrics']['travel_m'] for r in rows),'service_s':sum(r['metrics']['service_s'] for r in rows),
        'checks':sum(r['metrics']['selected_detours'] for r in rows),'violations':sum(r['metrics']['tasks_violating_exact_gap'] for r in rows),
        'wall_s':sum(r['wall_s'] for r in rows)}


def need(value,message):
    if not value:raise RuntimeError(message)


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.workspace.resolve();base=root/'research/2026-10-29-m12c1b32-formal';ev=base/'evidence'
    for name,row in read(ev/'protected-baseline.json')['files'].items():
        data=(root/name).read_bytes();need(len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],'Old tracked bytes drift '+name)
    protocol=load_protocol(root/'research/2026-10-29-m12c1b31-interface/正式实验冻结方案.json')
    latest={}
    manifests=sorted([read(v) for v in base.glob('*-manifest.json') if 'created_utc' in read(v)],key=lambda v:v['created_utc'])
    for manifest in manifests:
        archive=base/'source'/manifest['archive']
        need(hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['sha256'],'Source archive bytes')
        with tarfile.open(archive) as tar:
            members={v.name:v for v in tar.getmembers() if v.isfile()};need(set(members)==set(manifest['files']),'Exact archive membership')
            for name,row in manifest['files'].items():
                raw=tar.extractfile(members[name]).read();need(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],'Source member')
                latest[name]=row
    for name,row in latest.items():need(hashlib.sha256((root/name).read_bytes()).hexdigest()==row['sha256'],'Current source differs '+name)
    for name,row in read(ev/'transfer-manifest.json').items():
        data=(ev/name).read_bytes();need(len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],'Transferred original bytes '+name)
    for name,row in read(ev/'origin-manifest.json').items():
        data=(ev/name).read_bytes();data=gzip.decompress(data) if row['lossless_gzip_wrapper_added'] else data
        need(len(data)==row['source_bytes'] and hashlib.sha256(data).hexdigest()==row['source_sha256'],'Original lossless wrapper')
    gate=read(ev/'code-reviewed-gate/gate/full-tests.json')
    need(gate['success'] and gate['tests']>=221 and not(gate['skipped'] or gate['failures'] or gate['errors']),'Fresh full Torch gate')
    repair_gate=read(ev/'audit-repair-gate/gate/full-tests.json')
    need(repair_gate['success'] and repair_gate['tests']>=222 and not(repair_gate['skipped'] or repair_gate['failures'] or repair_gate['errors']),'Actual Torch reapproval after audit format repair')
    selections={k:read(ev/(k+'-selection.json')) for k in ('development','validation')}
    result=read(ev/'report/aggregate.json');all_rows=[];phase_rows={};audits={};physical={}
    for phase,size in (('development',2),('validation',2),('test',4)):
        previous=None if phase=='development' else selections['development' if phase=='validation' else 'validation']
        methods=method_definitions(protocol,phase,previous);part=[];sources=[]
        for i in range(size):
            name=f'{phase}-{i}';folder=ev/name;frozen=read(folder/'phase-frozen.json');rows=read(folder/'case-index.json')
            audit=read(ev/(name+'-audit.json'));audits[name]=audit
            expected=[(k,s) for k in methods for s in conditions(protocol,phase,i)]
            need([(r['alias'],r['specification']) for r in rows]==expected,'Full frozen split coverage '+name)
            need(frozen['methods']==methods and frozen['conditions']==conditions(protocol,phase,i),'Frozen group definitions')
            need(audit['actual_cases_replayed']==len(rows) and audit['exact_violations']==0 and audit['global_optional_budget']==1,'Complete execution replay '+name)
            recovered=audit.get('recovered_prior_weight_replays',0);fresh=audit.get('new_weight_replays_this_attempt',len(rows))
            need(recovered+fresh==len(rows),'Exactly one fresh replay per formal case across attempts')
            checkpoints=[json.loads(v) for v in (ev/(name+'-audit.progress.jsonl')).read_text().splitlines()]
            need(len(checkpoints)==len(rows) and [v['case'] for v in checkpoints]==[v['case'] for v in rows],'Complete per-case audit checkpoints')
            need(sum(v['new_weight_replay'] for v in checkpoints)==fresh,'Replay accounting matches checkpoints')
            if recovered:
                need(phase=='development' and recovered==21 and fresh==107,'Only documented development JSON recovery')
                binding=ev/(name+'-historical-binding.json')
                need(hashlib.sha256(binding.read_bytes()).hexdigest()==audit['recovery']['historical_binding_sha256'],'Recovery binds historical source/log/filesystem evidence')
                history=read(binding);start=int(history['job']['Start'])
                need(history['job']['State']=='FAILED' and history['job']['ExitCode']=='1:0' and
                    all(v['ctime_ns']<=start*10**9 and v['mtime_ns']<=start*10**9 for v in history['files']),'Recovered inputs predate the failed audit start')
            digest=hashlib.sha256(bytes_original(folder/'case-index.json')).hexdigest()
            need(audit['case_index_sha256']==digest,'Execution audit bound to complete original index')
            if previous:
                predecessor=ev/('development-selection.json' if phase=='validation' else 'validation-selection.json')
                need(frozen['selection_sha256']==hashlib.sha256(predecessor.read_bytes()).hexdigest() and frozen['predecessor_selection']==previous,'Shared selection freeze chain')
                need(frozen['started_unix']>=previous['frozen_unix'],'No split opened before predecessor freeze')
            for r in rows:
                need(r['definition']==methods[r['alias']],'All aliases execute frozen method')
                need(r['metrics']['tasks_violating_exact_gap']==0 and r['metrics']['selected_detours']<=1,'Hard constraints')
                windows=r['anomaly_intervals'];need(sum(not w['detected'] for w in windows)==r['metrics']['missed_segments'],'Independent missed sum')
                need(same(sum(w['weight'] for w in windows),r['event_weight_sum']) and same(sum(w['weight']*w['delay_or_censor_s'] for w in windows),r['weighted_delay_sum']),'Independent delay and weight sums')
            for key,r in read(folder/'process-index.json').items():physical[key]=r['numeric_digest']
            sources.append({'chunk':i,'seed':protocol['physical_seeds'][phase][i],'case_index_sha256':digest,
                'audit_sha256':hashlib.sha256((ev/(name+'-audit.json')).read_bytes()).hexdigest(),'cases':len(rows)})
            part.extend(rows);all_rows.extend(dict(r,folder=name) for r in rows)
        need(len(part)==protocol['formal_case_upper_bounds'][phase],'Exact formal count '+phase)
        phase_rows[phase]=part;report=result['phases'][phase]
        coord=lambda r:tuple(r['specification'][k] for k in ('seed','scenario','duration_s','fault_start_s','noise_c'))
        baselines={coord(r):r for r in part if r['alias']=='tour'}
        for alias in methods:
            own=[r for r in part if r['alias']==alias];need(same(aggregate(own),report['groups'][alias]),'Every aggregate recomputed')
            for seed in protocol['physical_seeds'][phase]:
                need(same(aggregate([r for r in own if r['specification']['seed']==seed]),report['by_seed'][alias][str(seed)]),'Every physical seed stratum')
            for start in (335,705):need(same(aggregate([r for r in own if r['specification']['fault_start_s']==start]),report['by_start'][alias][str(start)]),'Every start stratum')
            for noise in (.2,1.):need(same(aggregate([r for r in own if r['specification']['noise_c']==noise]),report['by_noise'][alias][str(noise)]),'Every noise stratum')
            for key,summary in report['by_fixture'][alias].items():
                scenario,duration=key.rsplit('--',1)
                need(same(aggregate([r for r in own if r['specification']['scenario']==scenario and r['specification']['duration_s']==int(duration)]),summary),'Every fault-family stratum')
            pair=report['paired_to_tour'][alias];need(len(pair['cases'])==len(own),'All paired effects present')
            harmful=helpful=lost_checks=0
            for row,effect in zip(own,pair['cases']):
                before=baselines[coord(row)];windows=row['anomaly_intervals'];old=before['anomaly_intervals']
                miss=row['metrics']['missed_segments']-before['metrics']['missed_segments']
                delay=(row['metrics']['consequence_weighted_censored_delay_s']-before['metrics']['consequence_weighted_censored_delay_s']) if windows else None
                lost=[w['task_id'] for x,w in zip(old,windows) if x['detected'] and not w['detected']]
                gained=[w['task_id'] for x,w in zip(old,windows) if not x['detected'] and w['detected']]
                outcome='harmful' if miss>0 or miss==0 and delay is not None and delay>1e-9 else 'helpful' if miss<0 or miss==0 and delay is not None and delay < -1e-9 else 'unchanged'
                check=row['metrics']['selected_detours']==1
                need(effect['case']==row['case'] and effect['missed_delta']==miss and same(effect['weighted_wait_delta_s'],delay),'Every paired primary difference')
                need(effect['lost_confirmations']==lost and effect['gained_confirmations']==gained and effect['outcome']==outcome,'Every collateral outcome')
                harmful+=int(check and outcome=='harmful');helpful+=int(check and outcome=='helpful');lost_checks+=int(check and bool(lost))
            need(pair['harmful_checks']==harmful and pair['helpful_checks']==helpful and pair['check_lost_any_other_confirmation']==lost_checks,'All helpful/harmful check counts')
            source_counts=Counter();cache=Counter()
            for row in own:
                source_counts.update(row['source_attribution']['target_source_occurrences']);cache.update(row['source_attribution']['history_cache_types'])
            need(report['sources'][alias]['target_source_occurrences']==dict(source_counts) and report['sources'][alias]['history_cache_types']==dict(cache),'Every input-source attribution count')
            need(report['waits'][alias]['cases_with_positive_now_but_never_checked']==sum(r['metrics']['selected_detours']==0 and r['waiting_positive_current_frames']>0 for r in own),'All unspent-positive-wait cases')
        if phase in selections:need(selections[phase]['audited_sources']==sources,'Selection complete audited sources')
    dev=selections['development'];val=selections['validation']
    groups={k:aggregate([r for r in phase_rows['development'] if r['alias']==k]) for k in ('C0','C1','C2','C3')}
    order=sorted(groups,key=lambda k:(groups[k]['missed'],groups[k]['weighted_wait_s'],groups[k]['travel_m'],k))
    need(dev['ranked']==order and dev['retained']==order[:2],'Exact registered development ranking')
    need(val['previous_sha256']==hashlib.sha256((ev/'development-selection.json').read_bytes()).hexdigest(),'Final choice predecessor SHA')
    eligible=[];p=phase_rows['validation'];tour=[r for r in p if r['alias']=='tour'];baseg=aggregate(tour)
    for k in dev['retained']:
        own=[r for r in p if r['alias']==k];g=aggregate(own)
        miss_deltas=[sum(r['metrics']['missed_segments'] for r in own if r['specification']['seed']==s)-sum(r['metrics']['missed_segments'] for r in tour if r['specification']['seed']==s) for s in (8600,8601)]
        delay=g['weighted_delay_sum']-baseg['weighted_delay_sum']
        ok=max(miss_deltas)<=0 and delay<=1e-10 and (g['missed']<baseg['missed'] or delay < -1e-10)
        need(val['rationale'][k]['eligible']==ok,'Each validation eligibility explanation')
        if ok:eligible.append((g['missed'],g['weighted_delay_sum'],g['travel_m'],k))
    selected=min(eligible)[-1] if eligible else 'TOUR'
    need(val['selected']==selected and val['fallback']==(selected=='TOUR'),'Frozen validation fallback/choice')
    need(read(ev/'report/complete-case-index.json.gz')==all_rows,'Complete original formal compact index')
    need(len(all_rows)==768 and result['actual_formal_cases']==768,'All original formal cases')
    need(result['physical_configurations']==len(physical) and result['distinct_numeric_fingerprints']==len(set(physical.values())),'Configured and numeric process counts separated')
    resources=resource_ledger(ev/'slurm-final.psv',protocol)
    explanations=read(ev/'offline-explanations.json')
    need(same(resources,explanations['resources']),'Independent final Slurm accounting')
    need(same(explain_examples(base),explanations['illustrative_forecast_checks']),'Illustrative future-signal reference checks')
    if (ev/'run-manifest.json').exists():
        for name,r in read(ev/'run-manifest.json')['files'].items():
            data=(root/name).read_bytes();need(len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256'],'Final byte binding '+name)
    receipt={'baseline':read(ev/'protected-baseline.json')['commit'],'protected_files':len(read(ev/'protected-baseline.json')['files']),
        'source_archives':len(manifests),'original_transferred_files':len(read(ev/'transfer-manifest.json')),
        'actual_Torch_tests':repair_gate['tests'],'development_cases':256,'validation_cases':192,'final_test_cases':320,
        'full_replayed_cases':sum(v['actual_cases_replayed'] for v in audits.values()),'selected_candidates':dev['retained'],
        'final_selection':val['selected'],'fallback':val['fallback'],'all_seed_fault_start_noise_strata_verified':True,
        'frozen_selection_before_each_split_verified':True,'scope':'Portable original/source/selection/statistics verification; binds full remote weight/route/physical-array replay.'}
    receipt.update({'allocated_CPU_minutes':resources['allocated_CPU_minutes'],
        'all_resource_caps_verified':resources['within_all_frozen_resource_caps'],
        'archived_example_reference_signals_verified':True})
    with args.output.open('x',encoding='utf-8',newline='\n') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
