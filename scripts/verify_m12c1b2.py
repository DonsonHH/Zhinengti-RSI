"""Portable B2 source, original delivery, complete paired result verification."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import tarfile

import numpy as np

FIXTURES=(('healthy',900),('coldplate_fouling',60),('coldplate_fouling',180),('coldplate_fouling',900),
    ('dc_contact_hotspot',900),('it_feeder_loss',180),('sensor_bias',900),('aux_feeder_loss+cdu_fouling',180))


def read(path):
    if not path.exists():
        path=path.with_name(path.name+'.gz')
    raw=path.read_bytes();return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(x,message):
    if not x:
        raise RuntimeError(message)


def same(a,b):
    if isinstance(a,dict):
        return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,list):
        return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,float):
        return isinstance(b,(float,int)) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8)
    return a==b


def aggregate(rows):
    weight=sum(r['event_weight_sum'] for r in rows)
    return {'cases':len(rows),'segments':sum(r['metrics']['anomaly_segments'] for r in rows),
        'missed':sum(r['metrics']['missed_segments'] for r in rows),
        'weighted_wait_s':sum(r['weighted_delay_sum'] for r in rows)/weight if weight else None,
        'travel_m':sum(r['metrics']['travel_m'] for r in rows),'service_s':sum(r['metrics']['service_s'] for r in rows),
        'extra_checks':sum(r['metrics']['selected_detours'] for r in rows),
        'false_measured_confirmations':sum(r['metrics']['false_measured_confirmations'] for r in rows),
        'exact_violations':sum(r['metrics']['tasks_violating_exact_gap'] for r in rows)}


def condition(r):
    return tuple(r['specification'][k] for k in ('scenario','duration_s','seed','fault_start_s','noise_c'))


def main():
    p=argparse.ArgumentParser()
    for key in ('workspace','research','output'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();root=args.workspace;base=args.research;ev=base/'evidence';current={}
    sources=sorted([read(p) for p in base.glob('*_manifest.json') if 'archive' in read(p)],key=lambda r:r['created_at_utc'])
    for m in sources:
        archive=base/'source'/m['archive'];need(sha(archive)==m['sha256'],'Source archive bytes')
        with tarfile.open(archive) as tar:
            members={v.name:v for v in tar.getmembers() if v.isfile()};need(set(members)==set(m['files']),'Exact archived sources')
            for rel,r in m['files'].items():
                raw=tar.extractfile(members[rel]).read()
                need(len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256'],'Archived member bytes')
                current[rel]=r
    for rel,r in current.items():
        need(sha(root/rel)==r['sha256'],'Current source drift '+rel)
    initial=read(base/'source-v1_manifest.json')
    frozen=[rel for rel in initial['files'] if rel.startswith('src/') or rel=='tests/test_robust_timing.py' or rel.endswith('/复核与实验设计.md')]
    for rel in frozen:
        need(sha(root/rel)==initial['files'][rel]['sha256'],'Policy, evaluation and design changed after freeze '+rel)
    transfer=read(ev/'transfer-manifest.json')
    for rel,r in transfer.items():
        path=ev/rel;need(path.stat().st_size==r['bytes'] and sha(path)==r['sha256'],'Original delivered bytes '+rel)
    for rel,r in read(ev/'origin-manifest.json').items():
        raw=(ev/rel).read_bytes();raw=gzip.decompress(raw) if r['lossless_gzip_wrapper_added'] else raw
        need(len(raw)==r['source_bytes'] and hashlib.sha256(raw).hexdigest()==r['source_sha256'],'Lossless wrapper exact original')
    all_rows=[];physical={};audits={};service={}
    configs=read(ev/'chunk-0/controls-frozen-before-test.json')['configurations']
    need(len(configs)==10 and list(configs)==['tour','cue_immediate','evidence_now','lookahead','lookahead_permuted',
        'lookahead_masked','lookahead_conservative','blind_0','blind_1','blind_2'],'All ten methods')
    for name in ['smoke']+[f'chunk-{i}' for i in range(8)]:
        rows=read(ev/name/'case-index.json');summary=read(ev/name/'summary.json');audit=read(ev/(name+'-audit.json'));audits[name]=audit
        need(summary['actual_cases']==len(rows)==audit['actual_cases_replayed'],'Every raw actual case replayed '+name)
        need(audit['exact_gap_violations']==0 and audit['global_optional_budget']==1,'All actual constraint audits')
        need(summary['new_neural_training'] is False and summary['LLM_calls']==0 and summary['old_actors_unchanged'],'Frozen old models, no new calls')
        if name!='smoke':
            i=int(name[-1]);phase=(150,335,520,705)[i//2];seed=8300+i%2
            coords={(a,b,seed,phase,noise) for a,b in FIXTURES for noise in (.2,1.)}
            need(len(rows)==160 and summary['chunk']==i,'Complete phase task')
        else:
            coords={(a,b,8399,335,.2) for a,b in (('healthy',900),('coldplate_fouling',180))}
            need(len(rows)==20,'Complete smoke methods')
        frozen_config=read(ev/name/'controls-frozen-before-test.json')
        need(frozen_config['configurations']==configs,'Controls identical in every task')
        need({tuple(r[k] for k in ('scenario','duration_s','seed','fault_start_s','noise_c')) for r in frozen_config['conditions']}==coords and
            len(frozen_config['conditions'])==len(coords),'Pre-execution condition matrix')
        groups=read(ev/name/'groups.json');lookup={r['case']:r for r in rows}
        for tag,g in groups.items():
            selected=[lookup[k] for k in g['case_ids']]
            need({condition(r) for r in selected}==coords and len(selected)==len(coords),'Exact common factorial conditions')
            need(all(r['deployment']['timing_config']==configs[tag] for r in selected),'Actual deployment config fixed')
            need(same(aggregate(selected),g['aggregate']),'Individual task group statistics')
        for r in rows:
            m=r['metrics'];need(m['selected_detours']<=1 and m['tasks_violating_exact_gap']==0,'Every case constraints')
            need(m['missed_segments']==sum(not v['detected'] for v in r['anomaly_intervals']),'Per-case miss count')
            need(same(r['weighted_delay_sum'],sum(v['weight']*v['delay_or_censor_s'] for v in r['anomaly_intervals'])),'Censored waiting weights')
            key=r['process_key'];before=service.setdefault(key,m['unserved_IT_kwh']);need(same(before,m['unserved_IT_kwh']),'Inspection is information, not repair')
        if name!='smoke':
            all_rows.extend(dict(r,folder=name) for r in rows)
            for key,r in read(ev/name/'process-index.json').items():
                if key in physical:
                    need(physical[key]==r['numeric_digest'],'Common physical digest')
                physical[key]=r['numeric_digest']
    report=read(ev/'report/aggregate.json')
    need(len(all_rows)==report['formal_cases']==1280 and len(physical)==report['physical_configurations']==64,'Full scope')
    need(len({r['case'] for r in all_rows})==len(all_rows),'All cross-phase case identifiers unique')
    need(report['unique_numeric_process_fingerprints']==len(set(physical.values())),'Configured vs numerically distinct processes')
    baselines={condition(r):r for r in all_rows if r['tag']=='tour'}
    need(len(baselines)==report['observation_conditions']==128,'Unique paired observation conditions')
    for tag,config in configs.items():
        rows=[r for r in all_rows if r['tag']==tag]
        need(same(aggregate(rows),report['groups'][tag]),'Pooled full outcomes '+tag)
        pairs=[]
        for r in rows:
            a,b=baselines[condition(r)]['metrics'],r['metrics']
            pairs.append({'case':r['case'],'process_key':r['process_key'],'specification':r['specification'],
                'missed_delta':b['missed_segments']-a['missed_segments'],
                'weighted_wait_delta_s':b['consequence_weighted_censored_delay_s']-a['consequence_weighted_censored_delay_s'] if r['anomaly_intervals'] else None,
                'travel_delta_m':b['travel_m']-a['travel_m'],'service_delta_s':b['service_s']-a['service_s']})
        expected={'fewer_misses':sum(r['missed_delta']<0 for r in pairs),'same_misses':sum(r['missed_delta']==0 for r in pairs),
            'more_misses':sum(r['missed_delta']>0 for r in pairs),'cases':pairs}
        need(same(expected,report['paired_to_tour'][tag]),'Every positive and negative paired outcome')
        for phase in (150,335,520,705):
            need(same(aggregate([r for r in rows if r['specification']['fault_start_s']==phase]),report['phase_groups'][tag][str(phase)]),'Every phase stratum')
        for noise in (.2,1.):
            need(same(aggregate([r for r in rows if r['specification']['noise_c']==noise]),report['noise_groups'][tag][str(noise)]),'Every noise stratum')
        for scenario,duration in FIXTURES:
            need(same(aggregate([r for r in rows if r['specification']['scenario']==scenario and r['specification']['duration_s']==duration]),
                report['scenario_groups'][tag][scenario+'--'+str(duration)]),'Every scenario and duration stratum')
        times=[r['optional_action']['time_s'] for r in rows if r['optional_action'] is not None];actions=report['actions_and_waits'][tag]
        need(actions['times_s']==times and actions['actions']==len(times) and actions['unused_budget_cases']==128-len(times),'Actual action summaries')
        need(actions['unused_after_positive_wait']==sum(r['optional_action'] is None and r['positive_now_wait_frames']>0 for r in rows) and
            actions['positive_now_wait_frames']==sum(r['positive_now_wait_frames'] for r in rows),'All stalled/positive-now wait summaries')
        if tag=='lookahead_masked':
            need(all(r['optional_action'] is None for r in rows) and report['groups'][tag]==report['groups']['tour'],'Masked information equals tour outcomes')
    blind=report['blind_all_three'];need(blind['missed']==[report['groups'][f'blind_{i}']['missed'] for i in range(3)] and
        same(blind['mean_missed'],float(np.mean(blind['missed']))),'All blind seeds included without best selection')
    compact=read(ev/'report/compact-index.json.gz');need(compact==all_rows,'Complete original compact case index')
    strata=read(ev/'strata-review.json')
    need(strata['input_sha256']==sha(ev/'report/compact-index.json.gz') and strata['all_formal_rows']==len(all_rows) and
        strata['postprocessing_only'],'Seed review bound to full original data')
    for tag in configs:
        for seed in (8300,8301):
            rows=[r for r in all_rows if r['tag']==tag and r['specification']['seed']==seed];g=aggregate(rows)
            entry=strata['seed_groups'][tag][str(seed)]
            need(entry=={'cases':g['cases'],'segments':g['segments'],'missed':g['missed'],'weighted_wait_s':g['weighted_wait_s'],'actions':g['extra_checks']},'Every underlying seed reported')
            expected_delta=g['missed']-strata['seed_groups']['tour'][str(seed)]['missed']
            need(strata['miss_difference_vs_tour_by_seed'][tag][str(seed)]==expected_delta,'Opposing seed effects preserved')
    if 'failure_timelines' in strata:
        lookup={r['case']:r for r in all_rows}
        for example in strata['failure_timelines']['cases']:
            records=[]
            for field in ('baseline_case','method_case'):
                case=example[field];path=ev/'failure-examples'/(case+'.json.gz')
                need(sha(path)==lookup[case]['sha256'],'Exact original post hoc explanatory record')
                records.append(read(path))
            before,after=records
            extra=[r for r in after['workflow_decisions'] if r['task']!=r['baseline_next']]
            need(len(extra)==1 and example['action']==extra[0],'Failure example action')
            need(example['missed_delta']==after['metrics']['missed_segments']-before['metrics']['missed_segments'],'Failure example full-episode loss')
    with (ev/'slurm-final.psv').open(encoding='utf-8') as f:
        ledger={r['JobID']:r for r in csv.DictReader(f,delimiter='|')}
    for label,jobs in read(ev/'job-identities.json').items():
        for job in jobs if isinstance(jobs,list) else [jobs]:
            need(ledger[str(job)]['State']=='COMPLETED' and ledger[str(job)]['ExitCode']=='0:0','Every task terminal success '+label)
    smoke=read(ev/'job-identities.json')['smoke'];log=(ev/f'jobs/smoke-{smoke}.err').read_text(encoding='utf-8')
    need('Ran 188 tests' in log and '\nOK\n' in log and 'skipped' not in log,'All tests including Torch, no skips')
    figures=[]
    for path in sorted((ev/'report').glob('*.png')):
        raw=path.read_bytes();need(raw[:8]==b'\x89PNG\r\n\x1a\n','PNG integrity')
        figures.append({'file':path.name,'pixels':list(struct.unpack('>II',raw[16:24])),'sha256':sha(path)})
    need(len(figures)==3,'All three report figures')
    if (ev/'run_manifest.json').exists():
        for rel,r in read(ev/'run_manifest.json')['files'].items():
            path=root/rel;need(path.stat().st_size==r['bytes'] and sha(path)==r['sha256'],'Final bound bytes '+rel)
    result={'source_archives':len(sources),'current_source_members':len(current),'pretest_source_files_unchanged':len(frozen),
        'original_transferred_files':len(transfer),'complete_tests_no_skips':188,'formal_actual_cases':len(all_rows),
        'smoke_cases':20,'physical_configurations':64,'unique_numeric_process_fingerprints':len(set(physical.values())),
        'observation_conditions':128,'all_phase_noise_scenario_pairing_and_information_controls_verified':True,
        'full_execution_audits':{k:v['actual_cases_replayed'] for k,v in audits.items()},'figures':figures,
        'scope':'Portable source/original/statistic review; full physical/public-model/action replay bound to Slurm audit evidence.'}
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
