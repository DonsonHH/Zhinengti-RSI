"""Portable B1 original, source and independent outcome/quality verification."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import tarfile

import numpy as np


def read(p):
    if not p.exists():
        p=p.with_name(p.name+'.gz')
    raw=p.read_bytes();return json.loads(gzip.decompress(raw) if p.suffix=='.gz' else raw)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


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


def main():
    p=argparse.ArgumentParser()
    for key in ('workspace','research','output'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();root=args.workspace;base=args.research;ev=base/'evidence';current={}
    manifests=sorted([read(p) for p in base.glob('*_manifest.json') if 'archive' in read(p)],key=lambda v:v['created_at_utc'])
    for m in manifests:
        archive=base/'source'/m['archive'];need(sha(archive)==m['sha256'],'Source archive bytes')
        with tarfile.open(archive) as tar:
            members={v.name:v for v in tar.getmembers() if v.isfile()};need(set(members)==set(m['files']),'Exact source members')
            for rel,row in m['files'].items():
                raw=tar.extractfile(members[rel]).read();need(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],'Archived source member')
                current[rel]=row
    for rel,row in current.items():
        need(sha(root/rel)==row['sha256'],'Current source drift '+rel)
    initial=read(base/'source-v1_manifest.json')
    frozen=[rel for rel in initial['files'] if rel.startswith('src/') or rel=='tests/test_inspection_timing.py' or rel.endswith('/实验设计.md')]
    for rel in frozen:
        need(sha(root/rel)==initial['files'][rel]['sha256'],'Policy, environment, evaluation and declared design unchanged since pre-test source '+rel)
    transfer=read(ev/'transfer-manifest.json')
    for rel,row in transfer.items():
        path=ev/rel;need(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Original delivered bytes '+rel)
    for rel,row in read(ev/'origin-manifest.json').items():
        raw=(ev/rel).read_bytes();raw=gzip.decompress(raw) if row['lossless_gzip_wrapper_added'] else raw
        need(len(raw)==row['source_bytes'] and hashlib.sha256(raw).hexdigest()==row['source_sha256'],'Lossless original JSON wrapper')
    indices={k:read(ev/k/'case-index.json') for k in ('smoke','validation','formal')}
    lookup={r['case']:r for r in indices['formal']};physical={};service={};audits={}
    for stage,rows in indices.items():
        summary=read(ev/stage/'summary.json');audit=read(ev/(stage+'-audit.json'));audits[stage]=audit
        need(summary['actual_cases']==len(rows)==audit['all_actual_cases_replayed'],'All actual cases replayed '+stage)
        need(audit['one_global_check_budget']==1 and audit['exact_revisit_violations']==0,'Whole-loop constraints')
        for key,row in read(ev/stage/'process-index.json').items():
            if key in physical:
                need(physical[key]==row['numeric_digest'],'Same physical process across controls/noise')
            physical[key]=row['numeric_digest']
        for row in rows:
            m=row['metrics'];need(m['selected_detours']<=1 and m['tasks_violating_exact_gap']==0,'Every individual case contract')
            need(m['missed_segments']==sum(not v['detected'] for v in row['anomaly_intervals']),'Anomaly miss count')
            need(same(row['weighted_delay_sum'],sum(v['weight']*v['delay_or_censor_s'] for v in row['anomaly_intervals'])),'Censored waiting sum')
            k=row['process_key'];previous=service.setdefault(k,m['unserved_IT_kwh']);need(same(previous,m['unserved_IT_kwh']),'Observation is not repair')
    selection=read(ev/'validation/selection-frozen.json');val={r['case']:r for r in indices['validation']}
    need([v['gate'] for v in selection['all_options']]==[0,60,180,300,'off'],'Fixed five gate options')
    for option in selection['all_options']:
        rows=[val[k] for k in option['case_ids']];need(len(rows)==8 and {r['specification']['seed'] for r in rows}=={8150,8151},'Validation only')
        need(same(aggregate(rows),option['validation']),'Independent gate aggregation')
    chosen=min(selection['all_options'],key=lambda v:(v['validation']['missed'],v['validation']['weighted_wait_s'] or 0.,v['validation']['travel_m']))
    need(selection['selected_gate']==chosen['gate'],'Chosen gate fixed tie order')
    gate=read(ev/'formal/test-start-gate.json')
    need(gate['selection_frozen_at_unix']<gate['started_at_unix'] and same(gate['frozen_configs'],selection['all_configs']),'All policies frozen before testing')
    groups=read(ev/'formal/groups.json')
    report_dir=ev/'report-v2' if (ev/'report-v2').is_dir() else ev/'report'
    report=read(report_dir/'aggregate.json')
    need(same(report['groups'],groups),'Unmodified reported controls')
    conditions=None
    for tag,entry in groups.items():
        rows=[lookup[k] for k in entry['case_ids']]
        need(len(rows)==32 and same(aggregate(rows),entry['aggregate']),'Whole test aggregation '+tag)
        coords={tuple(r['specification'][k] for k in ('scenario','duration_s','seed','noise_c','fault_start_s')) for r in rows}
        conditions=coords if conditions is None else conditions;need(coords==conditions,'Common paired test conditions')
        need({r['specification']['seed'] for r in rows}=={8200,8201} and {r['specification']['noise_c'] for r in rows}=={.2,1.},'Independent new seed/noise split')
        if tag!='lookahead-nominal':
            need(all(same(r['deployment']['timing_config'],selection['all_configs'][tag]) for r in rows),'Frozen test config')
    need(len(groups)==9 and sum(len(g['case_ids']) for g in groups.values())==288,'All nine deployed groups')
    def condition(row):
        return tuple(row['specification'][k] for k in ('scenario','duration_s','seed','noise_c','fault_start_s'))
    baselines={condition(r):r for r in indices['formal'] if r['tag']=='tour'}
    for tag,entry in groups.items():
        pairs=report['paired_to_tour'][tag];actual=[]
        for key in entry['case_ids']:
            row=lookup[key];a,b=baselines[condition(row)]['metrics'],row['metrics']
            actual.append({'case':key,'process_key':row['process_key'],'noise_c':row['specification']['noise_c'],
                'missed_delta':b['missed_segments']-a['missed_segments'],
                'weighted_wait_delta_s':b['consequence_weighted_censored_delay_s']-a['consequence_weighted_censored_delay_s'] if row['anomaly_intervals'] else None,
                'travel_delta_m':b['travel_m']-a['travel_m'],'service_delta_s':b['service_s']-a['service_s']})
        need(same(pairs['cases'],actual) and pairs['fewer_misses']==sum(r['missed_delta']<0 for r in actual) and
            pairs['more_misses']==sum(r['missed_delta']>0 for r in actual) and pairs['same_misses']==sum(r['missed_delta']==0 for r in actual),'Every paired improvement and failure '+tag)
        decisions=report['decisions'][tag]
        need(decisions['actions']==sum(lookup[k]['metrics']['selected_detours'] for k in entry['case_ids'])==len(decisions['times_s']),
            'Reported action timing denominator '+tag)
        need(decisions['actions_first_30s']==sum(t<=30 for t in decisions['times_s']) and sum(decisions['decision_counts'].values())==359*32,'All dispatch-frame timing records')
    atlas=read(ev/'formal/atlas-before-branches.json');branches=read(ev/'formal/branch-index.json')
    expected_branches={};test_atlas={}
    for key,value in atlas.items():
        if lookup[key]['specification']['phase']=='test':
            test_atlas[key]=value
        for i,sample in enumerate(value['samples']):
            expected_branches[key.rsplit('--',1)[0]+'--branch-'+str(i)]=(key,sample)
    need(set(expected_branches)=={r['branch_case'] for r in branches},'All and only predeclared branch selections')
    for row in branches:
        base_key,sample=expected_branches[row['branch_case']]
        need(row['base_case']==base_key and all(row[k]==v for k,v in sample.items()),'Declared branch source')
        before,after=lookup[base_key],lookup[row['branch_case']];a,b=before['metrics'],after['metrics'];v=row['labels']
        need(v['missed_delta']==b['missed_segments']-a['missed_segments'],'Whole-episode miss change')
        gain=(a['consequence_weighted_censored_delay_s']-b['consequence_weighted_censored_delay_s']) if before['anomaly_intervals'] else 0.
        need(same(v['net'],-v['missed_delta']+gain/3000-(b['travel_m']-a['travel_m'])/100000),'Net label')
        lost=sum(x['detected'] and not y['detected'] for x,y in zip(before['anomaly_intervals'],after['anomaly_intervals']))
        need(v['collateral_lost_segments']==lost,'Full collateral loss count')
    for key,a in test_atlas.items():
        samples=[lookup[r['branch_case']] for r in branches if r['base_case']==key];baseline=lookup[key]
        rank=lambda r:(r['metrics']['missed_segments'],r['metrics']['consequence_weighted_censored_delay_s'] or 0.,r['metrics']['travel_m'])
        best=min([baseline]+samples,key=rank)
        ref=next(r for r in read(report_dir/'reference-cases.json') if r['base_case']==key)
        need(ref['best_sample_case']==best['case'] and ref['sample_actions']==len(samples),'Finite sampled reference includes whole-episode effects')
    for key,a in test_atlas.items():
        for segment in a['segments']:
            saved=next(r for r in report['event_catalogue'] if r['base_case']==key and r['segment_index']==segment['segment_index'])
            need(same(saved['interval'],segment['interval']) and saved['legal_witness_instances']==segment['legal_witness_instances'] and
                saved['old_cued_witnesses']==segment['old_cued_witnesses'] and saved['new_cued_witnesses']==segment['new_cued_witnesses'],'Event catalogue current evidence')
            samples=[r for r in branches if r['base_case']==key and any(s['segment_index']==segment['segment_index'] for s in r['sources'])]
            need(saved['sample_branches']==[r['branch_case'] for r in samples] and saved['target_recovered_and_total_misses_reduced']==
                any(r['labels']['target_recovered_segments']>0 and r['labels']['missed_delta']<0 for r in samples),'Event-level target recovery plus net benefit')
    ref=report['opportunity_reference'];reference_rows=read(report_dir/'reference-cases.json')
    need(ref['baseline_missed']==sum(r['base_missed'] for r in reference_rows) and ref['best_sample_missed']==sum(r['best_missed'] for r in reference_rows),'One best sample per condition, no combined rescue count')
    need(same(ref['best_sample_weighted_wait_s'],sum(r['best_weighted_delay_sum'] for r in reference_rows)/sum(r['weight_sum'] for r in reference_rows)),'Finite reference weighted waiting')
    point=[p for a in test_atlas.values() for p in a['temperature']];y=np.array([p['truth_c'] for p in point]);fresh=np.array([p['fresh'] for p in point],dtype=bool)
    for name in ('hold','linear','nominal','history'):
        pred=np.array([p[name] for p in point]);error=pred-y
        for subset,mask in (('all',np.ones(y.shape,dtype=bool)),('hot',y>65),('normal',y<=65),('not_currently_observed_hot',(y>65)&~fresh),('not_currently_observed',~fresh)):
            entry=report['temperature_quality'][name][subset]
            expected_rmse=float(np.sqrt(np.mean(error[mask]**2))) if mask.any() else None
            need(entry['node_times']==int(mask.sum()) and same(entry['rmse_c'],expected_rmse),'Conditional temperature error '+name)
            hot,found=y[mask]>65,pred[mask]>65
            need([entry[k] for k in ('tp','fp','fn')]==[int((hot&found).sum()),int((~hot&found).sum()),int((hot&~found).sum())],'Hotspot counts '+name)
            need(same(entry['mae_c'],float(np.mean(abs(error[mask]))) if mask.any() else None),'Conditional absolute error')
            tp,fp,fn=entry['tp'],entry['fp'],entry['fn']
            need(same(entry['precision'],tp/(tp+fp) if tp+fp else None) and same(entry['recall'],tp/(tp+fn) if tp+fn else None),'Conditional precision and recall')
    for name,entry in report['signal_quality'].items():
        counts=Counter()
        for a in test_atlas.values():
            counts.update(a['confusion'][name])
        need(all(entry[k]==counts[k] for k in ('tp','fp','fn','tn')),'All candidate positive/negative cues')
        lead=[r['interval']['onset_s']-r['event_signal_first_s'][name] for r in report['event_catalogue'] if r['event_signal_first_s'][name] is not None]
        need(entry['lead_to_event_onset_s']==lead and entry['event_records']==len(report['event_catalogue']) and
            entry['event_records_with_legal_optional_trigger_in_window']==len(lead),'Event trigger window and lead denominator')
    for name,entry in report.get('event_summary',{}).items():
        events=report['event_catalogue'] if name=='all' else [r for r in report['event_catalogue'] if r['interval']['kind']==name]
        missed=[r for r in events if not r['baseline_detected']]
        need(entry['event_records']==len(events) and entry['tour_missed_records']==len(missed),'Event summary independent professional denominators')
        need(entry['tour_missed_with_legal_one_check_witness']==sum(r['legal_witness_instances']>0 for r in missed) and
            entry['tour_missed_target_recovered_and_net_misses_reduced_in_a_sample']==sum(r['target_recovered_and_total_misses_reduced'] for r in missed),'Event actionability and net improvement distinguished')
        need(entry['tour_missed_with_old_cued_witness']==sum(r['old_cued_witnesses']>0 for r in missed) and
            entry['tour_missed_with_new_cued_witness']==sum(r['new_cued_witnesses']>0 for r in missed),'Event cues restricted to legal completion witnesses')
    with (ev/'slurm-final.psv').open(encoding='utf-8') as f:
        ledger={r['JobID']:r for r in csv.DictReader(f,delimiter='|')}
    for name,job in read(ev/'job-identities.json').items():
        expected='CANCELLED' if name=='cancelled_pending_report_enrichment' else 'COMPLETED'
        need(ledger[str(job)]['State'].startswith(expected) and ledger[str(job)]['ExitCode']=='0:0','Expected terminal state '+name)
        if expected=='CANCELLED':
            need(ledger[str(job)]['ElapsedRaw']=='0','Cancelled before any report computation')
    log=(ev/'jobs/smoke-130337.err').read_text(encoding='utf-8');need('Ran 181 tests' in log and '\nOK\n' in log and 'skipped' not in log,'All 181 tests no skips')
    figures=[]
    for path in sorted(report_dir.glob('*.png')):
        raw=path.read_bytes();need(raw[:8]==b'\x89PNG\r\n\x1a\n','PNG header')
        figures.append({'file':path.name,'pixels':list(struct.unpack('>II',raw[16:24])),'sha256':sha(path)})
    need(len(figures)==3,'All three figures')
    if (ev/'report-enrichment-transfer.json').exists():
        for rel,row in read(ev/'report-enrichment-transfer.json').items():
            path=ev/'enrichment-jobs'/Path(rel).name if rel.startswith('jobs/') else ev/rel
            need(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Original enriched report bytes')
    for entry in read(ev/'posthoc-illustration/selection.json')['files']:
        row=lookup[entry['original_case']];path=ev/'posthoc-illustration'/entry['file']
        need(path.stat().st_size==row['bytes']==entry['bytes'] and sha(path)==row['sha256']==entry['sha256'],'Explanatory case is exact original output')
    if (ev/'run_manifest.json').exists():
        for rel,row in read(ev/'run_manifest.json')['files'].items():
            path=root/rel;need(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Final bound file '+rel)
    result={'source_archives':len(manifests),'current_source_members':len(current),'original_transferred_files':len(transfer),
        'complete_tests_no_skips':181,'smoke_cases':len(indices['smoke']),'validation_cases':40,
        'formal_cases':len(indices['formal']),'public_test_cases':288,'independent_branches':len(branches),
        'formal_atlas_baselines':len(atlas),'all_unique_physical_processes_including_smoke':len(physical),
        'policy_environment_evaluation_and_design_files_unchanged_from_pretest':len(frozen),
        'outcome_selection_branch_labels_temperature_and_signal_counts_recomputed':True,'figures':figures,
        'scope':'Portable byte/statistic review. Full physical/route/forecast/control execution is bound to original Slurm audits.'}
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
