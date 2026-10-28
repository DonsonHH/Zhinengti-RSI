"""Portable source/original, complete-case, label and result verification."""
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


def need(value,message):
    if not value:
        raise RuntimeError(message)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    if not p.exists():
        p=p.with_name(p.name+'.gz')
    raw=p.read_bytes()
    return json.loads(gzip.decompress(raw) if p.suffix=='.gz' else raw)


def same(a,b):
    if isinstance(a,dict):
        return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,list):
        return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,float):
        return isinstance(b,(float,int)) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8)
    return a==b


def aggregate(rows):
    weights=sum(r['event_weight_sum'] for r in rows)
    return dict(cases=len(rows),segments=sum(r['metrics']['anomaly_segments'] for r in rows),
        missed=sum(r['metrics']['missed_segments'] for r in rows),
        weighted_wait_s=sum(r['weighted_delay_sum'] for r in rows)/weights if weights else None,
        travel_m=sum(r['metrics']['travel_m'] for r in rows),service_s=sum(r['metrics']['service_s'] for r in rows),
        extra_checks=sum(r['metrics']['selected_detours'] for r in rows),
        false_measured_confirmations=sum(r['metrics']['false_measured_confirmations'] for r in rows),
        exact_violations=sum(r['metrics']['tasks_violating_exact_gap'] for r in rows))


def key_rank(v):
    return v['missed'],v['weighted_wait_s'] or 0.,v['travel_m']


def model_id(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def predict(pack,x):
    x=np.asarray(x,dtype=np.float64)
    if pack['kind']=='ridge':
        w=np.asarray(pack['weights']);return x@w[:-1]+w[-1]
    state=pack['state']
    for i in (0,2,4):
        x=x@np.asarray(state[str(i)+'.weight']).T+np.asarray(state[str(i)+'.bias'])
        if i!=4:
            x=np.tanh(x)
    return x.ravel()


def main():
    p=argparse.ArgumentParser()
    for name in ('workspace','research','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();base=args.research;ev=base/'evidence';current={};archives=0
    def source_order(path):
        label=path.name.removesuffix('_manifest.json')
        # These are the actual creation/deployment order; report-v2 is a layout repair.
        order=('source-v1','analysis-v1','report-v1','analysis-v2','analysis-v3','package-v1','report-v2','package-v2')
        need(label in order,'Declared source archive version')
        return order.index(label)
    manifests=sorted(base.glob('*_manifest.json'),key=source_order)
    for path in manifests:
        manifest=read(path)
        if 'archive' not in manifest or 'files' not in manifest:
            continue
        archive=base/'source'/manifest['archive'];need(sha(archive)==manifest['sha256'],'Source archive hash')
        with tarfile.open(archive) as tar:
            members={m.name:m for m in tar.getmembers() if m.isfile()}
            need(set(members)==set(manifest['files']),'Source archive exact member set')
            for rel,entry in manifest['files'].items():
                raw=tar.extractfile(members[rel]).read()
                need(len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],'Archived source bytes')
                current[rel]=entry
        archives+=1
    for rel,entry in current.items():
        raw=(args.workspace/rel).read_bytes().replace(b'\r\n',b'\n')
        need(hashlib.sha256(raw).hexdigest()==entry['sha256'],'Current source drift '+rel)
    transfer=read(ev/'transfer-manifest.json')
    for rel,entry in transfer.items():
        path=ev/rel;need(path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],'Original delivery '+rel)
    for rel,entry in read(ev/'origin-manifest.json').items():
        raw=(ev/rel).read_bytes()
        if entry['lossless_gzip_wrapper_added']:
            raw=gzip.decompress(raw)
        need(len(raw)==entry['source_bytes'] and hashlib.sha256(raw).hexdigest()==entry['source_sha256'],'Original bytes through lossless wrapper')
    folders=['data']+[f'fit-{i}' for i in range(3)]+['test-common']+[f'test-{i}' for i in range(3)]+['heldout']
    indices={folder:read(ev/folder/'case-index.json') for folder in folders}
    lookups={folder:{r['case']:r for r in rows} for folder,rows in indices.items()}
    physical={};service={};audits={};all_cases=0
    for folder,rows in indices.items():
        summary=read(ev/folder/'summary.json');audit=read(ev/(folder+'-audit.json'));audits[folder]=audit
        need(summary['actual_cases']==len(rows)==audit['all_actual_cases_replayed'],'Complete actual replay '+folder)
        need(len(lookups[folder])==len(rows),'Unique local case IDs')
        need(audit['exact_violations']==0 and audit['global_optional_budget']==1,'Actual one-check constraint audit')
        need(read(ev/folder/'completed.json')['old_actors_unchanged'] and summary['LLM_calls']==0,'Old models fixed and no LLM calls')
        for key,entry in read(ev/folder/'process-index.json').items():
            if key in physical:
                need(physical[key]==entry['numeric_digest'],'Same physics across models/noise')
            physical[key]=entry['numeric_digest']
        for row in rows:
            metrics=row['metrics'];need(metrics['tasks_violating_exact_gap']==0 and metrics['selected_detours']<=1,'Every case contract')
            need(metrics['missed_segments']==sum(not r['detected'] for r in row['anomaly_intervals']),'Per-case anomaly counts')
            need(same(row['event_weight_sum'],sum(r['weight'] for r in row['anomaly_intervals'])) and
                same(row['weighted_delay_sum'],sum(r['weight']*r['delay_or_censor_s'] for r in row['anomaly_intervals'])),'Censored wait weights')
            key=row['process_key'];previous=service.setdefault(key,metrics['unserved_IT_kwh'])
            need(same(previous,metrics['unserved_IT_kwh']),'Inspection is not physical repair')
        all_cases+=len(rows)
    need(len(physical)==40,'Sixteen train, eight validation, sixteen test physical processes')
    train=read(ev/'data/dataset.json');need(len(train)==read(ev/'data/summary.json')['labeled_actions'],'All training labels')
    for sample in train:
        before=lookups['data'][sample['base_case']];after=lookups['data'][sample['branch_case']];label=sample['labels']
        missed=after['metrics']['missed_segments']-before['metrics']['missed_segments']
        gain=(before['metrics']['consequence_weighted_censored_delay_s']-after['metrics']['consequence_weighted_censored_delay_s']) if before['anomaly_intervals'] else 0.
        travel=after['metrics']['travel_m']-before['metrics']['travel_m']
        need(label['missed_delta']==missed and same(label['net'],-missed+gain/3000-travel/100000),'Independent net action target')
        lost=sum(a['detected'] and not b['detected'] for a,b in zip(before['anomaly_intervals'],after['anomaly_intervals']))
        need(label['collateral_lost_segments']==lost,'Collateral losses retained')
        need(len(sample['x'])==34 and sample['provenance'],'Public 34-feature vector and sampling source')
    selections=[]
    for trial in range(3):
        folder='fit-'+str(trial);selection=read(ev/folder/'selections-frozen.json');selections.append(selection)
        need(read(ev/folder/'training.json')['updates']==400,'Fixed training updates')
        for name,entry in selection['models'].items():
            pack=read(ev/folder/'models'/(name+'.json'));need(model_id(pack)==entry['model_id'],'Selected frozen model identity')
            options=entry['all_candidates'];need([r['threshold'] for r in options]==[0.,.1,.3,'off'],'Common fixed threshold budget')
            for option in options:
                rows=[lookups[folder][k] for k in option['case_ids']]
                need(len(rows)==8 and {r['specification']['seed'] for r in rows}=={7850,7851},'Independent validation split')
                need(same(option['validation'],aggregate(rows)),'Every threshold validation summary')
            winner=min(options,key=lambda r:key_rank(r['validation']))
            need(entry['threshold']==winner['threshold'] and same(entry['validation'],winner['validation']),'Threshold selection from validation only')
    group_tables={}
    for folder in ['test-common']+[f'test-{i}' for i in range(3)]:
        gate=read(ev/folder/'test-start-gate.json')
        need(len(gate['all_three_trials_frozen'])==3 and all(r['frozen_at_unix']<gate['test_start_unix'] for r in gate['all_three_trials_frozen']),'All selections frozen before any test')
        groups=read(ev/folder/'test-groups.json');group_tables[folder]=groups
        for name,entry in groups.items():
            rows=[lookups[folder][k] for k in entry['case_ids']]
            need(len(rows)==32 and {r['specification']['seed'] for r in rows}=={7900,7901} and
                {r['specification']['noise_c'] for r in rows}=={.2,1.},'Declared paired test conditions')
            need(same(entry['aggregate'],aggregate(rows)),'Independent test group aggregate')
            for row in rows:
                dep=row['deployment']
                if dep['model_id'] is not None:
                    trial=0 if folder=='test-common' else int(folder[-1]);selection=selections[trial]['models'][dep['public_controller']]
                    need(dep['model_id']==selection['model_id'] and dep['threshold']==selection['threshold'],'Frozen selected deployment')
    report=read(ev/'report/aggregate.json')
    baseline=group_tables['test-common']['tour']['aggregate']
    for name,entry in report['outcomes'].items():
        values=[group_tables['test-common'][name]['aggregate']] if name in group_tables['test-common'] else [group_tables[f'test-{i}'][name]['aggregate'] for i in range(3)]
        need(same(entry['by_initialization'],values) and same(entry['mean_missed'],float(np.mean([v['missed'] for v in values]))),'Outcome aggregation and initialization denominator')
        need(same(entry['mean_weighted_wait_s'],float(np.mean([v['weighted_wait_s'] for v in values]))) and
            entry['missed_range']==[min(v['missed'] for v in values),max(v['missed'] for v in values)] and
            same(entry['wait_range_s'],[min(v['weighted_wait_s'] for v in values),max(v['weighted_wait_s'] for v in values)]),'Outcome wait means and ranges')
        need(same(entry['mean_extra_travel_m_per_case'],float(np.mean([(v['travel_m']-baseline['travel_m'])/32 for v in values]))) and
            same(entry['mean_optional_actions'],float(np.mean([v['extra_checks'] for v in values]))),'Actual action and travel means')
    need(report['total_actual_formal_rollouts']==all_cases and report['independent_physical_processes_across_splits']==len(physical),'Report execution scope')
    held=read(ev/'heldout/heldout-labels.json');pred=read(ev/'heldout/predictions-frozen-before-labels.json')['samples']
    need(len(held)==len(pred)==len(indices['heldout']),'Complete heldout predictions and labels')
    for a,b in zip(pred,held):
        need(all(same(a[k],b[k]) for k in a),'Predictions saved before label fields')
        for name,value in a['predictions'].items():
            trial,method=name.split('/');pack=read(ev/('fit-'+trial)/'models'/(method+'.json'))
            need(math.isclose(value,float(predict(pack,[a['x']])[0]),rel_tol=3e-5,abs_tol=2e-6),'Independent NumPy prediction from frozen tensors')
    for key,entry in report['heldout_value_regression'].items():
        objective='target' if key.split('/')[1]=='target' else 'net'
        error=np.asarray([r['predictions'][key]-r['labels'][objective] for r in held])
        need(entry['objective']==objective and entry['samples']==len(held) and
            same(entry['mae'],float(np.mean(abs(error)))) and same(entry['mse'],float(np.mean(error**2))),'Heldout regression against the declared objective')
    for name,rows in (('all_training',train),('natural_training',[r for r in train if any(s.startswith('natural_') for s in r['provenance'])]),
        ('teacher_training',[r for r in train if any(s.startswith('teacher_') for s in r['provenance'])]),('heldout_natural',held)):
        entry=report['counterfactual_samples'][name];deltas=Counter();lost=Counter();net=[]
        for row in rows:
            delta=row['labels']['missed_delta'];deltas['fewer_misses' if delta<0 else 'more_misses' if delta>0 else 'same_misses']+=1
            lost[str(row['labels']['collateral_lost_segments'])]+=1;net.append(row['labels']['net'])
        need(entry['samples']==len(rows) and entry['miss_change_counts']==dict(deltas) and
            entry['collateral_lost_segments_histogram']==dict(lost) and same(entry['net_value_mean'],float(np.mean(net))) and
            same(entry['net_value_min'],min(net)) and same(entry['net_value_max'],max(net)) and
            entry['target_recoveries']==sum(r['labels']['target_recovered_segments'] for r in rows),'Counterfactual sample tradeoff aggregates '+name)
    quality_rows=read(ev/'test-common/test-tour-samples-and-quality.json');confusion=Counter()
    for row in quality_rows:
        confusion.update(row['confusion'])
        need(len(row['samples'])<=4 and all(all(s.startswith('natural_') for s in x['provenance']) for x in row['samples']),'Heldout public sampling excludes teachers')
    public_samples={(r['case'],i):s for r in quality_rows for i,s in enumerate(r['samples'])}
    need(len(public_samples)==len(pred),'Every heldout sample from the frozen public tour catalogue')
    for row in pred:
        source=public_samples[(row['baseline'],row['sample_number'])]
        need(all(same(row[k],source[k]) for k in source),'Heldout sample source without added teacher actions')
    cue=report['test_quality']['public_cue']
    need(all(cue[k]==confusion[k] for k in ('tp','fp','fn','tn')) and cue['candidate_instances']==sum(confusion.values()),'All optional-candidate cue confusion counts')
    details=read(ev/'quality-details-audit.json')
    need(details['fixed_test_tour_replays']==32 and not details['new_policy_rollouts_or_search'],'Fixed-tour duplicate/lead-time replay scope')
    counts=Counter()
    for row in details['cases']:
        counts.update(row['counts'])
    need(dict(counts)==details['counts'] and counts['all_candidates']==cue['candidate_instances'],'Complete duplicate-cue counts')
    need(counts['public_filtered_valid_completion']==cue['tp'] and counts['public_filtered_no_valid_completion']==cue['fp'],'New vs repeated valid measurement agrees with original cue confusion')
    need(sum(counts['new_opportunity_'+k] for k in ('tp','fp','fn','tn'))==counts['all_candidates'],'New confirmation opportunity confusion denominator')
    deployment=read(ev/'deployment-diagnosis-audit.json')
    need(deployment['all_saved_test_cases_checked']==len(deployment['cases'])==512 and deployment['postprocessing_only_no_new_search_or_policy_rollout'],'All deployment records, no extra search')
    baselines={tuple(r['specification'][k] for k in ('scenario','duration_s','seed','noise_c','fault_start_s')):r
        for r in indices['test-common'] if r['tag']=='tour'}
    for row in deployment['cases']:
        actual=lookups[row['folder']][row['case']];spec=actual['specification']
        base_row=baselines[tuple(spec[k] for k in ('scenario','duration_s','seed','noise_c','fault_start_s'))]
        a,b=base_row['metrics'],actual['metrics'];miss=b['missed_segments']-a['missed_segments']
        wait=(a['consequence_weighted_censored_delay_s']-b['consequence_weighted_censored_delay_s']) if base_row['anomaly_intervals'] else 0.
        travel=b['travel_m']-a['travel_m']
        need(row['missed_delta']==miss and same(row['realized_net_value'],-miss+wait/3000-travel/100000),'Realized selected action value from all episode outcomes')
        need((row['intervention'] is not None)==bool(b['selected_detours']),'Optional action timing present exactly when used')
    for method in ('hold','linear','nominal','history'):
        y=np.array([p['truth_c'] for r in quality_rows for p in r['temperature']]);estimate=np.array([p[method] for r in quality_rows for p in r['temperature']])
        direct=np.array([p['fresh'] for r in quality_rows for p in r['temperature']],dtype=bool)
        for name,mask in (('all',np.ones(y.shape,dtype=bool)),('not_currently_observed',~direct)):
            error=estimate[mask]-y[mask];a,b=y[mask]>65,estimate[mask]>65;entry=report['test_quality']['temperature'][method][name]
            need(same(entry['rmse_c'],float(np.sqrt(np.mean(error**2)))) and entry['node_time_samples']==int(mask.sum()),'Current estimation error '+method)
            need([entry[k] for k in ('tp','fp','fn','tn')]==[int((a&b).sum()),int((~a&b).sum()),int((a&~b).sum()),int((~a&~b).sum())],'Hotspot confusion '+method)
    with (ev/'slurm-final.psv').open() as f:
        ledger={r['JobID']:r for r in csv.DictReader(f,delimiter='|')}
    jobs=read(ev/'job-identities.json')
    for name,entries in jobs.items():
        for job in entries if isinstance(entries,list) else [entries]:
            need(ledger[str(job)]['State']=='COMPLETED' and ledger[str(job)]['ExitCode']=='0:0','Own job terminal success '+name)
    log=(ev/'jobs/smoke-130306.err').read_text()
    need('Ran 174 tests' in log and '\nOK\n' in log and 'skipped' not in log,'All 174 tests ran with torch')
    figures=[]
    display=ev/'report-v2' if (ev/'report-v2').is_dir() else ev/'report'
    if display.name=='report-v2':
        need(same(read(display/'aggregate.json'),report),'Layout revision has unchanged results')
        for rel,entry in read(ev/'figure-transfer-v2.json').items():
            path=ev/rel
            need(path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],'Revised original figure bytes')
    for path in sorted(display.glob('*.png')):
        raw=path.read_bytes();need(raw[:8]==b'\x89PNG\r\n\x1a\n','PNG integrity')
        figures.append({'file':path.name,'pixels':list(struct.unpack('>II',raw[16:24])),'sha256':sha(path)})
    need(len(figures)==3,'Three report figures')
    manifest=ev/'run_manifest.json'
    if manifest.exists():
        for rel,entry in read(manifest)['files'].items():
            path=args.workspace/rel
            need(path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],'Final bound file bytes '+rel)
    result={'source_archives_verified':archives,'current_source_members_verified':len(current),'original_transferred_files':len(transfer),
        'complete_tests_no_skips':174,'formal_actual_rollouts':all_cases,'physical_processes':len(physical),
        'counterfactual_training_samples':len(train),'heldout_action_samples':len(held),'frozen_MLP_training_runs':6,
        'frozen_training_initializations':3,'all_case_statistics_labels_selection_quality_and_frozen_predictions_checked':True,
        'audited_full_public_candidate_instances':sum(v['independent_public_candidates'] for v in audits.values()),'figures':figures,
        'scope':'Portable original/source/statistic verification; full actual candidate, physical and model-update replay is bound to the independent Slurm audits.'}
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
