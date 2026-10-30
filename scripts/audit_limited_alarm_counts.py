"""Audit saved alarm/FPR accounting and describe fixed probability compression."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from patrol_rsi.limited_prediction import CELLS,HeadData,load_reviewed,need,read,sha,write


def same(a,b,message):
    need((a is None and b is None) or (a is not None and b is not None and bool(np.isclose(a,b,rtol=1e-9,atol=1e-12))),message)


def main():
    parser=argparse.ArgumentParser()
    for name in ('protocol','review','cache','selection','final-root','diagnostics','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();load_reviewed(args.protocol,args.review);selection=read(args.selection)
    diagnostics=read(args.diagnostics/'prediction-diagnostics.json');compression=[];models_checked=0
    candidates=[];all_alarms={'full':[],'time':[]};condition_ids=set()
    for ci in range(7):
        data=HeadData(args.cache/'final_test'/f'cell-{ci}',expected_phase='final_test',expected_cell=ci,expected_protocol=sha(args.protocol))
        meta=np.asarray(data.meta[:data.n]);ids=np.unique(meta[:,2]);condition_ids.update(ids.tolist())
        keys=meta[:,[2,1,7]];candidates.append(keys)
        release=read(args.final_root/f'cell-{ci}'/'released-metrics.json')
        groups={v['label']:v for v in selection['selected_heads'] if v['cell']==ci}
        for row in release['models']:
            path=args.final_root/f'cell-{ci}'/(row['id']+'--private-probabilities.npy')
            need(sha(path)==row['raw_predictions_SHA256'],'Same saved probability bytes')
            p=np.load(path,mmap_mode='r');alarm=p>row['threshold'];li=int(row['label']=='B_action');y=data.y[:data.n,li]
            for seed in row['seeds']:
                mask=meta[:,3]==seed['seed'];negative=mask&(y==0);n=int(negative.sum())
                need(seed['negative_samples']==n,'Independent per-seed negative count for GO guard')
                same(seed['FPR'],float(alarm[negative].mean()) if n else None,'Independent per-seed FPR for GO guard')
            sums=[]
            for cid in ids:
                mask=meta[:,2]==cid;own=meta[mask];a=alarm[mask]
                valid_event=(data.y[:data.n,1][mask]==1)&(own[:,5]>0)
                detected=np.unique(own[a&valid_event,5]);events=int((a&valid_event).sum())
                sums.append((int(a.sum()),int(a.sum())-len(np.unique(own[a,1])),events-len(detected)))
            means=np.mean(sums,axis=0)
            for j,key in enumerate(('alarms_per1800s','repeat_task_alarms_per1800s','repeat_event_alarms_per1800s')):
                same(row['summary'][key],float(means[j]),'Independent alarm/task/event repeat mean')
            need(row['summary']['alarms']==int(alarm.sum()),'Independent total alarms')
            chosen=next(v for v in groups[row['label']]['views'] if v['id']==row['id'])
            need(sha(Path(chosen['model_path']))==chosen['model_sha256'],'Frozen model/calibrator bytes')
            model=read(chosen['model_path']);cal=model['calibrator']
            compression.append({'cell':ci,'kind':CELLS[ci][0],'horizon_s':CELLS[ci][1],'label':row['label'],'id':row['id'],
                'family':row['family'],'calibration_a':cal['a'] if cal else None,'calibration_b':cal['b'] if cal else None,
                'minimum_probability':float(p.min()),'maximum_probability':float(p.max()),'probability_std':float(np.std(p)),
                'probability_quantiles_0_25_50_75_100':np.quantile(p,[0,.25,.5,.75,1]).tolist(),
                'limit':'Descriptive fixed-probability range,not a causal calibration ablation and not used to revise models.'})
            if row['label']=='B_action':
                for alias,key in (('full','primary_full'),('time','time_comparator')):
                    if groups['B_action'][key]==row['id']:all_alarms[alias].append(keys[alarm])
            models_checked+=1
    joined=np.concatenate(candidates);unique=np.unique(joined,axis=0)
    for alias,arrays in all_alarms.items():
        alarms=np.concatenate(arrays);dedup=np.unique(alarms,axis=0);rows=[]
        for cid in sorted(condition_ids):
            n=int((joined[:,0]==cid).sum());nu=int((unique[:,0]==cid).sum())
            a=int((alarms[:,0]==cid).sum());au=int((dedup[:,0]==cid).sum())
            rows.append({'candidate_predictions':n,'unique_completable_candidates':nu,'alarms':a,
                'unique_alarm_completions':au,'cross_horizon_repeat_alarms':a-au})
        report=next(v for v in diagnostics['joint_alarm_diagnostics'] if v['view']==alias)
        need(report['deduplicated_conditions']==len(rows),'Independent joint alarm condition denominator')
        for key in rows[0]:same(report['mean_per1800s'][key],np.mean([v[key] for v in rows]),'Independent cross-horizon candidate/alarm dedup')
    args.output.mkdir(exist_ok=False)
    write(args.output/'alarm-probability-audit.json',{'status':'passed','protocol_SHA256':sha(args.protocol),
        'selection_SHA256':sha(args.selection),'frozen_models_checked':models_checked,
        'per_seed_FPR_and_negative_counts_independently_verified':True,'alarm_task_event_cross_horizon_counts_verified':True,
        'probability_compression':compression,'model_calls_or_refits':0,'raw_final_rows_exported':False})
    print('LIMITED_ALARM_PROBABILITY_AUDIT_PASSED',models_checked,flush=True)


if __name__=='__main__':main()
