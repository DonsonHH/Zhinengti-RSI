"""Frozen-score alarm/measurement diagnostics;final rows never released."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from patrol_rsi.limited_prediction import CELLS,COL,HeadData,load_reviewed,need,read,sha,write
from patrol_rsi.limited_metrics import probabilities
from select_limited_predictors import predict


def main():
    p=argparse.ArgumentParser()
    for n in ('protocol','review','cache','selection','final-root','output'):p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args();protocol,review=load_reviewed(args.protocol,args.review);selection=read(args.selection)
    need(selection['status']=='frozen_before_final_open' and selection['protocol_SHA256']==sha(args.protocol),'Fixed selected views')
    args.output.mkdir(exist_ok=False);failures=[];profiles=[];perseed=[];cvae=[]
    candidate_keys=[];alarm_keys=defaultdict(list);seed_by_condition={}
    for ci,(kind,horizon) in enumerate(CELLS):
        final=HeadData(args.cache/'final_test'/f'cell-{ci}',expected_phase='final_test',expected_cell=ci,expected_protocol=sha(args.protocol))
        nonfinal=HeadData(args.cache/'validation_selection'/f'cell-{ci}',expected_phase='validation_selection',expected_cell=ci,expected_protocol=sha(args.protocol))
        group=next(v for v in selection['selected_heads'] if v['cell']==ci and v['label']=='B_action')
        released=read(args.final_root/f'cell-{ci}'/'released-metrics.json')
        meta=np.asarray(final.meta[:final.n]);keys=meta[:,[2,1,7]].copy();candidate_keys.append(keys)
        for cond in np.unique(meta[:,2]):seed_by_condition[int(cond)]=int(meta[meta[:,2]==cond,3][0])
        for alias,key in (('full','primary_full'),('time','time_comparator')):
            identity=group[key];chosen=next(v for v in group['views'] if v['id']==identity)
            row=next(v for v in released['models'] if v['label']=='B_action' and v['id']==identity)
            path=args.final_root/f'cell-{ci}'/(identity+'--private-probabilities.npy')
            need(sha(path)==row['raw_predictions_SHA256'],'Same once-frozen final probability bytes')
            prob=np.load(path,mmap_mode='r');alarm=prob>chosen['threshold'];y=np.asarray(final.y[:final.n,1])
            alarm_keys[alias].append(keys[alarm])
            channel={'thermal':'temperature_measurement','electrical':'contact','control':'sensor_error'}[kind]
            available=final.x[:final.n,COL[channel+'.available']]>0
            age=np.asarray(final.x[:final.n,COL[channel+'.age_s']]);bands=np.where(~available,3,np.digitize(age,[60,300],right=True))
            rootbin=((final.x[:final.n,COL['root_time_s']]-1)//300).astype(int)
            for band in range(4):
                for clock in range(6):
                    mask=(bands==band)&(rootbin==clock)
                    profiles.append({'cell':ci,'kind':kind,'horizon_s':horizon,'view':alias,
                        'observation_age_band':('0-60','60-300','300+','missing')[band],'root_300s_bin':clock,
                        'n':int(mask.sum()),'TP':int((mask&alarm&(y==1)).sum()),'FP':int((mask&alarm&(y==0)).sum()),
                        'FN':int((mask&~alarm&(y==1)).sum()),'TN':int((mask&~alarm&(y==0)).sum())})
            for seed in np.unique(meta[:,3]):
                mask=meta[:,3]==seed
                value=probabilities(y[mask],np.asarray(prob[mask]),final.w[:final.n][mask])
                perseed.append({'cell':ci,'kind':kind,'horizon_s':horizon,'view':alias,'seed':int(seed),**value})
            if alias=='full':
                model=read(chosen['model_path']);p_nonfinal=predict(nonfinal,model,kind)
                yn=np.asarray(nonfinal.y[:nonfinal.n,1]);an=p_nonfinal>chosen['threshold']
                for error,mask in (('false_negative',(yn==1)&~an),('false_positive',(yn==0)&an)):
                    index=np.flatnonzero(mask)
                    if not len(index):continue
                    j=int(index[np.argmin(p_nonfinal[index]) if error=='false_negative' else np.argmax(p_nonfinal[index])])
                    raw=nonfinal.x[j]
                    failures.append({'source_split':'validation_selection','final_sample':False,'cell':ci,'kind':kind,'horizon_s':horizon,
                        'failure':error,'root_s':float(raw[COL['root_time_s']]),'completion_s':float(raw[COL['completion_time_s']]),
                        'task':int(nonfinal.meta[j,1]),'measurement_available':bool(raw[COL[channel+'.available']]),
                        'last_public_measurement':float(raw[COL[channel+'.value']]) if raw[COL[channel+'.available']] else None,
                        'measurement_age_s':float(raw[COL[channel+'.age_s']]) if raw[COL[channel+'.available']] else None,
                        'frozen_probability':float(p_nonfinal[j]),'frozen_threshold':chosen['threshold'],
                        'offline_A':int(nonfinal.y[j,0]),'offline_B':int(nonfinal.y[j,1]),
                        'interpretation':'Concrete nonfinal diagnostic only;not a final sample or cause proof,not used to replace selectedmodel.'})
        if kind=='thermal':
            flags=[]
            for channel in ('cvae_raw','history_assimilated'):
                jointly=(final.x[:final.n,COL[channel+'.available']]>0)&(final.x[:final.n,COL['temperature_measurement.available']]>0)
                equal=jointly&(final.x[:final.n,COL[channel+'.value']]==final.x[:final.n,COL['temperature_measurement.value']])
                flags.append({'channel':channel,'jointly_available_rows':int(jointly.sum()),'exact_measurement_equal_rows':int(equal.sum()),
                    'equal_fraction':float(equal.sum()/jointly.sum()) if jointly.any() else None})
            cvae.append({'cell':ci,'horizon_s':horizon,'counts':flags,
                'limit':'Cache can duplicate observations;additional correlated columns may change ridge penalty. Fitted augmentation gainalone is not uniquely causal CVAE attribution.'})
    combined=np.concatenate(candidate_keys);unique=np.unique(combined,axis=0)
    summary=[]
    for alias,arrays in alarm_keys.items():
        alarms=np.concatenate(arrays);unique_alarms=np.unique(alarms,axis=0);rows=[]
        for cond,seed in sorted(seed_by_condition.items()):
            n=int((combined[:,0]==cond).sum());nu=int((unique[:,0]==cond).sum())
            a=int((alarms[:,0]==cond).sum());au=int((unique_alarms[:,0]==cond).sum())
            rows.append({'seed':seed,'candidate_predictions':n,'unique_completable_candidates':nu,
                'alarms':a,'unique_alarm_completions':au,'cross_horizon_repeat_alarms':a-au})
        summary.append({'view':alias,'deduplicated_conditions':len(rows),
            'mean_per1800s':{k:float(np.mean([v[k] for v in rows])) for k in rows[0] if k!='seed'},
            'per_seed':[{'seed':seed,**{k:float(np.mean([v[k] for v in rows if v['seed']==seed])) for k in rows[0] if k!='seed'}} for seed in sorted(set(seed_by_condition.values()))]})
    write(args.output/'prediction-diagnostics.json',{'protocol_SHA256':sha(args.protocol),'selection_SHA256':sha(args.selection),
        'final_aggregate_failure_profiles':profiles,'primary_and_time_per_seed_probabilities':perseed,
        'nonfinal_concrete_failure_cases':failures,'joint_alarm_diagnostics':summary,'CVAE_observation_duplication':cvae,
        'raw_final_rows_exported':False,'selected_models_changed':False,'new_fits':0,
        'alarm_scope':'Individually legal candidate predictions,including repeatedheads/roots;counts do not imply allcan be jointlyexecuted or actual closed-loop operational alarms.'})
    print('LIMITED_FROZEN_SCORE_DIAGNOSTICS_DONE',len(failures),'nonfinalexamples',flush=True)


if __name__=='__main__':main()
