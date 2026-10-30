"""Weighted probabilities and seed/process/event metrics, never row bootstrap."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .limited_prediction import need


def weighted_ap(y,p,w):
    order=np.argsort(-p,kind='stable');y=y[order];p=p[order];w=w[order]
    positive=float((w*y).sum())
    if positive==0:return None
    ends=np.r_[np.flatnonzero(p[1:]!=p[:-1]),len(p)-1]
    tp=np.cumsum(w*y)[ends];total=np.cumsum(w)[ends]
    return float(np.sum(np.diff(np.r_[0.,tp])/positive*(tp/total)))


def probabilities(y,p,w):
    y=np.asarray(y);p=np.asarray(p,dtype=float);w=np.asarray(w,dtype=float)
    need(len(y)>0 and y.shape==p.shape==w.shape and np.isin(y,(0,1)).all(),'Valid binary probability cohort')
    need(np.isfinite(p).all() and np.all((p>=0)&(p<=1)) and (w>0).all(),'Finite weighted probabilities')
    w=w/w.sum();q=np.clip(p,1e-6,1-1e-6)
    index=np.minimum((p*10).astype(int),9)
    bins=[];ece=0.
    for i in range(10):
        mask=index==i;mass=float(w[mask].sum())
        if mass:
            pred=float(np.dot(w[mask],p[mask])/mass);observed=float(np.dot(w[mask],y[mask])/mass)
            ece+=mass*abs(pred-observed)
        else:pred=observed=None
        bins.append({'lower':i/10,'upper':(i+1)/10,'n':int(mask.sum()),'weight':mass,'mean_probability':pred,'observed_fraction':observed})
    return {'samples':len(y),'positive':int(y.sum()),'weighted_prevalence':float(np.dot(w,y)),
        'Brier':float(np.dot(w,(p-y)**2)),
        'LogLoss':float(-np.dot(w,y*np.log(q)+(1-y)*np.log1p(-q))),
        'AUPRC':weighted_ap(y,p,w),'ECE':ece,'reliability_bins':bins}


def fpr_threshold(y,p,budget=.01):
    negative=np.asarray(p)[np.asarray(y)==0]
    need(len(negative)>0 and budget==.01,'Frozen1percent empirical selection negative budget')
    k=int(np.floor(budget*len(negative)))
    threshold=float(np.partition(negative,len(negative)-1-k)[len(negative)-1-k])
    need(int((negative>threshold).sum())<=k,'Strict threshold conservatively respects ties')
    return threshold


def group_rows(meta):
    """Contiguous source-condition blocks; no hidden identifier is a feature."""
    ids=np.asarray(meta[:,2]);start=np.r_[0,np.flatnonzero(ids[1:]!=ids[:-1])+1];end=np.r_[start[1:],len(ids)]
    return [(int(a),int(b)) for a,b in zip(start,end)]


def evaluate(data,label_index,p,threshold,compute_ap=True):
    y=np.asarray(data.y[:data.n,label_index]);w=np.asarray(data.w[:data.n]);meta=np.asarray(data.meta[:data.n])
    predicted=np.asarray(p)>threshold
    confirmation=np.asarray(data.y[:data.n,1])
    summary=probabilities(y,p,w) if compute_ap else {'Brier':float(np.dot(w,(p-y)**2)/w.sum())}
    negative=y==0
    summary.update(threshold=threshold,empirical_FPR=float(predicted[negative].mean()) if negative.any() else None,
        sample_recall=float(predicted[y==1].mean()) if y.any() else None,
        alarms=int(predicted.sum()),false_alarms=int((predicted&negative).sum()))
    conditions=[];processes=defaultdict(list)
    for start,end in group_rows(meta):
        local_y=y[start:end];local_p=p[start:end];alarm=predicted[start:end];local_meta=meta[start:end]
        retained=local_meta[:,5];confirmed=confirmation[start:end]
        # Fixed confirmation-opportunity event denominator for BOTH heads.
        # A-only transient/noise-positive endpoints cannot inflate eligibility.
        eligible=set(int(v) for v in np.unique(retained[(confirmed==1)&(retained>0)]))
        detected=set(int(v) for v in np.unique(retained[alarm&(confirmed==1)&(retained>0)]))
        recall=len(detected)/len(eligible) if eligible else None
        nnegative=int((local_y==0).sum());fp=int((alarm&(local_y==0)).sum())
        event_alarm_count=int((alarm&(confirmed==1)&(retained>0)).sum())
        alarm_tasks=np.unique(local_meta[alarm,1])
        row={'condition_id':int(local_meta[0,2]),'seed':int(local_meta[0,3]),'process':int(local_meta[0,4]),
            'samples':end-start,'positive':int(local_y.sum()),'negative':nnegative,
            'Brier':float(np.mean((local_p-local_y)**2)),'event_eligible':len(eligible),'event_detected':len(detected),
            'eligible_event_ids':sorted(eligible),'event_recall':recall,'FPR':fp/nnegative if nnegative else None,
            'alarms_per1800s':int(alarm.sum()),'false_alarms_per1800s':fp,'repeat_task_alarms':int(alarm.sum())-len(alarm_tasks),
            'repeat_event_alarms':event_alarm_count-len(detected),'unique_legal_candidates':len(np.unique(local_meta[:,[1,7]],axis=0)),
            'unique_alarm_completions':len(np.unique(local_meta[alarm][:,[1,7]],axis=0))}
        conditions.append(row);processes[row['seed'],row['process']].append(row)
    seed_processes=defaultdict(list)
    for (seed,process),rows in processes.items():
        need(len(rows)==2 and rows[0]['eligible_event_ids']==rows[1]['eligible_event_ids'],'Paired noise eligibility invariant;no favorable denominator changes')
        seed_processes[seed].append({'process':process,'Brier':float(np.mean([r['Brier'] for r in rows])),
            'event_recall':float(np.mean([r['event_recall'] for r in rows])) if rows[0]['event_recall'] is not None else None,
            'unique_eligible_events':rows[0]['event_eligible'],
            'alarms_per1800s':float(np.mean([r['alarms_per1800s'] for r in rows]))})
    seeds=[]
    for seed,rows in sorted(seed_processes.items()):
        own=[r for r in conditions if r['seed']==seed];nonempty=[r['event_recall'] for r in rows if r['event_recall'] is not None]
        seed_negative=sum(r['negative'] for r in own);seed_fp=sum(r['false_alarms_per1800s'] for r in own)
        seeds.append({'seed':seed,'Brier':float(np.mean([r['Brier'] for r in rows])),
            'event_recall':float(np.mean(nonempty)) if nonempty else None,'event_eligible_processes':len(nonempty),
            'event_eligible':sum(r['unique_eligible_events'] for r in rows),'numeric_processes':len(rows),
            'negative_samples':seed_negative,'FPR':seed_fp/seed_negative if seed_negative else None,
            'alarms_per1800s':float(np.mean([r['alarms_per1800s'] for r in rows]))})
    summary['seed_macro_Brier']=float(np.mean([r['Brier'] for r in seeds]))
    recalls=[r['event_recall'] for r in seeds if r['event_recall'] is not None]
    summary['seed_macro_event_recall']=float(np.mean(recalls)) if recalls else None
    summary['eligible_seeds']=len(recalls);summary['seed_groups']=len(seeds)
    summary['event_eligibility_and_detection_label']='B_action for both prediction heads;Aprobability/samplemetrics remain A_physical'
    summary['alarms_per1800s']=float(np.mean([r['alarms_per1800s'] for r in conditions]))
    summary['repeat_task_alarms_per1800s']=float(np.mean([r['repeat_task_alarms'] for r in conditions]))
    summary['repeat_event_alarms_per1800s']=float(np.mean([r['repeat_event_alarms'] for r in conditions]))
    summary['unique_eligible_events']=sum(r['unique_eligible_events'] for rows in seed_processes.values() for r in rows)
    # Public callers may publish seed summaries; per-condition event IDs remain private for final.
    return {'summary':summary,'seeds':seeds,'conditions':conditions,'processes':dict((str(seed),rows) for seed,rows in seed_processes.items())}


def rank(result,identity):
    s=result['summary'];recall=s['seed_macro_event_recall']
    return (s['seed_macro_Brier'],-recall if recall is not None else float('inf'),identity)
