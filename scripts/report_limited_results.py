"""Exploratory post-scope paired-seed inference of pre-frozen model views."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from patrol_rsi.limited_prediction import CELLS,load_reviewed,need,read,sha,write


def interval(values,tail=.025):
    values=np.asarray(values,float)
    if not np.isfinite(values).all():return {'defined':False,'lower':None,'upper':None,'undefined_draws':int((~np.isfinite(values)).sum())}
    return {'defined':True,'lower':float(np.quantile(values,tail)),'upper':float(np.quantile(values,1-tail)),'undefined_draws':0}


def difference(full,base,indices):
    fs=sorted(full['seeds'],key=lambda v:v['seed']);bs=sorted(base['seeds'],key=lambda v:v['seed'])
    need([v['seed'] for v in fs]==[v['seed'] for v in bs],'Same independent seed groups for paired differences')
    f_b=np.array([v['Brier'] for v in fs]);b_b=np.array([v['Brier'] for v in bs])
    f_r=np.array([v['event_recall'] if v['event_recall'] is not None else np.nan for v in fs])
    b_r=np.array([v['event_recall'] if v['event_recall'] is not None else np.nan for v in bs])
    need(np.array_equal(np.isnan(f_r),np.isnan(b_r)),'No model-dependent event eligibility')
    def average_rows(x):
        count=np.isfinite(x).sum(axis=1);return np.divide(np.nansum(x,axis=1),count,out=np.full(len(x),np.nan),where=count>0)
    delta_recall=average_rows(f_r[indices])-average_rows(b_r[indices])
    base_brier=b_b[indices].mean(axis=1);delta_brier=base_brier-f_b[indices].mean(axis=1)
    return {'recall_gain':float(np.nanmean(f_r)-np.nanmean(b_r)),
        'Brier_gain':float(b_b.mean()-f_b.mean()),'relative_Brier_gain':float((b_b.mean()-f_b.mean())/b_b.mean()),
        'recall_boot':delta_recall,'Brier_boot':delta_brier,'relative_Brier_boot':delta_brier/base_brier,
        'seed_ids':[v['seed'] for v in fs]}


def public_difference(d,simultaneous_tail):
    return {k:d[k] for k in ('recall_gain','Brier_gain','relative_Brier_gain','seed_ids')} | {
        'recall_gain_CI95':interval(d['recall_boot']),'Brier_gain_CI95':interval(d['Brier_boot']),
        'relative_Brier_gain_CI95':interval(d['relative_Brier_boot']),
        'recall_gain_simultaneous':interval(d['recall_boot'],simultaneous_tail),
        'Brier_gain_simultaneous':interval(d['Brier_boot'],simultaneous_tail),
        'inference_scope':'Exploratory conditional on priorcoverage-based scope selection;not unconditional confirmatoryCI.'}


def main():
    p=argparse.ArgumentParser()
    for n in ('protocol','review','selection','final-root','output'):p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args();protocol,review=load_reviewed(args.protocol,args.review);selection=read(args.selection)
    need(selection['protocol_SHA256']==sha(args.protocol) and selection['status']=='frozen_before_final_open','Original finalscore frozen receipt')
    args.output.mkdir(exist_ok=False);models={};reports=[]
    for ci in range(7):
        path=args.final_root/f'cell-{ci}'/'released-metrics.json';r=read(path)
        need(r['selection_SHA256']==sha(args.selection) and not(r['model_refits'] or r['calibration_refits'] or r['threshold_changes']),'No post-test refitting/tuning')
        for v in r['models']:models[ci,v['label'],v['id']]=v
        reports.append(r)
    rng=np.random.default_rng(protocol['uncertainty']['seed']);indices=rng.integers(0,24,size=(2000,24))
    cells=[];all_differences={'time':[],'fixed':[]};cvae=[]
    for ci,(kind,horizon) in enumerate(CELLS):
        group=next(v for v in selection['selected_heads'] if v['cell']==ci and v['label']=='B_action')
        full=models[ci,'B_action',group['primary_full']];row={'cell':ci,'kind':kind,'horizon_s':horizon,'primary_full':group['primary_full'],
            'FPR':full['summary']['empirical_FPR'],'negative_samples':sum(v['negative_samples'] for v in full['seeds']),
            'per_seed_FPR_guard':all(v['FPR']<=.05 for v in full['seeds'] if v['negative_samples']>=100)}
        percell_pass=True
        for alias,key in (('time','time_comparator'),('fixed','fixed_comparator')):
            baseline=models[ci,'B_action',group[key]];d=difference(full,baseline,indices);all_differences[alias].append(d)
            row['versus_'+alias]=public_difference(d,.05/28)
            lower_r=row['versus_'+alias]['recall_gain_simultaneous'];lower_b=row['versus_'+alias]['Brier_gain_simultaneous']
            percell_pass &= (d['recall_gain']>=.05 and d['relative_Brier_gain']>=.05 and
                lower_r['defined'] and lower_b['defined'] and lower_r['lower']>0 and lower_b['lower']>0)
        row['LIMITED_GO_cell']=bool(percell_pass and row['FPR']<=.02 and row['per_seed_FPR_guard'])
        cells.append(row)
        if kind=='thermal':
            by_family={models[ci,'B_action',v['id']]['family']:models[ci,'B_action',v['id']] for v in group['views']}
            for family in ('public_raw_lr','public_assimilated_lr','public_all_lr'):
                if family not in by_family or 'public_lr' not in by_family:
                    cvae.append({'cell':ci,'kind':kind,'horizon_s':horizon,'added_view':family,
                        'available':False,'exploratory_increment_supported':False,'reason':'Fixed-budget auxiliary fit/calibration family unavailable;no rescue model.'})
                    continue
                d=difference(by_family[family],by_family['public_lr'],indices);v=public_difference(d,.05/18)
                cvae.append({'cell':ci,'kind':kind,'horizon_s':horizon,'added_view':family,'against':'public_lr',**v,
                    'exploratory_increment_supported':v['recall_gain_simultaneous']['defined'] and v['Brier_gain_simultaneous']['defined'] and
                        v['recall_gain_simultaneous']['lower']>0 and v['Brier_gain_simultaneous']['lower']>0,
                    'attribution_limit':'Representation-conditional fitted prediction increment. Cache can repeat measurements/correlated timestamps;not a physical causal effect or transfer confirmation.'})
    aggregate={};go=True
    for alias,values in all_differences.items():
        recall=np.mean([v['recall_boot'] for v in values],axis=0);bg=np.mean([v['Brier_boot'] for v in values],axis=0)
        # Relative improvement of equal-cell macroBrier, not mean percentages.
        base_mean=np.mean([models[i,'B_action',next(g for g in selection['selected_heads'] if g['cell']==i and g['label']=='B_action')[alias+'_comparator']]['summary']['seed_macro_Brier'] for i in range(7)])
        point_b=float(np.mean([v['Brier_gain'] for v in values]));point_r=float(np.mean([v['recall_gain'] for v in values]))
        value={'recall_gain':point_r,'Brier_gain':point_b,'relative_Brier_gain':point_b/base_mean,
            'recall_gain_CI95':interval(recall),'Brier_gain_CI95':interval(bg),'macro_Brier_comparator':base_mean}
        aggregate[alias]=value
        go &= (point_r>=.05 and point_b/base_mean>=.05 and value['recall_gain_CI95']['defined'] and value['Brier_gain_CI95']['defined'] and
            value['recall_gain_CI95']['lower']>0 and value['Brier_gain_CI95']['lower']>0)
    fpr=float(np.mean([v['FPR'] for v in cells]));no_loss=all(v['versus_'+alias]['recall_gain']>=-.05 for v in cells for alias in ('time','fixed'))
    go &= fpr<=.02 and all(v['per_seed_FPR_guard'] for v in cells) and no_loss
    verdict='GO' if go else 'LIMITED GO' if any(v['LIMITED_GO_cell'] for v in cells) else 'NO-GO'
    outcome={'stage':'B4-B1b-Limited','protocol_SHA256':sha(args.protocol),'selection_SHA256':sha(args.selection),
        'B4C_exploratory_recommendation':verdict,'aggregate_comparisons':aggregate,'cells':cells,'CVAE_increments':cvae,
        'macro_FPR':fpr,'no_cell_recall_loss_gt_005':no_loss,'model_results':reports,
        'inference_scope':'Coverage-selected exploratory held-out model scores. New scope-blind same-distribution seeds required for reliabilityconfirmation;unseen-onset/layout/site transfer remains untested.',
        'neural_retraining':0,'Qwen_calls':0,'B4C_policy_search':0,'closed_loop_gain_claimed':False,
        'LR_slots_consumed':selection['LR_slots_consumed'],'calibration_slots_consumed':selection['calibration_slots_consumed']}
    write(args.output/'results.json',outcome)
    lines=['# 七单元限定预测验证','',f'对下一步限定B4-C／确认协议的探索性建议：**{verdict}**。不是整轮巡检收益或可迁移性确认。','',
        '|专业|视野秒|模型|目标|事件召回|宏Brier|AUPRC|LogLoss|FPR|每回合报警|',
        '|---|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        for v in r['models']:
            s=v['summary'];lines.append('|'+ '|'.join(map(str,[r['kind'],r['horizon_s'],v['id'],v['label'],
                s['seed_macro_event_recall'],s['seed_macro_Brier'],s['AUPRC'],s['LogLoss'],s['empirical_FPR'],s['alarms_per1800s']]))+'|')
    (args.output/'模型比较.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print('LIMITED_POST_SCOPE_RESULTS',verdict,flush=True)


if __name__=='__main__':main()
