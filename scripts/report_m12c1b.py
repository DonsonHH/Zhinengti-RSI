"""C1-B0 report from frozen cases; no training or selection occurs here."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(p):
    return json.loads(p.read_text())


def write(p,v):
    with p.open('x') as f:
        json.dump(v,f,indent=2);f.write('\n')


def quality(rows):
    confusion=Counter()
    truth=[];fresh=[];est={k:[] for k in ('hold','linear','nominal','history')}
    for row in rows:
        confusion.update(row['confusion'])
        for point in row['temperature']:
            truth.extend(point['truth_c']);fresh.extend(point['fresh'])
            for name in est:
                est[name].extend(point[name])
    y=np.asarray(truth);direct=np.asarray(fresh,dtype=bool);result={}
    for name,values in est.items():
        pred=np.asarray(values);error=pred-y
        subsets={}
        for group,mask in (('all',np.ones(len(y),dtype=bool)),('not_currently_observed',~direct)):
            a,b=y[mask]>65,pred[mask]>65
            tp=int((a&b).sum());fp=int((~a&b).sum());fn=int((a&~b).sum());tn=int((~a&~b).sum())
            subsets[group]={'node_time_samples':int(mask.sum()),'rmse_c':float(np.sqrt(np.mean(error[mask]**2))),
                'mae_c':float(np.mean(abs(error[mask]))),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
                'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None}
        result[name]=subsets
    tp,fp,fn,tn=(confusion[k] for k in ('tp','fp','fn','tn'))
    return {'baseline_observation_conditions':len(rows),'temperature':result,'public_cue':dict(confusion)|{
        'candidate_instances':sum(confusion.values()),'precision':tp/(tp+fp) if tp+fp else None,
        'recall':tp/(tp+fn) if tp+fn else None,'false_positive_rate':fp/(fp+tn) if fp+tn else None},
        'sampling_note':'Repeated legal optional candidate instances; not independent plants or calibrated uncertainty. Current estimates, not future forecasts.'}


def branch_stats(rows):
    counts=Counter();net=[];collateral=Counter()
    for row in rows:
        label=row['labels'];delta=label['missed_delta']
        counts['fewer_misses' if delta<0 else 'more_misses' if delta>0 else 'same_misses']+=1
        collateral[str(label['collateral_lost_segments'])]+=1;net.append(label['net'])
    return {'samples':len(rows),'miss_change_counts':dict(counts),'collateral_lost_segments_histogram':dict(collateral),
        'net_value_min':min(net) if net else None,'net_value_mean':float(np.mean(net)) if net else None,
        'net_value_max':max(net) if net else None,'target_recoveries':sum(r['labels']['target_recovered_segments'] for r in rows)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.run_root;out=args.output;out.mkdir(exist_ok=False)
    common=read(root/'test-common/test-groups.json')
    trials=[read(root/f'test-{i}/test-groups.json') for i in range(3)]
    selections=[read(root/f'fit-{i}/selections-frozen.json') for i in range(3)]
    data=read(root/'data/dataset.json');heldout=read(root/'heldout/heldout-labels.json')
    quality_result=quality(read(root/'test-common/test-tour-samples-and-quality.json'))
    natural=[r for r in data if any(s.startswith('natural_') for s in r['provenance'])]
    teacher=[r for r in data if any(s.startswith('teacher_') for s in r['provenance'])]
    stats={'all_training':branch_stats(data),'natural_training':branch_stats(natural),'teacher_training':branch_stats(teacher),
        'heldout_natural':branch_stats(heldout),'overlap_training':sum(r in teacher for r in natural)}
    regression={}
    for key in heldout[0]['predictions'] if heldout else []:
        method=key.split('/')[1];objective='target' if method=='target' else 'net'
        error=np.array([r['predictions'][key]-r['labels'][objective] for r in heldout])
        regression[key]={'objective':objective,'samples':len(error),'mae':float(np.mean(abs(error))),
            'mse':float(np.mean(error**2)),'note':'Natural fixed sample, not all-candidate calibrated value.'}
    table={k:[v['aggregate']] for k,v in common.items()}
    for name in ('untrained','target','net','net-nominal'):
        table[name]=[v[name]['aggregate'] for v in trials]
    order=('tour','cue_rule','actor','untrained','target','net','ridge','net-nominal')
    outcome={}
    baseline=common['tour']['aggregate']
    for name in order:
        entries=table[name]
        outcome[name]={'by_initialization':entries,'mean_missed':float(np.mean([r['missed'] for r in entries])),
            'missed_range':[min(r['missed'] for r in entries),max(r['missed'] for r in entries)],
            'mean_weighted_wait_s':float(np.mean([r['weighted_wait_s'] for r in entries])),
            'wait_range_s':[min(r['weighted_wait_s'] for r in entries),max(r['weighted_wait_s'] for r in entries)],
            'mean_extra_travel_m_per_case':float(np.mean([(r['travel_m']-baseline['travel_m'])/r['cases'] for r in entries])),
            'mean_optional_actions':float(np.mean([r['extra_checks'] for r in entries]))}
    folders=['data']+[f'fit-{i}' for i in range(3)]+['test-common']+[f'test-{i}' for i in range(3)]+['heldout']
    counts={name:read(root/name/'summary.json')['actual_cases'] for name in folders}
    audits={name:read(root/(name+'-audit.json')) for name in folders}
    physical={};signatures=set();compact=[]
    for name in folders:
        for key,value in read(root/name/'process-index.json').items():
            if key in physical and physical[key]!=value['numeric_digest']:
                raise RuntimeError('Cross-stage physical process drift')
            physical[key]=value['numeric_digest']
        for row in read(root/name/'case-index.json'):
            sig=json.dumps({'specification':row['specification'],'deployment':row['deployment']},sort_keys=True)
            signatures.add(sig)
            compact.append({'folder':name,'case':row['case'],'specification':row['specification'],
                'deployment':row['deployment'],'metrics':row['metrics'],'event_weight_sum':row['event_weight_sum'],
                'weighted_delay_sum':row['weighted_delay_sum'],'anomaly_intervals':row['anomaly_intervals']})
    result={'stage':'C1-B0','outcomes':outcome,'test_quality':quality_result,'counterfactual_samples':stats,
        'heldout_value_regression':regression,'selections':selections,'actual_cases_by_stage':counts,
        'total_actual_formal_rollouts':sum(counts.values()),'unique_specification_deployment_signatures':len(signatures),
        'independent_physical_processes_across_splits':len(physical),'test_physical_processes':16,'test_observation_conditions':32,
        'old_model_weights_unchanged':True,'PPO_or_CVAE_retrained':False,'new_MLP_training_runs':6,'training_initializations':3,
        'global_optional_budget':1,'LLM_calls':0,'improver_updated':False,'audits':audits,
        'limitations':['Synthetic known fixed layout and deterministic access/time contracts',
            'Teacher-biased training sample; heldout four-action sample is not deployment prevalence',
            'One global optional action with tour continuation; no repeated-detour value claim',
            'Three small scorer initializations share physical tests; not three independent plants or PPO seeds',
            'No test or automatic RSI gain claim follows from training fit']}
    write(out/'aggregate.json',result)
    (out/'compact-metrics.json.gz').write_bytes(gzip.compress((json.dumps(compact)+'\n').encode(),mtime=0))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),gridspec_kw={'width_ratios':[1.3,1]})
    methods=('hold','linear','nominal','history');x=np.arange(4)
    for offset,subset,label in ((-.18,'all','All node-times'),(.18,'not_currently_observed','Not directly observed now')):
        axes[0].bar(x+offset,[quality_result['temperature'][k][subset]['rmse_c'] for k in methods],width=.35,label=label)
    axes[0].set(xticks=x,xticklabels=['Hold','Linear','Nominal','CVAE'],ylabel='Current temperature RMSE (C)',title='Same tour observation histories');axes[0].legend(fontsize=9)
    cue=quality_result['public_cue'];rates=[cue['recall'] or 0.,cue['false_positive_rate'] or 0.]
    bars=axes[1].bar(['Cue on valid\nconfirmation','Cue on other\ncandidates'],rates,color=['#2369ad','#d18a35'])
    for bar,v in zip(bars,rates):
        axes[1].text(bar.get_x()+bar.get_width()/2,v+.015,f'{v:.1%}',ha='center')
    axes[1].set(ylim=(0,1.15),ylabel='Candidate-instance fraction',title='Unconfirmed public cue at threshold 0.05')
    axes[1].text(.02,.98,'TP={tp:,}  FP={fp:,}\nFN={fn:,}  TN={tn:,}'.format(**cue),transform=axes[1].transAxes,va='top',fontsize=10)
    fig.text(.06,-.02,'32 observation conditions / 16 physical processes. Repeated candidates are dependent. This is current estimation.',fontsize=10)
    fig.tight_layout();save(fig,'c1b-cue-and-estimation')
    fig,axes=plt.subplots(1,2,figsize=(16,4.8));x=np.arange(len(order));labels=['Tour','Cue','Frozen\nPPO','Untrained\nMLP','Target\nMLP','Net\nMLP','Ridge','Net +\nnominal']
    for ax,field,ylabel in ((axes[0],'missed','Missed anomaly records'),(axes[1],'weighted_wait_s','Pooled weighted censored delay (s)')):
        means=[np.mean([r[field] for r in table[k]]) for k in order]
        ax.bar(x,means,color=['#8b96a4','#52a2b4','#7b83b8','#b2b2b2','#d4a35c','#28649c','#60977a','#9a73aa'])
        for j,k in enumerate(order):
            for seed,row in enumerate(table[k]):
                ax.scatter(j+(seed-1)*.09 if len(table[k])>1 else j,row[field],s=22,c='black',zorder=4)
        ax.set(xticks=x,xticklabels=labels,ylabel=ylabel)
        ax.tick_params(axis='x',labelsize=9)
        ax.axhline(baseline[field],color='#334155',linestyle='--',lw=1)
    axes[0].set_title(f"Same {baseline['segments']} anomaly records / 32 observation conditions")
    axes[1].set_title('Points = scorer initializations; not extra physical trials')
    fig.text(.06,-.01,'At most one optional global inspection, then tour. All exact revisit constraints checked. No repeated-detour or RSI claim.',fontsize=10)
    fig.tight_layout();save(fig,'c1b-one-check-outcomes')
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.1),gridspec_kw={'width_ratios':[1.05,1]});groups=('natural_training','teacher_training','heldout_natural');bottom=np.zeros(3)
    for key,label,color in (('fewer_misses','Fewer misses','#387a9b'),('same_misses','Same misses','#aeb8c5'),('more_misses','More misses','#c46a56')):
        values=[stats[g]['miss_change_counts'].get(key,0) for g in groups]
        axes[0].bar(np.arange(3),values,bottom=bottom,label=label,color=color);bottom+=values
    axes[0].set(xticks=np.arange(3),xticklabels=['Natural\ntraining','Teacher\ntraining','Heldout\nnatural'],ylabel='Separate one-action branch records',title='Whole-episode miss effect of sampled actions');axes[0].legend(fontsize=9)
    keys=list(regression);errors=[regression[k]['mae'] for k in keys]
    axes[1].barh(np.arange(len(keys)),errors,color=['#d4a35c' if regression[k]['objective']=='target' else '#28649c' for k in keys])
    axes[1].set(yticks=np.arange(len(keys)),yticklabels=['Seed '+k.replace('/',' / ') for k in keys],xlabel='MAE against declared training objective',title='Frozen predictions before heldout labels')
    axes[1].invert_yaxis();axes[1].tick_params(axis='y',labelsize=9)
    fig.text(.06,-.025,'Teacher and natural training sets can overlap. Target-model MAE uses the target-only objective; blue bars use net value.',fontsize=10)
    fig.tight_layout();save(fig,'c1b-labels-and-value')
    print('C1B_REPORT',sum(counts.values()),'actual formal rollouts',flush=True)


if __name__=='__main__':
    main()
