"""All frozen B2 groups, phase/noise strata and actual wait/action attribution."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(p):
    return json.loads(p.read_bytes())


def write(p,v):
    with p.open('x') as f:
        json.dump(v,f,indent=2);f.write('\n')


def condition(r):
    return tuple(r['specification'][k] for k in ('scenario','duration_s','seed','fault_start_s','noise_c'))


def aggregate(rows):
    weights=sum(r['event_weight_sum'] for r in rows)
    return {'cases':len(rows),'segments':sum(r['metrics']['anomaly_segments'] for r in rows),
        'missed':sum(r['metrics']['missed_segments'] for r in rows),
        'weighted_wait_s':sum(r['weighted_delay_sum'] for r in rows)/weights if weights else None,
        'travel_m':sum(r['metrics']['travel_m'] for r in rows),'service_s':sum(r['metrics']['service_s'] for r in rows),
        'extra_checks':sum(r['metrics']['selected_detours'] for r in rows),
        'false_measured_confirmations':sum(r['metrics']['false_measured_confirmations'] for r in rows),
        'exact_violations':sum(r['metrics']['tasks_violating_exact_gap'] for r in rows)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.run;out=args.output;out.mkdir(exist_ok=False)
    index=[];audits={};processes={};summaries={}
    for i in range(8):
        folder=root/f'chunk-{i}'
        rows=read(folder/'case-index.json')
        index.extend(dict(r,folder=f'chunk-{i}') for r in rows)
        audits[str(i)]=read(root/f'chunk-{i}-audit.json');summaries[str(i)]=read(folder/'summary.json')
        for key,v in read(folder/'process-index.json').items():
            if key in processes and processes[key]!=v['numeric_digest']:
                raise RuntimeError('Same configuration has differing physical arrays')
            processes[key]=v['numeric_digest']
    tags=list(read(root/'chunk-0/groups.json'));baselines={condition(r):r for r in index if r['tag']=='tour'}
    groups={tag:aggregate([r for r in index if r['tag']==tag]) for tag in tags}
    paired={};actions={};phase_groups={};noise_groups={};scenario_groups={}
    for tag in tags:
        rows=[r for r in index if r['tag']==tag];pairs=[]
        for r in rows:
            base=baselines[condition(r)];a,b=base['metrics'],r['metrics']
            pairs.append({'case':r['case'],'process_key':r['process_key'],'specification':r['specification'],
                'missed_delta':b['missed_segments']-a['missed_segments'],
                'weighted_wait_delta_s':b['consequence_weighted_censored_delay_s']-a['consequence_weighted_censored_delay_s'] if r['anomaly_intervals'] else None,
                'travel_delta_m':b['travel_m']-a['travel_m'],'service_delta_s':b['service_s']-a['service_s']})
        paired[tag]={'fewer_misses':sum(r['missed_delta']<0 for r in pairs),'same_misses':sum(r['missed_delta']==0 for r in pairs),
            'more_misses':sum(r['missed_delta']>0 for r in pairs),'cases':pairs}
        times=[r['optional_action']['time_s'] for r in rows if r['optional_action'] is not None]
        actions[tag]={'actions':len(times),'times_s':times,'unused_budget_cases':len(rows)-len(times),
            'unused_after_positive_wait':sum(r['optional_action'] is None and r['positive_now_wait_frames']>0 for r in rows),
            'positive_now_wait_frames':sum(r['positive_now_wait_frames'] for r in rows),
            'total_wait_value_frames':sum(r['total_wait_value_frames'] for r in rows),
            'mean_decision_wall_s':float(np.mean([r['metrics']['decision_CPU_seconds'] for r in rows])),
            'mean_estimator_inference_s':float(np.mean([r['metrics']['history_estimator_inference_s'] for r in rows]))}
        phase_groups[tag]={str(phase):aggregate([r for r in rows if r['specification']['fault_start_s']==phase]) for phase in (150,335,520,705)}
        noise_groups[tag]={str(noise):aggregate([r for r in rows if r['specification']['noise_c']==noise]) for noise in (.2,1.)}
        scenarios=sorted({(r['specification']['scenario'],r['specification']['duration_s']) for r in rows})
        scenario_groups[tag]={a+'--'+str(b):aggregate([r for r in rows if r['specification']['scenario']==a and r['specification']['duration_s']==b]) for a,b in scenarios}
    contrasts={}
    for name,left,right in (('horizon_vs_now','lookahead','evidence_now'),('horizon_vs_permuted','lookahead','lookahead_permuted'),
        ('conservative_vs_original','lookahead_conservative','lookahead')):
        a={condition(r):r for r in index if r['tag']==left};b={condition(r):r for r in index if r['tag']==right}
        rows=[]
        for key,x in a.items():
            y=b[key];rows.append({'specification':x['specification'],'left_case':x['case'],'right_case':y['case'],
                'missed_delta_left_minus_right':x['metrics']['missed_segments']-y['metrics']['missed_segments'],
                'left_action':x['optional_action'],'right_action':y['optional_action'],
                'left_positive_wait_frames':x['positive_now_wait_frames'],'right_positive_wait_frames':y['positive_now_wait_frames']})
        contrasts[name]=rows
    blind=[groups[f'blind_{i}'] for i in (0,1,2)]
    result={'stage':'C1-B2','groups':groups,'paired_to_tour':paired,'phase_groups':phase_groups,'noise_groups':noise_groups,
        'scenario_groups':scenario_groups,'actions_and_waits':actions,'method_contrasts':contrasts,
        'blind_all_three':{'missed':[r['missed'] for r in blind],'weighted_wait_s':[r['weighted_wait_s'] for r in blind],
            'mean_missed':float(np.mean([r['missed'] for r in blind])),'mean_weighted_wait_s':float(np.mean([r['weighted_wait_s'] for r in blind]))},
        'formal_cases':len(index),'physical_configurations':len(processes),'unique_numeric_process_fingerprints':len(set(processes.values())),
        'observation_conditions':len(baselines),'physical_seeds':[8300,8301],'fault_start_grid_s':[150,335,520,705],
        'summaries':summaries,'audits':audits,'LLM_calls':0,'new_neural_training':False,'improver_updated':False,
        'notes':['Same physical seeds shared across scenarios/phases; not 64 independent random plants',
            'Paired noise and methods are dependent; blind seeds are action controls, not new physical worlds',
            'Permutation changes association of public observations to tasks; legal schedules stay actual',
            'Conservative controller only lowers future utilities; current utilities unchanged',
            'Current comparison is a controlled timing subproblem of the paper/RSI goal']}
    write(out/'aggregate.json',result)
    (out/'compact-index.json.gz').write_bytes(gzip.compress((json.dumps(index)+'\n').encode(),mtime=0))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    labels=['Tour','Cue\nnow','Evidence\nnow','H120','H120\nshuffle','H120\nmask','H120\nconserv.','Blind 0','Blind 1','Blind 2']
    colors=['#8995a4','#8995a4','#5f9fa1','#286b9e','#b398bf','#a8adb5','#bc9459','#a9acae','#a9acae','#a9acae']
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(16,4.9));x=np.arange(len(tags))
    for ax,field,ylabel in ((axes[0],'missed','Missed anomaly records'),(axes[1],'weighted_wait_s','Weighted censored delay (s)')):
        values=[groups[k][field] for k in tags];bars=ax.bar(x,values,color=colors)
        for bar,v in zip(bars,values):
            ax.annotate(f'{v:.1f}',(bar.get_x()+bar.get_width()/2,v),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
        ax.set(xticks=x,xticklabels=labels,ylabel=ylabel);ax.tick_params(axis='x',labelsize=8)
        ax.axhline(groups['tour'][field],color='#334155',ls='--',lw=1);ax.margins(y=.16)
    fig.suptitle('128 paired observation conditions / 64 configurations / 2 underlying physical seeds')
    fig.text(.05,-.015,'One optional global check. All three public blind seeds are shown; no best-seed selection.',fontsize=10)
    fig.tight_layout();save(fig,'b2-confirmation-outcomes')
    fig,axes=plt.subplots(1,2,figsize=(15,5.3),gridspec_kw={'width_ratios':[1.15,1]})
    phases=(150,335,520,705)
    heat=np.array([[phase_groups[k][str(p)]['missed']-phase_groups['tour'][str(p)]['missed'] for p in phases] for k in tags])
    bound=max(1,int(abs(heat).max()));im=axes[0].imshow(heat,cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto')
    axes[0].set(xticks=range(4),xticklabels=phases,yticks=range(10),yticklabels=[s.replace('\n',' ') for s in labels],xlabel='Evaluator injection phase (s)',title='Miss difference vs tour: negative is better')
    for i in range(10):
        for j in range(4):
            axes[0].text(j,i,str(int(heat[i,j])),ha='center',va='center',fontsize=9,color='white' if abs(heat[i,j])>.6*bound else 'black')
    fig.colorbar(im,ax=axes[0],fraction=.04)
    subset=('lookahead','lookahead_conservative','lookahead_permuted','blind_0','blind_1','blind_2');x=np.arange(len(subset))
    for offset,noise,color in ((-.18,.2,'#286b9e'),(.18,1.,'#d39b5c')):
        values=[noise_groups[k][str(noise)]['missed']-noise_groups['tour'][str(noise)]['missed'] for k in subset]
        axes[1].bar(x+offset,values,.35,label=f'Noise {noise} C',color=color)
    axes[1].set(xticks=x,xticklabels=['H120','Conserv.','Shuffle','Blind 0','Blind 1','Blind 2'],ylabel='Miss difference vs tour',title='Noise strata, same physical configurations')
    axes[1].axhline(0,color='#334155',lw=1);axes[1].legend(fontsize=9)
    fig.text(.05,-.015,'Injection phase is evaluator metadata and never supplied to a deployed policy.',fontsize=10)
    fig.tight_layout();save(fig,'b2-phase-and-noise')
    fig,axes=plt.subplots(1,2,figsize=(15,5.2));y=np.arange(len(tags))
    used=[actions[k]['actions'] for k in tags];unused=[actions[k]['unused_budget_cases'] for k in tags]
    axes[0].barh(y,used,color='#286b9e');axes[0].barh(y,unused,left=used,color='#e4e7eb')
    for i,k in enumerate(tags):
        axes[0].text(129,i,f"{actions[k]['unused_after_positive_wait']} stalled",va='center',fontsize=8)
    axes[0].set(yticks=y,yticklabels=[s.replace('\n',' ') for s in labels],xlabel='Blue = executed; gray = unused (cases / 128)',xlim=(0,165),title='Unused after rejecting positive NOW = stalled')
    axes[0].invert_yaxis()
    for i,k in enumerate(tags):
        values=actions[k]['times_s'];axes[1].scatter(values,[i]*len(values),s=16,alpha=.25,color=colors[i])
    axes[1].set(yticks=y,yticklabels=[s.replace('\n',' ') for s in labels],xlabel='Actual optional dispatch time (s)',title='Repeated actions overlap; no point = no check');axes[1].invert_yaxis()
    fig.text(.05,-.015,'Stalled is a descriptive flag, not proof that checking would have helped. Shared configurations are dependent.',fontsize=10)
    fig.tight_layout();save(fig,'b2-actions-and-waiting')
    print('B2_REPORT_COMPLETE',len(index),flush=True)


if __name__=='__main__':
    main()
