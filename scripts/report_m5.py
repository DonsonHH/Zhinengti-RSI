"""M5 retraining/ablation plots from immutable result files only."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from patrol_rsi.m5 import METRICS,VARIANTS

GROUPS=[('deadline_vae','learned','Deadline + VAE'),('no_vae','learned','No VAE'),
        ('no_deadline','learned','No deadline input'),('no_shaping','learned','No map shaping'),
        ('deadline_vae','guard12','Full + guard12'),('deadline_vae','human_workflow','Full + human DSL'),
        ('deadline_vae','deadline_planner','Pure planner'),('deadline_vae','random','Random'),
        ('deadline_vae','testtime_no_vae','Full / VAE off at test'),('deadline_vae','untrained_network','Untrained network'),
        ('deadline_vae','old_m1','Old M1 DDQN')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    summaries={v:[json.loads((a.results/f'{v}-seed-{s}/summary.json').read_text()) for s in range(3)] for v in VARIANTS}
    table={};paired={};curves={};costs=[];prediction={}
    for subset in ['all','balanced','tight','noise_difference','profile_difference']:
        table[subset]={}
        for variant,method,label in GROUPS:
            key=variant+'/'+method;table[subset][key]={}
            for metric in METRICS:
                values=[]
                for s in summaries[variant]:
                    rows=s['test'][method]['cases']
                    if subset=='noise_difference':
                        clean={(r['layout_seed'],r['profile']):r for r in rows if r['noise']==0}
                        values.append(float(np.mean([r[metric]-clean[(r['layout_seed'],r['profile'])][metric] for r in rows if r['noise']>0])))
                    elif subset=='profile_difference':
                        balanced={(r['layout_seed'],r['noise']):r for r in rows if r['profile']=='balanced'}
                        values.append(float(np.mean([r[metric]-balanced[(r['layout_seed'],r['noise'])][metric] for r in rows if r['profile']=='tight'])))
                    else:
                        selected=[r for r in rows if subset=='all' or r['profile']==subset]
                        values.append(float(np.mean([r[metric] for r in selected])))
                table[subset][key][metric]={'per_training_seed':values,'mean':float(np.mean(values)),
                                          'sample_sd_across_training_seeds':float(np.std(values,ddof=1))}
            early=[];whole=[]
            for s in summaries[variant]:
                rows=s['test'][method]['cases'];early.append(float(np.mean([np.mean(r['prediction_mse_series'][:24]) for r in rows])))
                whole.append(float(np.mean([np.mean(r['prediction_mse_series']) for r in rows])))
            prediction[key]={'first_24_steps':early,'all_96_steps':whole}
    for variant in VARIANTS:
        curves[variant]=[]
        for seed,s in enumerate(summaries[variant]):
            curve=[json.loads(line) for line in (a.results/f'{variant}-seed-{seed}/learning.jsonl').read_text().splitlines()]
            curves[variant].append(curve)
            costs.append({'variant':variant,'seed':seed,'total_seconds':s['seconds'],
                          'training_plus_validation_seconds':curve[-1]['seconds'],
                          'post_selection_evaluation_seconds':s['seconds']-curve[-1]['seconds'],
                          'episodes':s['episodes'],'updates':s['ddqn_updates'],
                          'selected_episode':s['selected_episode'],'best_validation':s['best_validation']})
        if variant!='deadline_vae':
            paired[variant]=[{metric:summaries['deadline_vae'][seed]['test']['learned']['metrics'][metric]-summaries[variant][seed]['test']['learned']['metrics'][metric]
                              for metric in METRICS} for seed in range(3)]
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,variant in zip(axes.flat,VARIANTS):
        for seed,curve in enumerate(curves[variant]):
            ax.plot([r['episode'] for r in curve],[r['validation']['deadline_violation']*100 for r in curve],label=f'Seed {seed}')
        ax.set(title=variant,xlabel='Training episode',ylabel='Validation violation (%)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('M5 matched budget: 2,400 episodes per training seed; checkpoints selected by coverage-first validation rank')
    fig.savefig(a.output/'m5-learning.png',dpi=170);fig.savefig(a.output/'m5-learning.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,5.2),layout='constrained')
    colors=['#276a95','#7aa0bb','#8cac86','#bca371','#455f7e','#559281','#b68957','#bebebe','#91a5bc','#bbb0c5','#afa4a4']
    for ax,metric,title,factor in zip(axes,['target_coverage','deadline_violation','intervention_fraction'],
                                   ['Targets ever observed','Per-target deadline violation','Planner/workflow command share'],[100,100,100]):
        values=[table['all'][v+'/'+m][metric]['mean']*factor for v,m,_ in GROUPS]
        error=[table['all'][v+'/'+m][metric]['sample_sd_across_training_seeds']*factor for v,m,_ in GROUPS]
        ax.bar(range(len(GROUPS)),values,yerr=error,capsize=2,color=colors)
        ax.set_xticks(range(len(GROUPS)),[x[2] for x in GROUPS],rotation=48,ha='right',fontsize=7)
        ax.set_title(title);ax.set_ylabel('%')
    fig.suptitle('M5 new test maps x paired deadlines x paired noise | mean +/- sample SD of 3 training seeds')
    fig.savefig(a.output/'m5-attribution.png',dpi=180);fig.savefig(a.output/'m5-attribution.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,variant in zip(axes,['no_vae','no_deadline','no_shaping']):
        for seed in range(3):
            before=summaries[variant][seed]['test']['learned']['metrics']['deadline_violation']*100
            after=summaries['deadline_vae'][seed]['test']['learned']['metrics']['deadline_violation']*100
            ax.plot([0,1],[before,after],marker='o',label=f'Seed {seed}')
        ax.set_xticks([0,1],[variant,'Full']);ax.set_ylabel('Test violation (%)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Paired training-seed ablations; lower is better; raw policies without takeover')
    fig.savefig(a.output/'m5-paired.png',dpi=180);fig.savefig(a.output/'m5-paired.svg');plt.close(fig)
    # Same preselected seed, map and profile for every shown method.
    shown=[('deadline_vae','learned','Full learned'),('no_vae','learned','No VAE'),('no_deadline','learned','No deadline'),
           ('no_shaping','learned','No shaping'),('deadline_vae','guard12','Full + guard12'),('deadline_vae','deadline_planner','Pure planner')]
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for ax,(v,m,label) in zip(axes.flat,shown):
        row=summaries[v][0]['test'][m]['cases'][0];age=np.asarray(row['target_ages'])/np.asarray(row['target_deadlines'])
        im=ax.imshow(age.T,origin='lower',aspect='auto',vmin=0,vmax=3,cmap='magma')
        ax.set(title=label,xlabel='Patrol step',ylabel='Inspection point')
    fig.colorbar(im,ax=axes,shrink=.65,label='Raw age / public deadline (color capped at 3)')
    fig.suptitle('Preselected seed 0 / first test map / balanced / no noise | initial age=0 does not mean observed')
    fig.savefig(a.output/'m5-age-traces.png',dpi=170);fig.savefig(a.output/'m5-age-traces.svg');plt.close(fig)
    report={'table':table,'paired_full_minus_ablation':paired,'training_costs':costs,'learning_curves':curves,
            'prediction_error_by_phase':prediction,'training_runs':12,'test_conditions_per_method':48,'independent_training_seeds_per_variant':3,
            'main_results_no_planner_takeover':True,'equal_episode_budget_not_equal_compute_time':True,'paper_reproduction':False,'RSI_experiment':False}
    (a.output/'aggregate.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({label:{metric:table['all'][v+'/'+m][metric]['mean'] for metric in ['target_coverage','deadline_violation','weighted_idleness','intervention_fraction']} for v,m,label in GROUPS},indent=2))


if __name__=='__main__':main()
