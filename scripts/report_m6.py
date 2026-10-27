"""M6 high-level allocation comparison with common low-level navigation."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from patrol_rsi.m6 import METRICS,VARIANTS
from m6_evidence_io import read_json

GROUPS=[('joint_vae','learned','Joint + VAE'),('no_vae','learned','Joint / no VAE'),
        ('independent_targets','learned','Independent targets'),('joint_vae','untrained_network','Initial actor'),
        ('joint_vae','random_targets','Random targets'),('joint_vae','slack','Slack heuristic'),
        ('joint_vae','risk_age_distance','Risk-age/distance'),('no_vae','risk_age_distance','Heuristic / no VAE')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    ss={v:[read_json(a.results/f'{v}-seed-{i}/summary.json') for i in range(3)] for v in VARIANTS}
    table={};costs=[];curves={};paired={};prediction={}
    for subset in ('all','balanced','tight','paired_noise_difference','paired_profile_difference'):
        table[subset]={}
        for variant,method,_ in GROUPS:
            key=variant+'/'+method;table[subset][key]={}
            for metric in METRICS:
                values=[]
                for s in ss[variant]:
                    rows=s['test'][method]['cases']
                    if subset=='paired_noise_difference':
                        clean={(r['layout_seed'],r['profile']):r for r in rows if r['noise']==0}
                        values.append(float(np.mean([r[metric]-clean[(r['layout_seed'],r['profile'])][metric] for r in rows if r['noise']>0])))
                    elif subset=='paired_profile_difference':
                        balanced={(r['layout_seed'],r['noise']):r for r in rows if r['profile']=='balanced'}
                        values.append(float(np.mean([r[metric]-balanced[(r['layout_seed'],r['noise'])][metric] for r in rows if r['profile']=='tight'])))
                    else:values.append(float(np.mean([r[metric] for r in rows if subset=='all' or r['profile']==subset])))
                table[subset][key][metric]={'per_training_seed':values,'mean':float(np.mean(values)),
                                          'sample_sd_across_training_seeds':float(np.std(values,ddof=1))}
            prediction[key]={'first24':[float(np.mean([np.mean(r['prediction_mse_series'][:24]) for r in s['test'][method]['cases']])) for s in ss[variant]],
                             'all96':[s['test'][method]['metrics']['target_prediction_mse'] for s in ss[variant]]}
    for variant in VARIANTS:
        curves[variant]=[]
        for seed,s in enumerate(ss[variant]):
            curve=[json.loads(line) for line in (a.results/f'{variant}-seed-{seed}/learning.jsonl').read_text().splitlines()]
            curves[variant].append(curve)
            costs.append({'variant':variant,'seed':seed,'selected_episode':s['selected_episode'],'episodes':s['episodes'],
                          'training_plus_validation_seconds':curve[-1]['seconds'],
                          'post_selection_evaluation_seconds':s['seconds']-curve[-1]['seconds'],
                          'gradient_updates':s['gradient_updates'],'behavior_logp_max_error':s['behavior_logp_max_error']})
        if variant!='joint_vae':
            paired[variant]=[{k:ss['joint_vae'][seed]['test']['learned']['metrics'][k]-ss[variant][seed]['test']['learned']['metrics'][k]
                              for k in METRICS} for seed in range(3)]
    fig,axes=plt.subplots(1,3,figsize=(13,4.7),layout='constrained')
    for ax,v in zip(axes,VARIANTS):
        for seed,curve in enumerate(curves[v]):
            ax.plot([r['episode'] for r in curve],[r['validation']['deadline_violation']*100 for r in curve],label=f'Seed {seed}')
        ax.set(title=v,xlabel='Training episode',ylabel='Validation violation (%)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('M6 target-level policy; shared low-level controller; validation-only checkpoint selection')
    fig.savefig(a.output/'m6-learning.png',dpi=180);fig.savefig(a.output/'m6-learning.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    colors=['#2e7298','#79a4ba','#a3b68e','#b5b5bd','#c5c5c5','#b28c56','#629182','#b2bba0']
    for ax,metric,title,factor in zip(axes,['target_coverage','deadline_violation','duplicate_assignment_fraction'],
                                   ['Targets ever observed','Per-target deadline violation','Duplicate new target assignments'],[100,100,100]):
        ax.bar(range(len(GROUPS)),[table['all'][v+'/'+m][metric]['mean']*factor for v,m,_ in GROUPS],
               yerr=[table['all'][v+'/'+m][metric]['sample_sd_across_training_seeds']*factor for v,m,_ in GROUPS],capsize=2,color=colors)
        ax.set_xticks(range(len(GROUPS)),[label for _,_,label in GROUPS],rotation=40,ha='right',fontsize=8)
        ax.set_title(title);ax.set_ylabel('%')
    fig.suptitle('M6: same sensing, commitment release, shortest-distance navigation and cyclic collision handling')
    fig.savefig(a.output/'m6-comparison.png',dpi=180);fig.savefig(a.output/'m6-comparison.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for ax,variant in zip(axes,['no_vae','independent_targets']):
        for seed in range(3):
            values=[ss[variant][seed]['test']['learned']['metrics']['deadline_violation']*100,
                    ss['joint_vae'][seed]['test']['learned']['metrics']['deadline_violation']*100]
            ax.plot([0,1],values,marker='o',label=f'Seed {seed}')
        ax.set_xticks([0,1],[variant,'Joint + VAE']);ax.set_ylabel('Test violation (%)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Paired seed comparison; lower is better; all methods use the same low-level paths')
    fig.savefig(a.output/'m6-paired.png',dpi=180);fig.savefig(a.output/'m6-paired.svg');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    shown=GROUPS[:3]+[GROUPS[4],GROUPS[5],GROUPS[6]]
    for ax,(v,m,label) in zip(axes.flat,shown):
        row=ss[v][0]['test'][m]['cases'][0];age=np.asarray(row['target_ages'])/row['target_deadlines']
        im=ax.imshow(age.T,aspect='auto',origin='lower',vmin=0,vmax=3,cmap='magma')
        ax.set(title=label,xlabel='Patrol step',ylabel='Inspection point')
    fig.colorbar(im,ax=axes,shrink=.6,label='Age/deadline; display capped at 3')
    fig.suptitle('Preselected seed 0 / first M6 test map / balanced / clean; initial age=0 does not imply observed')
    fig.savefig(a.output/'m6-age-traces.png',dpi=170);fig.savefig(a.output/'m6-age-traces.svg');plt.close(fig)
    report={'table':table,'paired_joint_minus_ablation':paired,'training_costs':costs,'learning_curves':curves,
            'prediction_error_phase':prediction,'from_scratch_training_tasks':9,'paired_training_seeds_per_variant':3,
            'common_low_level_controller_is_human_engineered':True,'centralized_dispatch_not_decentralized_deployment':True,
            'multiple_method_changes_prevent_single_factor_comparison_to_M5':True,'RSI_experiment':False,'paper_reproduction':False}
    (a.output/'aggregate.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({label:{k:table['all'][v+'/'+m][k]['mean'] for k in ['target_coverage','deadline_violation','weighted_idleness','duplicate_assignment_fraction']}
                      for v,m,label in GROUPS},indent=2))


if __name__=='__main__':main()
