"""Source-backed M4 tool/workflow/meta-improver report and figures."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from patrol_rsi.m4 import METRICS,OPTIMIZERS,rank
from patrol_rsi.workflow import human_workflow

METHODS=['plain_ddqn','fixed_trigger12','human_workflow','human_workflow_random_fallback',*OPTIMIZERS]
LABELS=['DDQN','Trigger12','Human DSL','Human DSL/random','DSL search','LM/no memory','LM/memory','Random meta','Model meta']


def optimizer_cost(root):
    s=json.loads((root/'selection-frozen.json').read_text());rows=[json.loads((root/f'candidate-{g}.json').read_text()) for g in range(s['slots'])]
    signatures=[r['candidate_id'] for r in rows if 'candidate_id' in r]
    return {'slots':len(rows),'valid':len(signatures),'duplicates':len(signatures)-len(set(signatures)),
            'accepted':sum(r['accepted'] for r in rows),'model_calls':s['model_calls'],**s['cost']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    ss=[json.loads((a.results/f'trial-{i}/summary.json').read_text()) for i in range(9)]
    table={};costs=[];meta=[];programs=[];pairs=[]
    for subset in ['all','balanced','tight','noise_difference']:
        table[subset]={}
        for method in METHODS:
            table[subset][method]={}
            for key in METRICS:
                values=[]
                for s in ss:
                    rows=s['test'][method]['cases']
                    if subset=='noise_difference':
                        clean={r['layout_seed']:r for r in rows if r['noise']==0}
                        values.append(float(np.mean([r[key]-clean[r['layout_seed']][key] for r in rows if r['noise']>0])))
                    else:
                        rows=[r for r in rows if subset=='all' or r['profile']==subset]
                        values.append(float(np.mean([r[key] for r in rows])))
                means=[float(np.mean(values[i:i+3])) for i in [0,3,6]]
                table[subset][method][key]={'per_trial':values,'per_policy_seed':means,'mean':float(np.mean(means)),
                                           'sample_sd_across_policy_seeds':float(np.std(means,ddof=1))}
    for trial,s in enumerate(ss):
        root=a.results/f'trial-{trial}'
        for method in OPTIMIZERS:
            costs.append({'trial':trial,'method':method,'phase':'task',**optimizer_cost(root/method)})
            programs.append({'trial':trial,'method':method,'workflow_id':s['retained'][method]['workflow_id'],
                             'workflow':s['retained'][method]['workflow'],'protocol':s['protocols'][method]})
        for method in ['meta-random','meta-model']:
            for round in range(2):
                r=json.loads((root/method/f'round-{round}.json').read_text())
                total={'model_calls':int('model_proposal' in r),'input_tokens':0,'output_tokens':0,'generation_seconds':0.0,'inner_slots':0}
                if 'model_proposal' in r:
                    for k in ['input_tokens','output_tokens','generation_seconds']:total[k]+=r['model_proposal'][k]
                if 'error' not in r:
                    for side in ['incumbent','candidate']:
                        c=optimizer_cost(root/method/f'round-{round}-{side}')
                        total['model_calls']+=c['model_calls'];total['inner_slots']+=c['slots']
                        for k in ['input_tokens','output_tokens','generation_seconds']:total[k]+=c[k]
                meta.append({'trial':trial,'method':method,'round':round,'accepted':r['accepted'],
                             'valid':'error' not in r,'parent':r['parent'],'candidate':r.get('candidate'),**total})
        before=s['test']['lm_memory']['metrics'];after=s['test']['validated_model_meta']['metrics']
        pairs.append({'trial':trial,'policy_seed':s['policy_seed'],'proposal_seed':s['proposal_seed'],
                      'memory':before,'model_meta':after,'violation_difference':after['mean_deadline_violation']-before['mean_deadline_violation'],
                      'coverage_difference':after['target_coverage']-before['target_coverage']})
    colors=['#afbec9','#a2b68c','#4d8e7c','#9bbcae','#b7b7b7','#82a6bf','#306d99','#b78a62','#806294']
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for ax,key,title,factor in zip(axes,['target_coverage','mean_deadline_violation','intervention_fraction'],
                                   ['Targets ever observed','Per-target deadline violation','Workflow command share'],[100,100,100]):
        ax.bar(range(9),[table['all'][m][key]['mean']*factor for m in METHODS],
               yerr=[table['all'][m][key]['sample_sd_across_policy_seeds']*factor for m in METHODS],capsize=2,color=colors)
        ax.set_xticks(range(9),LABELS,rotation=42,ha='right',fontsize=8);ax.set_title(title);ax.set_ylabel('%')
    fig.suptitle('M4: executable workflows + validated improver updates | 3 policy x 3 generation seeds')
    fig.savefig(a.output/'m4-comparison.png',dpi=180);fig.savefig(a.output/'m4-comparison.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for pair in pairs:axes[0].plot([0,1],[pair['memory']['mean_deadline_violation']*100,pair['model_meta']['mean_deadline_violation']*100],marker='o',alpha=.65)
    axes[0].set_xticks([0,1],['Fixed LM + memory','Validated model-meta']);axes[0].set_ylabel('Deadline violation (%)');axes[0].set_title('New-task effects, with extra meta cost disclosed')
    for i,method in enumerate(['meta-random','meta-model']):
        rows=[r for r in meta if r['method']==method]
        axes[1].bar(i,sum(r['accepted'] for r in rows),color=['#b78a62','#806294'][i])
        axes[1].text(i,sum(r['accepted'] for r in rows)+.3,f"{sum(r['accepted'] for r in rows)} / {len(rows)}",ha='center')
    axes[1].set_xticks([0,1],['Random protocol','Model protocol']);axes[1].set_ylim(0,20);axes[1].set_ylabel('Accepted paired inner evaluations');axes[1].set_title('Protocol updates must pass, otherwise revert')
    fig.savefig(a.output/'m4-meta.png',dpi=180);fig.savefig(a.output/'m4-meta.svg');plt.close(fig)
    # Two preselected trial-0 executable programs, saved as readable diagrams.
    fig,axes=plt.subplots(1,2,figsize=(12,7.2),layout='constrained')
    for ax,method in zip(axes,['lm_memory','validated_model_meta']):
        p=ss[0]['retained'][method]['workflow'];ax.axis('off');steps=p['steps']
        for i,step in enumerate(steps):
            text=step['op']
            if step['op']=='dispatch':text+=f"\nwhen: {step['when']}\nrank: {step['priority']} | assign: {step['assignment']}\ncommit: {step['commit']}"
            elif step['op']=='fallback':text+=' -> '+step['policy']
            y=.93-i*(.82/max(len(steps),1));ax.text(.5,y,text,ha='center',va='top',fontsize=9,
                   bbox={'boxstyle':'round,pad=.5','fc':'#e8eef3','ec':'#427191'},transform=ax.transAxes)
            if i<len(steps)-1:
                following=.93-(i+1)*(.82/len(steps))
                ax.annotate('',xy=(.5,following+.025),xytext=(.5,y-(.135 if step['op']=='dispatch' else .05)),
                            arrowprops={'arrowstyle':'->','color':'#427191'},xycoords=ax.transAxes)
        unchanged=ss[0]['retained'][method]['workflow_id']==human_workflow().version_id
        ax.set_title(method+' | preselected trial 0\n'+('Retained initial human workflow' if unchanged else 'Retained generated workflow'),fontsize=10)
    fig.savefig(a.output/'m4-programs.png',dpi=180);fig.savefig(a.output/'m4-programs.svg');plt.close(fig)
    diff=np.array([r['violation_difference'] for r in pairs]);summary={'table':table,'task_costs':costs,'meta_preparation':meta,'programs':programs,'paired_model_meta_vs_memory':pairs,
        'violation_pair_counts':{'better':int((diff<-1e-12).sum()),'worse':int((diff>1e-12).sum()),'tied':int((abs(diff)<=1e-12).sum())},
        'total_real_model_calls':sum(s['actual_total_model_calls'] for s in ss),
        'statistics':'3 policy-seed means after averaging 3 generation seeds; repeated controls do not add independent evidence',
        'extra_meta_compute_not_equal_flat_budget':True,'bounded_DSL_not_open_ended_code_evolution':True}
    (a.output/'aggregate.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({m:{k:table['all'][m][k]['mean'] for k in ['target_coverage','mean_deadline_violation','weighted_idleness','ddqn_command_fraction']} for m in METHODS},indent=2))


if __name__=='__main__':main()
