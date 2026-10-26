"""Report procedure transfer against fixed diagnostics and enumeration."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from patrol_rsi.m2 import METRICS, rank
from patrol_rsi.m3 import METHODS

ALL=['human_default','fixed_trigger12','enumeration_selected',*METHODS]
LABELS=['Default16','Fixed12','Dev enumeration','Random search','Fixed coverage','Fixed revisit','Transferred']


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    summaries=[json.loads((a.results/f'trial-{i}/summary.json').read_text()) for i in range(9)]
    enums={seed:[json.loads(line) for line in (a.results/f'enumeration/policy-{seed}/configurations.jsonl').read_text().splitlines()] for seed in range(3)}
    table={};costs=[];trajectories=[];pairs=[]
    for subset in ['all','clean','noisy','paired_noise_difference']:
        table[subset]={}
        for method in ALL:
            table[subset][method]={}
            for key in METRICS:
                values=[]
                for s in summaries:
                    rows=s['test'][method]['cases']
                    if subset=='paired_noise_difference':
                        clean={r['layout_seed']:r for r in rows if r['noise']==0}
                        values.append(float(np.mean([r[key]-clean[r['layout_seed']][key] for r in rows if r['noise']>0])))
                    else:
                        selected=[r for r in rows if subset=='all' or (r['noise']==0 if subset=='clean' else r['noise']>0)]
                        values.append(float(np.mean([r[key] for r in selected])))
                by_policy=[float(np.mean(values[i:i+3])) for i in (0,3,6)]
                table[subset][method][key]={'per_trial':values,'per_policy_seed':by_policy,'mean':float(np.mean(by_policy)),
                                          'sample_sd_across_policy_seeds':float(np.std(by_policy,ddof=1))}
    for trial,s in enumerate(summaries):
        for method in METHODS:
            rows=[json.loads((a.results/f'trial-{trial}/{method}-{g}.json').read_text()) for g in range(6)]
            valid=[r for r in rows if 'metrics' in r];seen=set();duplicate=0;score=s['initial_development']
            ranks=[]
            for r in rows:
                if 'candidate' in r:
                    code=json.dumps(r['candidate'],sort_keys=True);duplicate+=code in seen;seen.add(code)
                if r['accepted']:score=r['metrics']
                ranks.append(1+sum(rank(e['metrics'])<rank(score) for e in enums[trial//3]))
            costs.append({'trial':trial,'method':method,**s['actual_costs'][method],'candidate_slots':6,
                          'valid_evaluations':len(valid),'invalid':6-len(valid),'duplicates':duplicate,
                          'accepted':sum(r['accepted'] for r in rows)})
            trajectories.append({'trial':trial,'method':method,'best_development_rank_after_slot':ranks})
        f=s['test']['fixed_revisit']['metrics'];t=s['test']['transferred_protocol']['metrics']
        pairs.append({'trial':trial,'policy_seed':s['policy_seed'],'proposal_seed':s['proposal_seed'],
                      'fixed_revisit':f,'transferred_protocol':t,
                      'overdue_difference':t['mean_overdue_fraction']-f['mean_overdue_fraction']})
    diff=np.array([p['overdue_difference'] for p in pairs]);wins=int((diff < -1e-12).sum());losses=int((diff>1e-12).sum())
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    colors=['#a1b4c3','#809c82','#cb9d49','#b7b7b7','#4987aa','#2d638e','#83659d']
    for ax,key,title,factor in zip(axes,['target_coverage','mean_overdue_fraction','mean_normalized_weighted_idleness'],
                                   ['Targets ever observed','Age >24 steps','Weighted idleness'],[100,100,1]):
        ax.bar(range(7),[table['all'][m][key]['mean']*factor for m in ALL],
               yerr=[table['all'][m][key]['sample_sd_across_policy_seeds']*factor for m in ALL],capsize=3,color=colors)
        ax.set_xticks(range(7),LABELS,rotation=35,ha='right',fontsize=8);ax.set_title(title)
        ax.set_ylabel('%' if factor==100 else 'Normalized value');ax.grid(axis='y',alpha=.15)
    fig.suptitle('M3: reset task parameters, transfer diagnostic procedure only | 12 fresh maps x 2 paired noise conditions')
    fig.savefig(a.output/'m3-comparison.png',dpi=180);fig.savefig(a.output/'m3-comparison.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for pair in pairs:
        axes[0].plot([0,1],[pair['fixed_revisit']['mean_overdue_fraction']*100,pair['transferred_protocol']['mean_overdue_fraction']*100],marker='o',alpha=.65)
    axes[0].set_xticks([0,1],['Human fixed revisit','Transferred procedure']);axes[0].set_ylabel('Overdue target-time pairs (%)')
    axes[0].set_title(f'Paired transfer: {wins} better / {losses} worse / {9-wins-losses} tied')
    for i,method in enumerate(METHODS):
        values=np.array([r['best_development_rank_after_slot'] for r in trajectories if r['method']==method])
        axes[1].plot(range(1,7),values.mean(0),marker='o',label=LABELS[3+i])
    axes[1].axhline(1,color='black',ls=':',label='110-config development best')
    axes[1].set(xlabel='Candidate slot',ylabel='Rank among 110 development configurations',title='How well does the procedure propose?')
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.savefig(a.output/'m3-transfer.png',dpi=180);fig.savefig(a.output/'m3-transfer.svg');plt.close(fig)
    # Enumeration is an extra-budget development anchor, never a test oracle.
    report={'table':table,'costs':costs,'development_rank_trajectories':trajectories,'paired_transfer':pairs,
            'transferred_vs_fixed_revisit_overdue':{'better':wins,'worse':losses,'tied':9-wins-losses},
            'enumeration_selected':[json.loads((a.results/f'enumeration/policy-{i}/selection-frozen.json').read_text()) for i in range(3)],
            'design':{'policy_training_seeds':3,'proposal_seeds_per_policy':3,'unique_test_maps':12,'paired_noise_conditions':2,
                      'enumeration_is_extra_budget':True,'enumeration_selected_on_development_not_test':True,
                      'no_task_config_or_source_test_feedback_transfer':True,'not_open_ended_recursive_improvement':True}}
    (a.output/'aggregate.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({m:{k:table['all'][m][k]['mean'] for k in ['target_coverage','mean_overdue_fraction','mean_normalized_weighted_idleness']} for m in ALL},indent=2))
    print(json.dumps(report['transferred_vs_fixed_revisit_overdue']))


if __name__=='__main__':main()
