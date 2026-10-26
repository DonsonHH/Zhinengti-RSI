"""M2 aggregate report: separate tool benefit from improver benefit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from patrol_rsi.m2 import METRICS, splits

METHODS = ['plain_ddqn', 'human_revisit_skill', 'route_greedy', 'random_search', 'fixed_improver', 'updated_improver']
LABELS = ['DDQN', 'Human revisit', 'Route greedy', 'Random search', 'Fixed improver', 'Updated improver']


def main():
    p = argparse.ArgumentParser();p.add_argument('--results', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    a = p.parse_args();a.output.mkdir(parents=True, exist_ok=False)
    summaries = [json.loads((a.results/f'trial-{i}/summary.json').read_text()) for i in range(9)]
    costs, paired, audit = [], [], []
    for i, s in enumerate(summaries):
        assert s['trial'] == i and s['slots_per_method'] == 6 and not s['test_used_for_selection']
        assert (s['policy_seed'],s['proposal_seed']) == (i//3,splits()['proposal_seeds'][i%3])
        for method in METHODS:
            rows = s['test'][method]['cases']
            assert len(rows) == 24 and [r['layout_seed'] for r in rows] == list(range(7000,7024))
            for key, value in s['test'][method]['metrics'].items():
                assert np.isclose(value,np.mean([r[key] for r in rows])),(i,method,key)
            for r in rows:
                ages = np.array(r['target_ages'])
                assert ages.shape == (97,8)
                assert np.isclose(r['mean_overdue_fraction'],(ages[1:]>24).mean())
                assert r['peak_unobserved_steps'] == ages.max()
        for method in ['random_search','fixed_improver','updated_improver']:
            records = [json.loads((a.results/f'trial-{i}/{method}-{g}.json').read_text()) for g in range(6)]
            valid = [r for r in records if 'metrics' in r]
            seen = set();duplicates = 0
            for r in records:
                assert r['method']==method
                if method != 'random_search':
                    user=r['user_prompt']
                    assert user['generation']==r['generation']
                    assert all(h['generation']<r['generation'] for h in user['history'])
                    assert all(set(h)<= {'generation','patch','accepted','metrics','error'} for h in user['history'])
                    assert all(set(h.get('patch',{}))<= {'estimator_samples','estimator_refresh_interval','coordination_order','revisit_trigger'} for h in user['history'])
                    assert r['input_tokens']<=4096 and r['output_tokens']<=512
                    assert all(6000<=d['layout_seed']<6006 for d in user['diagnostic_tool_results'])
                    if method=='fixed_improver':assert r['diagnostic_plan']['tool']=='coverage'
                if 'candidate' in r:
                    signature=json.dumps(r['candidate'],sort_keys=True)
                    duplicates+=signature in seen;seen.add(signature)
            costs.append({'trial':i,'method':method,**s['actual_costs'][method],
                          'candidate_slots':6,'valid_evaluations':len(valid),'invalid':6-len(valid),
                          'duplicate_candidates':duplicates,'accepted':sum(r['accepted'] for r in records),
                          'active_tools':[r['diagnostic_plan']['tool'] for r in records]})
        for method in ['random_search','fixed_improver','updated_improver']:
            paired.append({'trial':i,'policy_seed':s['policy_seed'],'proposal_seed':s['proposal_seed'],
                           'method':method,'retained_skill':s['retained'][method]['skill'],
                           'test':s['test'][method]['metrics'],
                           'difference_from_human':{k:s['test'][method]['metrics'][k]-s['test']['human_revisit_skill']['metrics'][k] for k in METRICS}})
        audit.append({'trial':i,'all_metrics_recomputed':True,'prompt_history_and_splits_verified':True})
    # The controls are repeated to keep per-trial execution self-contained;
    # these copies do not add independent evidence.
    for policy_seed in range(3):
        for method in ['plain_ddqn','human_revisit_skill','route_greedy']:
            reference=summaries[policy_seed*3]['test'][method]['metrics']
            assert all(summaries[policy_seed*3+j]['test'][method]['metrics']==reference for j in range(3))
    table={}
    for subset in ['all','clean','noisy']:
        table[subset]={}
        for method in METHODS:
            table[subset][method]={}
            for key in METRICS:
                values=[]
                for s in summaries:
                    rows=[r for r in s['test'][method]['cases'] if subset=='all' or (r['noise']==0 if subset=='clean' else r['noise']>0)]
                    values.append(float(np.mean([r[key] for r in rows])))
                by_policy=[float(np.mean(values[i:i+3])) for i in [0,3,6]]
                table[subset][method][key]={'per_trial':values,'per_policy_seed':by_policy,
                                          'mean':float(np.mean(by_policy)),'sample_sd_across_policy_seeds':float(np.std(by_policy,ddof=1))}
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    colors=['#a2b7c7','#459582','#879d73','#b6b6b6','#376e9a','#75579a']
    for ax,key,title,factor in zip(axes,['target_coverage','mean_overdue_fraction','skill_executor_fraction'],
                                  ['Targets ever observed','Age >24 steps (target-time pairs)','Commands replaced by revisit skill'],[100,100,100]):
        values=[table['all'][m][key]['mean']*factor for m in METHODS]
        errors=[table['all'][m][key]['sample_sd_across_policy_seeds']*factor for m in METHODS]
        ax.bar(range(6),values,yerr=errors,capsize=3,color=colors)
        ax.set_xticks(range(6),LABELS,rotation=32,ha='right',fontsize=8)
        ax.set_title(title,fontsize=11);ax.set_ylabel('%');ax.grid(axis='y',alpha=.15)
    fig.suptitle('M2 | 3 frozen policy seeds x 3 proposal seeds | 24 new test cases')
    fig.savefig(a.output/'m2-comparison.png',dpi=180);fig.savefig(a.output/'m2-comparison.svg');plt.close(fig)
    # Show all paired proposal outcomes; don't hide variability behind a mean.
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for trial in range(9):
        fixed=summaries[trial]['test']['fixed_improver']['metrics'];updated=summaries[trial]['test']['updated_improver']['metrics']
        axes[0].plot([0,1],[fixed['mean_overdue_fraction']*100,updated['mean_overdue_fraction']*100],marker='o',alpha=.65,label=f'P{trial//3}/G{trial%3}')
    axes[0].set_xticks([0,1],['Fixed improver','Updated improver']);axes[0].set_ylabel('Overdue target-time pairs (%)');axes[0].set_title('9 paired outcomes; lower is better')
    for i,method in enumerate(['random_search','fixed_improver','updated_improver']):
        counts=[r['valid_evaluations'] for r in costs if r['method']==method]
        axes[1].bar(i,np.mean(counts),color=colors[i+3]);axes[1].scatter(np.full(9,i),counts,color='#263747',s=12,zorder=3)
    axes[1].set_xticks(range(3),['Search','Fixed','Updated']);axes[1].set_ylim(0,6.6);axes[1].set_ylabel('Valid evaluations / 6 slots');axes[1].set_title('Invalid and duplicate proposals consume slots')
    fig.savefig(a.output/'m2-improvers.png',dpi=180);fig.savefig(a.output/'m2-improvers.svg');plt.close(fig)
    # Preselected case: policy 0, proposal 0, first M2 test map.
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for ax,method,label in zip(axes.flat,METHODS,LABELS):
        row=summaries[0]['test'][method]['cases'][0]
        ages=np.array(row['target_ages']);im=ax.imshow(ages.T,aspect='auto',origin='lower',vmin=0,vmax=50,cmap='magma')
        ax.set_title(label);ax.set_xlabel('Patrol step');ax.set_ylabel('Inspection point')
    fig.colorbar(im,ax=axes,shrink=.7,label='Steps since observation (color capped at 50)')
    fig.suptitle('Preselected trial 0 / case 7000 | age=0 initially does not mean inspected')
    fig.savefig(a.output/'m2-age-traces.png',dpi=180);fig.savefig(a.output/'m2-age-traces.svg');plt.close(fig)
    report={'table':table,'paired':paired,'costs':costs,'methodology':{'policy_training_seeds':3,'proposal_seeds_per_policy':3,'test_cases':24,
            'error_bars':'sample SD across 3 policy-seed means, each averaging 3 proposal seeds','equal_candidate_slots':6,'equal_total_compute':False,
            'same_test_layouts_across_methods':True,'clean_and_noisy_subsets_have_different_layouts':True,'no_statistical_significance_claim':True}}
    (a.output/'aggregate.json').write_text(json.dumps(report,indent=2))
    (a.output/'artifact-audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({m:{k:table['all'][m][k]['mean'] for k in ['target_coverage','mean_overdue_fraction','mean_normalized_weighted_idleness','skill_executor_fraction']} for m in METHODS},indent=2))


if __name__=='__main__':main()
