"""Report a frozen six-proposal pilot without turning it into an RSI claim."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
s=json.loads((a.pilot/'summary.json').read_text());records=[json.loads(f.read_text()) for f in sorted(a.pilot.glob('*improver-*.json'))]
groups=['initial','fixed_improver','updated_improver'];metrics=[s['test'][g]['metrics'] for g in groups]
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
before=[r['mean_normalized_weighted_idleness'] for r in s['test']['initial']['cases']]
after=[r['mean_normalized_weighted_idleness'] for r in s['test']['fixed_improver']['cases']]
for b,c in zip(before,after):axes[0].plot([0,1],[b,c],color='#a6b6c2',alpha=.7)
axes[0].scatter(np.zeros(12),before,color='#195f91',zorder=3);axes[0].scatter(np.ones(12),after,color='#459582',zorder=3)
axes[0].set_xticks([0,1],['Initial','Retained configuration']);axes[0].set_ylabel('Weighted idleness (lower is better)');axes[0].set_title('12 new paired test cases / frozen seed 0')
for i,key in enumerate(['estimator_calls','prior_samples']):
    axes[1].bar(np.array([0,1])+(-.18 if i==0 else .18),[metrics[0][key],metrics[1][key]],width=.36,label=key.replace('_',' '),color=['#195f91','#459582'][i])
axes[1].set_xticks([0,1],['Initial','Retained']);axes[1].set_title('Fewer refreshes; same total prior samples');axes[1].legend(fontsize=8)
coverage=[m['target_coverage']*100 for m in metrics]
axes[2].bar(range(3),coverage,color=['#195f91','#459582','#7aa6bb']);axes[2].set_xticks(range(3),['Initial','Fixed improver','Updated improver'],rotation=15)
axes[2].set_ylim(0,105);axes[2].set_ylabel('Targets ever observed (%)');axes[2].set_title('Coverage falls slightly after the edit')
fig.savefig(a.output/'improver-pilot.png',dpi=180);fig.savefig(a.output/'improver-pilot.svg');plt.close(fig)
costs={g:{'input_tokens':sum(r['input_tokens'] for r in records if r['group']==g),'output_tokens':sum(r['output_tokens'] for r in records if r['group']==g),'generation_seconds':sum(r['generation_seconds'] for r in records if r['group']==g)} for g in groups[1:]}
configurations={tuple(s['retained'][g]['version'][k] for k in ['estimator_samples','estimator_refresh_interval','coordination_order']) for g in groups[1:]}
report={'metrics':dict(zip(groups,metrics)),'actual_model_costs':costs,'relative_idleness_reduction':1-metrics[1]['mean_normalized_weighted_idleness']/metrics[0]['mean_normalized_weighted_idleness'],'coverage_change_percentage_points':(metrics[1]['target_coverage']-metrics[0]['target_coverage'])*100,'both_groups_retained_same_task_configuration':len(configurations)==1,'accepted_records':s['accepted_candidates'],'unique_retained_task_configurations':len(configurations),'not_rsi_effectiveness_proof':True}
(a.output/'analysis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
