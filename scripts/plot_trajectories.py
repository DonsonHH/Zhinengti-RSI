"""Visualize already stored frozen-policy traces; never select checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from patrol_rsi.environment import PatrolEnv
from patrol_rsi.experiment import room

p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
methods=['learned','route_greedy','local_greedy','random']
fig,axes=plt.subplots(2,4,figsize=(14,7),layout='constrained')
audit={}
for col,method in enumerate(methods):
    trace=json.loads((a.results/f'seed-0/test-example-{method}.json').read_text())
    case=trace['case'];env=PatrolEnv(room(case['layout_seed']),case['field_seed'])
    nav=env.scenario.navigation;targets=np.argwhere(env.scenario.targets)
    positions=np.concatenate([env.positions[None],np.array([e['positions'] for e in trace['events']])])
    ax=axes[0,col];ax.imshow(~nav,cmap='Greys',vmin=0,vmax=1)
    ax.scatter(targets[:,1],targets[:,0],marker='s',s=45,facecolors='none',edgecolors='#cc3e3e',label='Inspection point')
    for agent,color in enumerate(['#1965ad','#d18123','#38873b','#8564aa']):
        xy=positions[:,agent];ax.plot(xy[:,1],xy[:,0],color=color,alpha=.7,lw=1)
        ax.scatter(xy[0,1],xy[0,0],color=color,marker='o',s=25)
        ax.scatter(xy[-1,1],xy[-1,0],color=color,marker='x',s=35)
    ax.set_title(method.replace('_',' '));ax.set_xticks([]);ax.set_yticks([])
    values=[];coverage=[]
    for event in trace['events']:
        env.step(np.array(event['actions']))
        # Weighted idleness depends on fixed evaluation truth and executed
        # positions, not on a regenerated VAE prediction or policy action.
        metrics=env.evaluation();values.append(metrics['normalized_weighted_idleness']);coverage.append(metrics['target_coverage'])
    ax=axes[1,col];ax.plot(range(1,97),values,color='#195f91',label='Weighted idleness')
    ax.plot(range(1,97),coverage,color='#459582',label='Target coverage');ax.set_ylim(0,1.05)
    ax.set_xlabel('Patrol step');ax.grid(alpha=.2)
    audit[method]={'case':case,'mean_normalized_weighted_idleness':float(np.mean(values)),'target_coverage':coverage[-1]}
axes[1,0].legend(fontsize=8)
fig.suptitle('Preselected seed 0 / first frozen test case | circles: starts, crosses: ends')
fig.savefig(a.output/'trajectories.png',dpi=170);fig.savefig(a.output/'trajectories.svg');plt.close(fig)
(a.output/'trace-replay-audit.json').write_text(json.dumps(audit,indent=2))
