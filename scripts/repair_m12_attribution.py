"""Repair legend margins from frozen aggregate data, without rerunning cases."""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--aggregate', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
groups = json.loads(args.aggregate.read_text())['main_groups']
names = ['cycle', 'fixed_actor', 'human_risk_rules', 'random_search', 'llm_no_memory', 'llm_memory']
labels = ['Fixed tour', 'Frozen PPO', 'Human rules', 'Random search', 'LLM no memory', 'LLM memory']
colors = ['#7a8793', '#17445e', '#768f97', '#c29b3b', '#648e9e', '#9c5b49']
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
x, bottom = np.arange(6), np.zeros(6)
for kind, color, label in [('forced', '#d2d9dd', 'Only one legal candidate'),
        ('ppo', '#17445e', 'PPO fallback with choice'), ('rule', '#b77751', 'Rule tool with choice'),
        ('other', '#798a6e', 'Tour / heuristic choice')]:
    values = np.asarray([groups[n]['behavior_fractions'][kind] * 100 for n in names])
    axes[0].bar(x, values, bottom=bottom, color=color, label=label)
    bottom += values
axes[0].set_ylabel('Actual dispatches (%)')
axes[0].set_title('Who actually selected the task?', fontsize=12)
axes[1].bar(x, [groups[n]['mean_estimator_calls'] for n in names], color=colors)
axes[1].set_ylabel('CVAE calls / episode')
axes[1].set_title('Tool use is executed and counted', fontsize=12)
for ax in axes:
    ax.set_xticks(x, labels, rotation=32, ha='right')
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(.5, .99), ncol=2, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, .85))
for ext in ('png', 'svg'):
    fig.savefig(args.output / f'm12-attribution.{ext}', dpi=180, facecolor='white')
plt.close(fig)
print('M12_ATTRIBUTION_LAYOUT_REPAIRED')
