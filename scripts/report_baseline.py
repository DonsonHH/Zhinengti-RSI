"""Plot source-backed M1 results; variability is across training seeds."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
summaries=[];curves=[]
for seed in range(3):
    root=a.results/f'seed-{seed}'
    summaries.append(json.loads((root/'summary.json').read_text()))
    curves.append([json.loads(line) for line in (root/'learning.jsonl').read_text().splitlines()])
methods=['learned','route_greedy','local_greedy','random']
labels=['Shared DDQN + VAE','Route greedy + VAE','Local greedy + VAE','Random + VAE']
table={}
for method in methods:
    table[method]={}
    for metric in ['mean_normalized_weighted_idleness','target_coverage','max_unobserved_steps','reconstruction_mse','total_movement_distance']:
        values=[s['test'][method][metric] for s in summaries]
        table[method][metric]={'per_training_seed':values,'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1))}
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for seed,curve in enumerate(curves):
    axes[0].plot([r['episode'] for r in curve],[r['validation']['mean_normalized_weighted_idleness'] for r in curve],label=f'Train seed {seed}')
axes[0].set(xlabel='Training episode',ylabel='Validation mean weighted idleness',title='Validation-only checkpoint selection');axes[0].legend();axes[0].grid(alpha=.2)
values=[table[m]['mean_normalized_weighted_idleness']['mean'] for m in methods]
errors=[table[m]['mean_normalized_weighted_idleness']['sample_sd'] for m in methods]
axes[1].bar(range(4),values,yerr=errors,capsize=4,color=['#195f91','#459582','#a0b5c4','#b6b6b6'])
axes[1].set_xticks(range(4),['DDQN','Route greedy','Local greedy','Random'],rotation=12)
axes[1].set(ylabel='Test mean weighted idleness (lower is better)',title='12 test cases per training seed')
fig.savefig(a.output/'baseline.png',dpi=180);fig.savefig(a.output/'baseline.svg');plt.close(fig)
report={'training_seeds':3,'test_cases_per_seed':12,'variability':'sample SD across three training-seed means; not 36 independent training runs','table':table,'selected_episodes':[s['selected_episode'] for s in summaries],'paper_reproduction':False,'rsi_enabled':False}
(a.output/'aggregate.json').write_text(json.dumps(report,indent=2))
text='# M1固定基线结果\n\n3个独立训练种子，每种方法每个种子12个测试场景。表中为训练种子均值的平均值±样本标准差。\n\n'
text+='| 方法 | 加权未检程度（低优） | 检查点覆盖（高优） | 终点最长未检步数（低优） |\n|---|---:|---:|---:|\n'
for method,label in zip(methods,labels):
    cells=[]
    for metric in ['mean_normalized_weighted_idleness','target_coverage','max_unobserved_steps']:
        v=table[method][metric];cells.append(f"{v['mean']:.4f} ± {v['sample_sd']:.4f}")
    text+='| '+label+' | '+' | '.join(cells)+' |\n'
text+='\n![固定基线学习曲线与测试比较](baseline.png)\n\n检查点按验证集选择。本结果属于合成机房与缩小网络设置，未复现原论文表格，且未启用RSI。\n'
(a.output/'结果说明.md').write_text(text,encoding='utf-8')
print(json.dumps(report,indent=2))
