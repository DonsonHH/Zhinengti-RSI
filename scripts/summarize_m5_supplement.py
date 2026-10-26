"""Descriptive M5 report tables from frozen outputs; no model evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
from patrol_rsi.m5 import METRICS,VARIANTS,rank


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();summaries={v:[json.loads((a.results/f'{v}-seed-{s}/summary.json').read_text()) for s in range(3)] for v in VARIANTS}
    guarded={};directions={};costs={};peaks={}
    for v in VARIANTS:
        guarded[v]={}
        for key in METRICS:
            values=[s['test']['guard12']['metrics'][key] for s in summaries[v]]
            guarded[v][key]={'per_seed':values,'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1))}
        training=[];post=[]
        for seed,s in enumerate(summaries[v]):
            curve=[json.loads(line) for line in (a.results/f'{v}-seed-{seed}/learning.jsonl').read_text().splitlines()]
            training.append(curve[-1]['seconds']);post.append(s['seconds']-curve[-1]['seconds'])
        costs[v]={'selected_episodes':[s['selected_episode'] for s in summaries[v]],
                  'training_plus_validation_seconds':training,'mean_training_plus_validation_minutes':float(np.mean(training)/60),
                  'post_selection_evaluation_seconds':post,'mean_post_selection_evaluation_minutes':float(np.mean(post)/60)}
        if v!='deadline_vae':
            full=[s['test']['learned']['metrics'] for s in summaries['deadline_vae']]
            ablated=[s['test']['learned']['metrics'] for s in summaries[v]]
            delta=np.array([f['deadline_violation']-b['deadline_violation'] for f,b in zip(full,ablated)])
            directions[v]={'full_minus_ablation_violation_per_seed':delta.tolist(),
                           'full_violation_better':int((delta < -1e-12).sum()),'full_violation_worse':int((delta>1e-12).sum()),
                           'full_violation_tied':int((abs(delta)<=1e-12).sum()),
                           'full_coverage_first_rank_better':sum(rank(f)<rank(b) for f,b in zip(full,ablated))}
        for method in summaries[v][0]['test']:
            peaks[v+'/'+method]=max(r['peak_age'] for s in summaries[v] for r in s['test'][method]['cases'])
    old=json.loads((a.report/'aggregate.json').read_text())
    errors={key:{phase:float(np.mean(values)) for phase,values in phases.items()}
            for key,phases in old['prediction_error_by_phase'].items()}
    result={'guarded_training_variants':guarded,'paired_direction_counts':directions,'training_and_evaluation_costs':costs,
            'prediction_phase_means_along_own_trajectories':errors,'worst_case_peak_age':peaks,
            'scope':'Descriptive supplementary tables computed after frozen evaluation; no extra rollout or selection',
            'statistics':'3 paired seeds per variant, not 12 independent random initializations'}
    (a.report/'supplementary-tables.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
