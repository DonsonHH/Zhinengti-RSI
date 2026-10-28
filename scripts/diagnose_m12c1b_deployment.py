"""Describe every frozen test action and its realized tour-relative value.

This reads saved test cases only; it never proposes, selects or reruns a policy.
"""
import argparse
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path


def read(p):
    raw=p.read_bytes()
    return json.loads(gzip.decompress(raw) if p.suffix=='.gz' else raw)


def condition(row):
    s=row['specification']
    return tuple(s[k] for k in ('scenario','duration_s','seed','noise_c','fault_start_s'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.run_root
    common=read(root/'test-common/case-index.json')
    baseline={condition(r):r for r in common if r['tag']=='tour'}
    groups=defaultdict(list);all_rows=[]
    for folder in ('test-common','test-0','test-1','test-2'):
        for row in read(root/folder/'case-index.json'):
            base=baseline[condition(row)];original=read(root/folder/row['path'])
            selected=[r for r in original['workflow_decisions'] if r['task']!=r['baseline_next']]
            assert len(selected)==row['metrics']['selected_detours']<=1
            a,b=base['metrics'],row['metrics'];delta=b['missed_segments']-a['missed_segments']
            wait_gain=(a['consequence_weighted_censored_delay_s']-b['consequence_weighted_censored_delay_s']) if base['anomaly_intervals'] else 0.
            travel=b['travel_m']-a['travel_m'];service=b['service_s']-a['service_s']
            actual=-delta+wait_gain/3000-travel/100000
            if not selected:
                assert delta==0 and abs(wait_gain)<1e-7 and abs(travel)<1e-7 and abs(service)<1e-7
            event=selected[0] if selected else None
            record={'folder':folder,'case':row['case'],'tag':row['tag'],'intervention':
                ({k:event[k] for k in ('time_s','agent','task','tool','cue_strength','known_measured_alarm')} if event else None),
                'selected_score':event.get('skill_value') if event else None,
                'realized_net_value':actual,'missed_delta':delta,'weighted_wait_gain_s':wait_gain,
                'travel_delta_m':travel,'service_delta_s':service,
                'completed_inspections_delta':b['completed_inspections']-a['completed_inspections']}
            groups[folder+'/'+row['tag']].append(record);all_rows.append(record)
    summary={}
    for name,rows in groups.items():
        active=[r for r in rows if r['intervention']]
        times=[r['intervention']['time_s'] for r in active]
        errors=[r['selected_score']-r['realized_net_value'] for r in active if r['selected_score'] is not None and r['tag'] in ('net','net-nominal','ridge')]
        summary[name]={'cases':len(rows),'optional_actions':len(active),'action_time_min_s':min(times) if times else None,
            'action_time_max_s':max(times) if times else None,'actions_in_first_60s':sum(t<=60 for t in times),
            'fewer_misses_cases':sum(r['missed_delta']<0 for r in rows),'more_misses_cases':sum(r['missed_delta']>0 for r in rows),
            'same_misses_cases':sum(r['missed_delta']==0 for r in rows),
            'mean_travel_delta_m':sum(r['travel_delta_m'] for r in rows)/len(rows),
            'mean_service_delta_s':sum(r['service_delta_s'] for r in rows)/len(rows),
            'mean_completed_inspections_delta':sum(r['completed_inspections_delta'] for r in rows)/len(rows),
            'selected_net_score_mean_optimism':sum(errors)/len(errors) if errors else None}
    value={'all_saved_test_cases_checked':len(all_rows),'job_id':os.environ.get('SLURM_JOB_ID'),'groups':summary,'cases':all_rows,
        'postprocessing_only_no_new_search_or_policy_rollout':True,
        'note':'Net model score and realized value compared only for net/ridge. Target, actor and cue scores have different units. Lower travel can accompany fewer or different inspections within a fixed horizon.'}
    with args.output.open('x') as f:
        json.dump(value,f,indent=2);f.write('\n')
    print('C1B_DEPLOYMENT_DIAGNOSIS',len(all_rows),flush=True)


if __name__=='__main__':
    main()
