"""Reproducible seed-level review from the complete original compact index.

Postprocessing only: no policy selection, new rollout, training or model call.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def aggregate(rows):
    weight=sum(r['event_weight_sum'] for r in rows)
    return {'cases':len(rows),'segments':sum(r['metrics']['anomaly_segments'] for r in rows),
        'missed':sum(r['metrics']['missed_segments'] for r in rows),
        'weighted_wait_s':sum(r['weighted_delay_sum'] for r in rows)/weight if weight else None,
        'actions':sum(r['metrics']['selected_detours'] for r in rows)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();path=args.evidence/'report/compact-index.json.gz'
    rows=json.loads(gzip.decompress(path.read_bytes()))
    if len(rows)!=1280:
        raise RuntimeError('All eight formal tasks required, no partial cohort')
    tags=list(dict.fromkeys(r['tag'] for r in rows))
    groups={tag:{str(seed):aggregate([r for r in rows if r['tag']==tag and r['specification']['seed']==seed])
        for seed in (8300,8301)} for tag in tags}
    differences={tag:{str(seed):groups[tag][str(seed)]['missed']-groups['tour'][str(seed)]['missed'] for seed in (8300,8301)} for tag in tags}
    physical_pairs={}
    for tag in tags:
        entries={}
        for r in rows:
            if r['tag']!=tag:
                continue
            entries.setdefault(r['process_key'],[]).append(r)
        if len(entries)!=64 or any(len(v)!=2 for v in entries.values()):
            raise RuntimeError('Two noise evaluations per configuration required')
        physical_pairs[tag]=[{'process_key':key,'specification':{k:v[0]['specification'][k]
            for k in ('scenario','duration_s','seed','fault_start_s')},'noise_c':[r['specification']['noise_c'] for r in v],
            'missed':[r['metrics']['missed_segments'] for r in v],
            'weighted_wait_s':[r['metrics']['consequence_weighted_censored_delay_s'] for r in v]} for key,v in entries.items()]
    result={'input_path':'report/compact-index.json.gz','input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'all_formal_rows':len(rows),'seed_groups':groups,'miss_difference_vs_tour_by_seed':differences,
        'configuration_noise_pairs':physical_pairs,'postprocessing_only':True,
        'scope':'All strata from the same frozen tests. Two underlying physical seeds shared across scenarios/phases; no new independent evidence.'}
    examples=args.evidence/'failure-examples'
    if examples.is_dir():
        lookup={r['case']:r for r in rows};timelines=[]
        def original(case):
            p=examples/(case+'.json.gz');data=p.read_bytes()
            if hashlib.sha256(data).hexdigest()!=lookup[case]['sha256']:
                raise RuntimeError('Failure example differs from original frozen case')
            return json.loads(gzip.decompress(data))
        for phase in (335,520):
            selected=next(r for r in rows if r['tag']=='lookahead' and r['specification']['scenario']=='aux_feeder_loss+cdu_fouling' and
                r['specification']['duration_s']==180 and r['specification']['seed']==8301 and r['specification']['fault_start_s']==phase and r['specification']['noise_c']==.2)
            baseline=next(r for r in rows if r['tag']=='tour' and r['process_key']==selected['process_key'] and r['specification']['noise_c']==.2)
            before,after=original(baseline['case']),original(selected['case'])
            action=[r for r in after['workflow_decisions'] if r['task']!=r['baseline_next']][0]
            for a,b in zip(before['action_log'],after['action_log']):
                if a['time_s']<action['time_s'] and a!=b:
                    raise RuntimeError('Earlier actions differ before the one intervention')
            changed=[]
            for a,b in zip(before['anomaly_intervals'],after['anomaly_intervals']):
                if a['first_confirmation_s']==b['first_confirmation_s']:
                    continue
                later=[e['time_s'] for e in after['inspection_events'] if e['task']==a['task'] and e['time_s']>=a['onset_s']]
                changed.append({'task_id':a['task_id'],'onset_s':a['onset_s'],'end_s':a['end_s'],
                    'tour_first_confirmation_s':a['first_confirmation_s'],'branch_first_confirmation_s':b['first_confirmation_s'],
                    'branch_first_check_after_onset_s':min(later) if later else None})
            measurement=next(e for e in after['inspection_events'] if e['task']==action['task'] and e['agent']==action['agent'] and e['time_s']>action['time_s'])
            timelines.append({'baseline_case':baseline['case'],'method_case':selected['case'],'action':action,'extra_measurement':measurement,
                'missed_delta':selected['metrics']['missed_segments']-baseline['metrics']['missed_segments'],'changed_event_confirmations':changed})
        result['failure_timelines']={'selection':'Post hoc two prespecified-in-next-plan regression examples; explanatory, not additional experiments',
            'cases':timelines}
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print('B2_SEED_REVIEW',differences['lookahead'])


if __name__=='__main__':
    main()
