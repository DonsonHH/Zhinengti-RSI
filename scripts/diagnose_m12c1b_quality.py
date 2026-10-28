"""Separate duplicate cues and future-valid measurements in fixed test-tour replay.

Evaluator labels never enter the tour policy. No new candidate/threshold search.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import os
from pathlib import Path

import torch

import patrol_rsi.value_run as runner
from patrol_rsi.m12 import actor_scorer, load_actors
from patrol_rsi.m11 import frozen_predictor
from audit_m12c1b import checked_snapshot, positive_at_completion, need, sha
from audit_m111 import equivalent
from audit_m11 import independent_schedule, independent_events


class Details:
    def __init__(self, intervals):
        self.intervals=intervals;self.counts=Counter();self.by_role=defaultdict(Counter);self.first={}

    def observe(self,env,groups):
        for g in groups:
            for r in g['candidates']:
                if not r['is_optional']:
                    continue
                segment=next((i for i,v in enumerate(self.intervals) if v['task']==r['task'] and v['onset_s']<r['candidate_finish_s']<=v['end_s']),None)
                positive=segment is not None and positive_at_completion(env,r['task'],r['candidate_finish_s'])
                already=False
                if segment is not None:
                    when=self.intervals[segment]['first_confirmation_s']
                    already=when is not None and when<=g['time_s']
                raw=r['cue_strength']>=.05;filtered=raw and not r['known_measured_alarm']
                items=['all_candidates']
                if raw:
                    items.append('raw_cued')
                if filtered:
                    items.append('unconfirmed_public_cued')
                if r['known_measured_alarm']:
                    items.append('latest_measurement_already_alarm')
                for label,selected in (('raw',raw),('public_filtered',filtered)):
                    if selected:
                        items.append(label+('_valid_completion' if positive else '_no_valid_completion'))
                        if positive:
                            items.append(label+('_already_confirmed_segment' if already else '_not_previously_confirmed_segment'))
                # Confusion for actual new segment confirmation opportunity, separate from a valid remeasurement.
                new_positive=positive and not already
                items.append('new_opportunity_'+('tp' if filtered and new_positive else 'fp' if filtered else 'fn' if new_positive else 'tn'))
                self.counts.update(items);self.by_role[r['profession']].update(items)
                if filtered and new_positive and segment not in self.first:
                    v=self.intervals[segment]
                    self.first[segment]={'segment_index':segment,'task_id':r['task_id'],'profession':r['profession'],
                        'dispatch_time_s':g['time_s'],'completion_time_s':r['candidate_finish_s'],
                        'onset_s_evaluator_only':v['onset_s'],'end_s_evaluator_only':v['end_s'],
                        'dispatch_lead_to_onset_s':v['onset_s']-g['time_s'],
                        'completion_delay_from_onset_s':r['candidate_finish_s']-v['onset_s'],
                        'travel_and_check_s':r['candidate_finish_s']-g['time_s'],
                        'baseline_first_confirmation_s':v['first_confirmation_s']}


def main():
    p=argparse.ArgumentParser()
    for key in ('run-root','actors','estimator-weights','estimator-manifest','output'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    models,_=load_actors(args.actors);score=actor_scorer(models[0]);predictor=frozen_predictor(args.estimator_weights,args.estimator_manifest)
    root=args.run_root/'test-common';index=json.loads((root/'case-index.json').read_text())
    rows=[r for r in index if r['tag']=='tour'];need(len(rows)==32,'All fixed tour test conditions')
    original_snapshot=runner.snapshot;runner.snapshot=checked_snapshot
    cases=[];total=Counter();roles=defaultdict(Counter)
    try:
        for row in rows:
            path=root/row['path'];need(sha(path)==row['sha256'],'Original fixed-tour case bytes')
            original=json.loads(gzip.decompress(path.read_bytes()))
            observer=Details(original['anomaly_intervals'])
            replay,env=runner.run(row['specification'],predictor,score,observer=observer)
            for key in ('action_log','inspection_events','metrics','scalar_trace','anomaly_intervals'):
                equivalent(replay[key],original[key],'Unchanged test-tour '+key)
            independent_schedule(replay,env);independent_events(replay,env)
            total.update(observer.counts)
            for role,counts in observer.by_role.items():
                roles[role].update(counts)
            cases.append({'case':row['case'],'counts':dict(observer.counts),'by_role':dict(observer.by_role),
                'first_public_cue_with_new_valid_completion_window':[observer.first[k] for k in sorted(observer.first)]})
    finally:
        runner.snapshot=original_snapshot
    result={'fixed_test_tour_replays':len(rows),'job_id':os.environ.get('SLURM_JOB_ID'),'counts':dict(total),'by_role':dict(roles),
        'cases':cases,'new_policy_rollouts_or_search':False,
        'meaning':'A future-valid completion can remeasure an already confirmed segment. New-opportunity labels subtract only confirmations already available by dispatch.',
        'limitations':'Repeated candidate instances and shared worlds; valid new confirmation still does not establish positive net inspection value.'}
    with args.output.open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print('C1B_QUALITY_DETAILS_PASSED',len(rows),dict(total),flush=True)


if __name__=='__main__':
    main()
