"""Two development seeds: seal public Top-1 choices, then evaluate Tour branches."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import gzip
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from patrol_rsi.active_inspection import graph_for
from patrol_rsi.b2_actions import RANKERS,public_ranks,best_index,branch_schedule,evaluate_schedule,delta_outcome,verify_geometry_subset
from patrol_rsi.information_audit import current_legal_geometry,pack_geometry,analytic_continuations,reference_signals
from patrol_rsi.inspection_metrics import task_labels,anomaly_intervals
from patrol_rsi.limited_prediction import need,read,sha,write


def main():
    parser=argparse.ArgumentParser()
    for key in ('protocol','review','old-manifest','source-root','source-inventory','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--seed-index',type=int,required=True);args=parser.parse_args();os.umask(0o077)
    protocol=read(args.protocol);review=read(args.review)
    need(review['approved'] and review['protocol_SHA256']==sha(args.protocol),'Reviewed independentdevelopmentdiagnostic')
    need(args.seed_index in (0,1),'Exactly two fixed developmentseeds')
    seed=protocol['action']['seed_groups'][args.seed_index];args.output.mkdir(exist_ok=False)
    invpath=args.source_inventory;need(sha(invpath)=='ef404d3ecadb8b6e6b503e6bfac11551a32e9a10d66c117c74648f25acae8065','Originalsourceauthority')
    inventory={v['path']:v for v in read(invpath)['files']}
    def check(path):
        row=inventory[str(path)];need(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Everyused original byte')
    group=args.source_root/'development'/f'seed-{seed}';check(group/'geometry.npz')
    with np.load(group/'geometry.npz') as f:geometry={k:f[k] for k in f.files}
    seen=set();conditions=[];start=time.perf_counter();sourcefiles=[]
    for item in read(args.old_manifest)['manifest']:
        if item['split']!='development' or item['group_seed']!=seed:continue
        folder=group/item['id'];check(folder/'source-identity-not-features.json');identity=read(folder/'source-identity-not-features.json')
        key=identity['numeric_process_fingerprint'],item['specification']['noise_c']
        if key in seen:continue
        seen.add(key);conditions.append((item,folder,identity))
    need(len(conditions)==18,'Nine uniqueprocesses eachtwonoises')
    write(args.output/'condition-scope.json',{'seed':seed,'phase':'development','conditions':[v[0]['id'] for v in conditions],
        'protocol_SHA256':sha(args.protocol),'before_public_ranking':True})
    reports=[];selected_checks=0
    for item,folder,identity in conditions:
        t0=time.perf_counter()
        for name in ('public-states.jsonl.gz','tour-record.json.gz'):
            check(folder/name);sourcefiles.append({'path':str(folder/name),'sha256':sha(folder/name)})
        record=json.loads(gzip.decompress((folder/'tour-record.json.gz').read_bytes()))
        need(record['optional_checks']==record['exact_gap_violations']==0,'Unspent originalprotectedTour')
        scores={ranker:np.full(len(geometry['task']),-np.inf) for ranker in RANKERS}
        root_best=defaultdict(list);selections={};geometry_rebuilt=0;first_tasks=None
        observed=Counter();candidate_professions=Counter()
        with gzip.open(folder/'public-states.jsonl.gz','rt',encoding='utf-8') as stream:
            for ordinal,line in enumerate(stream):
                bundle=json.loads(line);now=bundle['time_s'];need(now==(ordinal+1)*5,'All359originalpublicroots')
                if first_tasks is None:first_tasks=bundle['tasks']
                observed['inflight_observation_opportunities']+=sum(v['busy_target'] is not None for v in bundle['agents'])
                observed['free_agent_frames']+=sum(v['busy_target'] is None for v in bundle['agents'])
                ids=np.flatnonzero(geometry['dispatch_time_s']==now)
                rebuilt=pack_geometry(current_legal_geometry(bundle));need(len(ids)==len(rebuilt['task']),'Actualcurrenthardlegalcount')
                if not len(ids):continue
                local={k:v[ids] for k,v in geometry.items()}
                verify_geometry_subset(local,rebuilt)
                geometry_rebuilt+=len(ids);values=public_ranks(bundle,local)
                task_kind=np.where(local['task']<128,0,np.where(local['task']<144,1,np.where(local['task']<160,2,3)))
                for v in task_kind:candidate_professions[('thermal','electrical','cooling','control')[v]]+=1
                for ranker,values_local in values.items():
                    scores[ranker][ids]=values_local;best=best_index(values_local,local)
                    if best is None:continue
                    original_index=int(ids[best]);root_best[ranker].append(original_index)
                    if ranker in selections:continue
                    task=int(local['task'][best]);agent=int(local['agent'][best]);a=next(v for v in bundle['agents'] if v['agent']==agent)
                    need(a['busy_target'] is None and task!=a['tour'][a['cursor']] and a['detours_used']<2,'NewoptionalnotinflightorTour')
                    row={k:(local[k][best].tolist() if isinstance(local[k][best],np.ndarray) else local[k][best].item()) for k in local}
                    row.update(geometry_index=original_index,score=float(values_local[best]),profession=bundle['tasks'][task]['kind'],
                        role=bundle['tasks'][task]['required_role'],quota_before={'global_extra_used':0,'per_agent_extra_used':a['detours_used'],'per_owned_round_limit':2},
                        decision='execute_currentTop1once_thenTour',future_readings_used=False)
                    selections[ranker]=row
        need(ordinal==358 and geometry_rebuilt==len(geometry['task']),'Allcurrentgeometry accounted')
        public_path=args.output/(item['id']+'--public-selections.json')
        write(public_path,{'condition':item['id'],'scope':'Currentpubliconly decisionsbefore anyevaluator data read','selections':selections,
            'root_Top1_counts':{k:len(v) for k,v in root_best.items()},'candidate_professions':dict(candidate_professions),
            'observation_counts':dict(observed),'source_public_SHA256':sha(folder/'public-states.jsonl.gz'),'protocol_SHA256':sha(args.protocol)})
        # Only after saving the public receipt do we read evaluator-side arrays.
        process=group/identity['source_process_path'];check(process);check(folder/'labels-only.npz')
        with np.load(process) as f:arrays={k:f[k] for k in f.files}
        with np.load(folder/'labels-only.npz') as f:B=f['B_reference']
        graph=graph_for(128,'open');asset=arrays['asset_weights'];weights=np.r_[asset,np.tile(asset.reshape(16,8).mean(1),3)]
        # Graph task order is thermal128,electrical16,cooling16,control16.
        need([v.kind for v in graph.tasks]==['thermal']*128+['electrical']*16+['cooling']*16+['control']*16,'Exacttaskorder')
        env=SimpleNamespace(trace=SimpleNamespace(arrays=arrays),config=graph.config,tasks=graph.tasks)
        physical,excess=task_labels(env);intervals=anomaly_intervals(physical,excess,graph.tasks,weights)
        original_schedule=[(float(v['time_s']),int(v['task'])) for v in record['inspection_events']]
        base=evaluate_schedule(original_schedule,intervals,B)
        for interval,when in zip(intervals,base['first_confirmation_s']):
            interval.update(detected=when is not None,first_confirmation_s=when,
                delay_or_censor_s=(when if when is not None else interval['end_s'])-interval['onset_s'])
        analytic=analytic_continuations(record|{'anomaly_intervals':intervals},geometry,reference_signals(arrays),measurement_positive=B)
        delta=analytic['missed_delta'];wait=analytic['weighted_wait_delta_s'];shift=geometry['shift_s']
        net=-np.nan_to_num(wait,nan=0)-.05*shift
        useful=(analytic['lost_confirmations']==0)&(delta<=0)&((delta<0)|(net>1e-9))
        harmful=(analytic['lost_confirmations']>0)|(delta>0)|((delta==0)&(net< -1e-9))
        oracle=min(range(len(delta)),key=lambda i:(delta[i],-net[i],geometry['dispatch_time_s'][i],geometry['agent'][i],geometry['task'][i]))
        choices={}
        for ranker in RANKERS:
            row=selections.get(ranker);indices=np.asarray(root_best[ranker],dtype=int)
            if row is None:out={'selected':False,'missed_delta':0,'weighted_wait_delta_s':0. if intervals else None,'net_delay_gain_s':0.,'gained_confirmations':0,'lost_confirmations':0,'net_beneficial':False,'net_harmful':False}
            else:
                i=row['geometry_index'];schedule=branch_schedule(record['inspection_events'],row['agent'],row['task'],row['dispatch_time_s'],row['finish_s'],row['shift_s'])
                branch=evaluate_schedule(schedule,intervals,B);out=delta_outcome(base,branch,row['shift_s'])
                need(out['missed_delta']==int(delta[i]) and out['lost_confirmations']==int(analytic['lost_confirmations'][i]) and
                    out['gained_confirmations']==int(analytic['gained_confirmations'][i]),'Independentselected full-episode miss/gain/loss')
                if intervals:need(np.isclose(out['weighted_wait_delta_s'],wait[i],atol=1e-8),'Independentselected weightedfullwait')
                need([v if v is not None else -1 for v in branch['first_confirmation_s']]==analytic['first_confirmation_s'][i].tolist(),'Independentallselectedeventfirsttimes')
                for j,task in enumerate(first_tasks):
                    times=[t for t,k in schedule if k==j];gaps=np.diff([0.]+times+[1800.])
                    need(gaps.max()<=task['deadline_s']+1e-9,'Selectedbranch exactinitial/terminalrevisit')
                out.update(selected=True,action=row,branch_missed=branch['missed'],branch_weighted_wait_s=branch['weighted_wait_s'],independent_schedule_verified=True)
                selected_checks+=1
            out['within_root_Top1_diagnostic']={'roots_with_accepted_candidate':len(indices),'net_beneficial_roots':int(useful[indices].sum()),
                'net_harmful_roots':int(harmful[indices].sum()),'not_executed_jointly':True}
            out['oracle_ceiling_regret']={'miss_count_regret':out['missed_delta']-int(delta[oracle]),
                'net_delay_regret_s_when_equal_misses':float(net[oracle]-out['net_delay_gain_s']) if out['missed_delta']==delta[oracle] else None,
                'scope':'Fullremainingepisode offline bestCHECK;futureinformation ceiling,not an executable publicpolicy.'}
            choices[ranker]=out
        all_distributions={}
        for name,value in {'completion_cost_s':geometry['finish_s']-geometry['dispatch_time_s'],'return_slack_s':geometry['return_slack_s'],
                'tour_target_arrival_s':geometry['tour_target_finish_s'],'other_task_shift_s':geometry['shift_s'],
                'total_other_task_delay_exposure_s':geometry['full_exposure_s'].sum(axis=1)}.items():
            all_distributions[name]={'n':len(value),'minimum':float(value.min()),'maximum':float(value.max()),
                'quantiles_0_25_50_75_100':np.quantile(value,[0,.25,.5,.75,1]).tolist()}
        reports.append({'condition':item['id'],'seed':seed,'numeric_process_fingerprint':identity['numeric_process_fingerprint'],'noise_c':item['specification']['noise_c'],
            'public_receipt_SHA256_before_labels':sha(public_path),'baseline':{k:v for k,v in base.items() if k!='first_confirmation_s'},
            'candidates':len(delta),'net_beneficial_candidates':int(useful.sum()),'net_harmful_candidates':int(harmful.sum()),
            'all_candidate_resource_certificate_distributions':all_distributions,
            'current_candidates_same30sroots':int((geometry['dispatch_time_s']%30==0).sum()),'rankers':choices,
            'oracle_ceiling_not_deployable':{'geometry_index':int(oracle),'missed_delta':int(delta[oracle]),'weighted_wait_delta_s':float(wait[oracle]) if intervals else None,
                'net_delay_gain_s':float(net[oracle]),'net_beneficial':bool(useful[oracle]),'exists_net_beneficial':bool(useful.any())},
            'wall_s':time.perf_counter()-t0})
        print('B2_ACTION_CONDITION_DONE',seed,item['id'],len(delta),flush=True)
    write(args.output/'action-diagnosis.json',{'protocol_SHA256':sha(args.protocol),'seed':seed,'conditions':len(reports),'rows':reports,
        'independent_selected_schedule_checks':selected_checks,'sourcefiles':sourcefiles,'new_physical_processes':0,'PPO_CVAE_training':0,'LLM_calls':0,
        'phases_opened':['development'],'final_access':False,'wall_s':time.perf_counter()-start,'job_id':os.environ.get('SLURM_JOB_ID'),
        'scope':'Descriptive existing2developmentseeds;one budgeted publicTop1 CHECKthenTour perranker/condition. Oraclecounterfactuals offlineonly;no newformaltestorRSIclaim.'})


if __name__=='__main__':main()
