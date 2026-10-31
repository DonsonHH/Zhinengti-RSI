"""Post-run raw-record, public-ranking, witness and hierarchical E1 audit."""
import argparse
import gzip
import json
import math
from pathlib import Path
import statistics

import numpy as np

from patrol_rsi.e_stage import verify_stage_identity
from patrol_rsi.limited_prediction import read,write,sha,need

METHODS=('Tour','human_professional','unconditional_PPO','evidence_gate_PPO','evidence_gate_urgency')


def hierarchical(rows):
    seeds=[]
    for seed in sorted({v['specification']['seed'] for v in rows}):
        process={}
        for row in rows:
            if row['specification']['seed']==seed:process.setdefault(row['process_fingerprint'],[]).append(row)
        need(len(process)==9,'Nine physical processes per seed')
        misses=[];wait=[]
        for pair in process.values():
            need(len(pair)==2 and {v['specification']['noise_c'] for v in pair}=={.2,1.},'Paired observation noises')
            misses.append(sum(v['metrics']['missed_segments'] for v in pair)/2)
            values=[v['metrics']['consequence_weighted_censored_delay_s'] for v in pair]
            need((values[0] is None)==(values[1] is None),'Same physical event eligibility')
            if values[0] is not None:wait.append(sum(values)/2)
        own=[v for v in rows if v['specification']['seed']==seed]
        seeds.append({'seed':seed,'macro_missed_count':sum(misses)/9,'macro_wait_s':sum(wait)/len(wait),
            'missed_noise_repeated_sum':sum(v['metrics']['missed_segments'] for v in own)})
    return {'seeds':seeds,'macro_missed_count':statistics.mean(v['macro_missed_count'] for v in seeds),
        'macro_wait_s':statistics.mean(v['macro_wait_s'] for v in seeds),'conditions':len(rows),
        'missed_noise_repeated_sum':sum(v['metrics']['missed_segments'] for v in rows)}


def expected_gate(row,thresholds):
    need(row['threshold']==thresholds[row['signal_name']],'Fixed profession threshold never changed')
    return bool(row['task']!=row['baseline_next'] and row['available'] and math.isfinite(row['signal']) and row['signal']>=row['threshold']
        and not row['already_alarm'] and row['lead_s']>0 and row['signal']*row['weight_scaled']*row['lead_s']>.05*row['shift_s'])


def event_delta(base,child):
    a=base['anomaly_intervals'];b=child['anomaly_intervals'];need(len(a)==len(b),'Same retained physical events')
    gains=sum(not x['detected'] and y['detected'] for x,y in zip(a,b));losses=sum(x['detected'] and not y['detected'] for x,y in zip(a,b))
    missed=sum(not v['detected'] for v in b)-sum(not v['detected'] for v in a)
    def delay(events):
        weights=sum(v['weight'] for v in events)
        return sum(v['weight']*v['delay_or_censor_s'] for v in events)/weights if weights else None
    w1,w2=delay(a),delay(b)
    return {'missed_delta':missed,'gained_confirmations':gains,'lost_confirmations':losses,'weighted_wait_delta_s':w2-w1 if w1 is not None else None}


def raw_record(row):
    path=Path(row['record_path']);need(sha(path)==row['record_SHA256'],'Fresh-read original record hash')
    return json.loads(gzip.decompress(path.read_bytes()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();r=a.run_root;s=a.source_root/'research/2026-10-31-m12e-timing-schema'
    config=read(s/'E1-Timing-v1.json');identity=r/'E1-execution-identity.json'
    verify_stage_identity(identity,{'E1':s/'E1-Timing-v1.json','manifest':s/'process-manifest.json','history':r/'historical-identities.json'},scope='E1')
    allrows=[];diagnostics=[];source_fps=set();counts={'records':0,'public_roots':0,'public_candidates':0,'witnesses':0,'public_Top1_predictions':0}
    prefix_checks=0;claimed=set();e1jobids=[]
    for seed in (33000,33001):
        root=r/'E1'/f'seed-{seed}';d=read(root/'complete.json');need(d['complete'] and d['execution_identity_SHA256']==sha(identity),'All actual source-bound development complete')
        need(d['main_episodes']==90 and d['actual_total_episode_attempts']<=126 and len(d['rows'])==d['actual_total_episode_attempts'],'Perseed main/witness accounting')
        claim=read(r/'E1-submission-claims'/f'seed-{seed}'/'receipt.json');e1jobids.append(claim['job_id'])
        need(claim['identity_SHA256']==sha(identity) and claim['job_id']==read(root/'started.json')['job_id'] and claim['job_id'] not in claimed,'Unique actual E1 job/source claim')
        claimed.add(claim['job_id']);actual=Path(claim['actual_batch_path']);need(sha(actual)==claim['actual_batch_SHA256']==sha(r/'m12e_timing.sbatch'),'Actual held Slurm source bytes')
        for row in d['rows']:
            path=Path(row['record_path']);need(sha(path)==row['record_SHA256'] and row['audit']['passed'],'Every actual record unchanged and independent route/event/certificate audit passed')
            raw=raw_record(row);counts['records']+=1
            need(raw['specification']==row['specification'] and raw['metrics']==row['metrics'] and raw['episode_purpose']==row['episode_purpose'],'Metadata equals actual episode')
            intervals=raw['anomaly_intervals'];weight=sum(v['weight'] for v in intervals);miss=sum(not v['detected'] for v in intervals)
            wait=sum(v['weight']*v['delay_or_censor_s'] for v in intervals)/weight if weight else None
            need(miss==row['metrics']['missed_segments'] and (wait is None and row['metrics']['consequence_weighted_censored_delay_s'] is None or wait is not None and math.isclose(wait,row['metrics']['consequence_weighted_censored_delay_s'],rel_tol=1e-12)),'Raw event mean/censor/miss arithmetic')
            need(row['metrics']['tasks_violating_exact_gap']==0 and row['metrics']['final_professional_coverage']==1 and row['usage']['global_optional_checks']<=1,'Common revisit/coverage/quota')
            extras=[v for v in raw['workflow_decisions'] if v['task']!=v['baseline_next']]
            checks=[v for v in row['decision_frames'] if v['decision']=='CHECK'];need(len(extras)==len(checks)==row['usage']['global_optional_checks'],'Actual CHECK frame and extra dispatch count')
            need(sum(v['actor_scoring_calls'] for v in row['decision_frames'])==row['usage']['actual_actor_network_calls_in_selector'],'Actual scorer call accounting')
            ppo=[v for v in raw['workflow_decisions'] if v['tool']=='ppo']
            need(len(ppo)==row['usage']['actual_PPO_dispatches'] and sum(v['legal_candidates']>1 for v in ppo)==row['usage']['PPO_multichoice_dispatches'],'Actual PPO authority and masked multiple-candidate count')
            first={v['agent']:v for v in row['first_root_actor_scores']}
            if row['method']=='unconditional_PPO':
                for action in ppo:
                    if action['time_s']!=5:continue
                    old=first[action['agent']];need(action['task']==old['ids'][int(np.argmax(old['scores']))],'First5s actual PPO argmax, not a WAIT decision')
            allrows.append(row)
        for condition in range(18):
            detail=read(root/'diagnostics'/f'condition-{condition:02d}.json');diagnostics.append(detail)
            sealed=root/'public'/f'condition-{condition:02d}--decisions-sealed.json';pub=read(sealed)
            need(sha(sealed)==detail['public_receipt_before_labels_SHA256'] and pub['before_offline_oracle_evaluation'],'Unedited public decisions before offline oracle')
            physical=detail['process_fingerprint'];source_fps.add(physical);spec=detail['specification']
            cohort=[v for v in d['rows'] if v['episode_purpose']=='main' and v['specification']==spec]
            need(len(cohort)==5 and {v['method'] for v in cohort}==set(METHODS),'Exactly five primary methods eachcondition')
            need(len({v['public_telemetry_sequence_SHA256'] for v in cohort})==len({v['process_fingerprint'] for v in cohort})==1,'Matched physical process and observation noise')
            raw_cohort={v['record_path']:raw_record(v) for v in cohort}
            base=next(raw_cohort[v['record_path']] for v in cohort if v['method']=='Tour')
            with np.load(root/'public'/f'condition-{condition:02d}--geometry.npz') as f:geom={k:f[k] for k in f.files}
            candidates=pub['candidate_public'];need(len(candidates)==len(geom['task'])==detail['legal_optional_candidates'],'All public legal candidates accounted')
            for i,v in enumerate(candidates):
                x=v['public'];need(v['eligible']==expected_gate(x,config['gate']['signals']),'Independent public evidence gate predicate')
                need(x['agent']==geom['agent'][i] and x['task']==geom['task'][i] and math.isclose(x['shift_s'],geom['shift_s'][i],abs_tol=1e-9) and math.isclose(x['lead_s'],geom['target_lead_s'][i],abs_tol=1e-9),'Public current completion/lead/shift matches independent geometry')
            trace=root/'public'/f'condition-{condition:02d}.jsonl.gz';need(sha(trace)==pub['public_trace_SHA256'],'Captured full public source unchanged')
            top1={m:{} for m in ('evidence_gate_PPO','evidence_gate_urgency')}
            with gzip.open(trace,'rt',encoding='utf-8') as f:
                for ordinal,line in enumerate(f):
                    packet=json.loads(line);now=packet['public']['time_s'];need(now==(ordinal+1)*5,'Every chronological public root')
                    ids=np.flatnonzero(geom['dispatch_time_s']==now);shift=ordinal%18;agents=list(range(18))[shift:]+list(range(18))[:shift]
                    for method in top1:
                        expected=None
                        for agent in agents:
                            pool=[int(i) for i in ids if candidates[int(i)]['eligible'] and int(geom['agent'][i])==agent]
                            if not pool:continue
                            def score(i):return candidates[i]['actor_score'] if method=='evidence_gate_PPO' else candidates[i]['public']['age_ratio']+candidates[i]['public']['eta_ratio']
                            winner=min(pool,key=lambda i:(-score(i),i));expected=(agent,int(geom['task'][winner]));break
                        reported=packet['hypothetical_public_Top1'][method]
                        need((reported is None and expected is None) or (reported is not None and expected==(reported['agent'],reported['task'])),'Independent masked ranking and common rotating first agent')
                        if reported:top1[method][now]=expected;counts['public_Top1_predictions']+=1
                    counts['public_roots']+=1
            counts['public_candidates']+=len(candidates)
            for row in cohort:
                if row['method'] not in top1:continue
                raw=raw_cohort[row['record_path']];extra=next((v for v in raw['workflow_decisions'] if v['task']!=v['baseline_next']),None)
                if extra:
                    when=extra['time_s'];need(top1[row['method']].get(when)==(extra['agent'],extra['task']),'Actual gated CHECK equals pre-sealed public ranking')
                    need([v for v in raw['action_log'] if v['time_s']<when]==[v for v in base['action_log'] if v['time_s']<when],'Actual quota preservation follows identical protected Tour prefix')
                    prefix_checks+=1
            verified_public=False
            for witness in detail['actual_witnesses']:
                raw=raw_record(witness);delta=event_delta(base,raw);counts['witnesses']+=1
                for key in ('missed_delta','gained_confirmations','lost_confirmations'):need(delta[key]==witness['actual'][key],'Independent complete event gains and losses')
                need(delta['weighted_wait_delta_s'] is None and witness['actual']['weighted_wait_delta_s'] is None or delta['weighted_wait_delta_s'] is not None and math.isclose(delta['weighted_wait_delta_s'],witness['actual']['weighted_wait_delta_s'],abs_tol=1e-9),'Independent witness weighted waiting')
                forced=witness['forced_current_legal_check'];extra=[v for v in raw['workflow_decisions'] if v['task']!=v['baseline_next']]
                need(len(extra)==1 and all(extra[0][k]==forced[k] for k in ('agent','task')) and extra[0]['time_s']==forced['dispatch_time_s'],'Witness executes exactly the preselected legal check')
                need([v for v in raw['action_log'] if v['time_s']<forced['dispatch_time_s']]==[v for v in base['action_log'] if v['time_s']<forced['dispatch_time_s']],'Offline witness shares Tour prefix')
                index=witness['geometry_index'];need((forced['agent'],forced['task'],forced['dispatch_time_s'])==(int(geom['agent'][index]),int(geom['task'][index]),float(geom['dispatch_time_s'][index])),'Exact preselected witness identity')
                net=-(delta['weighted_wait_delta_s'] or 0)-.05*float(geom['shift_s'][index]);useful=delta['lost_confirmations']==0 and delta['missed_delta']<=0 and (delta['missed_delta']<0 or net>1e-9)
                need(bool(useful)==witness['actual']['net_beneficial'] and math.isclose(net,witness['actual']['net_delay_gain_s'],abs_tol=1e-9),'Independent actual witness net utility, not analytic optimism')
                if 'earliest_public_Top1' in witness['tags']:
                    need(any(v.get(forced['dispatch_time_s'])==(forced['agent'],forced['task']) for v in top1.values()),'Useful public witness was selected by a fixed public ranker before labels')
                    verified_public|=bool(useful)
            need(verified_public==detail['public_Top1_actual_beneficial'],'Actual public opportunity flag independently reconstructed')
            print('M12E_AUDIT_CONDITION',seed,condition,flush=True)
    main=[v for v in allrows if v['episode_purpose']=='main'];need(len(main)==180 and len(source_fps)==18 and counts['records']<=252,'Exact phase scope and total charged budget')
    stats={method:hierarchical([v for v in main if v['method']==method]) for method in METHODS};witness_processes={}
    for seed in (33000,33001):
        grouped={}
        for v in diagnostics:
            if v['specification']['seed']==seed:grouped.setdefault(v['process_fingerprint'],[]).append(v)
        useful=[fp for fp,pair in grouped.items() if len(pair)==2 and all(v['public_Top1_actual_beneficial'] for v in pair)]
        witness_processes[str(seed)]={'count':len(useful),'numeric_processes':useful,'both_noise_versions_required':True}
    comparators={};candidate=stats['evidence_gate_PPO']
    for method in ('Tour','human_professional','evidence_gate_urgency'):
        base=stats[method];old={v['seed']:v for v in base['seeds']};perseed=all(v['missed_noise_repeated_sum']<=old[v['seed']]['missed_noise_repeated_sum'] for v in candidate['seeds'])
        miss_delta=candidate['missed_noise_repeated_sum']-base['missed_noise_repeated_sum'];gain=(base['macro_wait_s']-candidate['macro_wait_s'])/base['macro_wait_s']
        comparators[method]={'no_seed_miss_increase':perseed,'missed_noise_repeated_sum_delta':miss_delta,'macro_missed_delta_exact':miss_delta/36,
            'macro_wait_gain_s':base['macro_wait_s']-candidate['macro_wait_s'],'relative_macro_wait_gain':gain,'passed':bool(perseed and (miss_delta<0 or miss_delta==0 and gain>=.05))}
    waited={}
    for seed in (33000,33001):
        own=[v for v in main if v['method']=='evidence_gate_PPO' and v['specification']['seed']==seed]
        waited[str(seed)]=any(any(f['decision']=='CHECK' and f['time_s']>5 and any(q['decision']=='WAIT_CONTINUE_TOUR' and q['time_s']<f['time_s'] for q in v['decision_frames']) for f in v['decision_frames']) for v in own)
    result={'passed':True,'audit_counts':counts,'actual_main_episodes':180,'actual_witness_episodes':counts['witnesses'],'actual_total_episodes':counts['records'],
        'independently_reconstructed_main_statistics':stats,'public_witness_processes':witness_processes,'public_quota_preservation_then_CHECK':waited,
        'E1_opportunity_gate':all(v['count']>=2 for v in witness_processes.values()),'E1_timing_comparators':comparators,
        'E1_timing_gate':all(waited.values()) and all(v['passed'] for v in comparators.values()),'verified_gated_action_prefixes':prefix_checks,
        'source_identity_SHA256':sha(identity),'actual_E1_job_ids':e1jobids,'new_model_calls':0,'new_environment_episodes':0,'old_final_raw_access':False,
        'scope':'Post-publication source/record hashes, event arithmetic, independent gate/ranker/Tour-prefix and witness checks; per-episode independent route/event/full-return audits already recorded. No second neural execution.'}
    write(r/'E1-independent-audit.json',result);print('M12E_E1_AUDIT_COMPLETE',result['E1_opportunity_gate'],result['E1_timing_gate'],counts,flush=True)


if __name__=='__main__':main()
