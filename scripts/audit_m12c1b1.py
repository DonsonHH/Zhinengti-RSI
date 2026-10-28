"""Actual B1 replay and independent public model / check-vs-wait verification."""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path

import numpy as np
import torch

import patrol_rsi.timing_run as runner
import patrol_rsi.inspection_timing as timing
from patrol_rsi.m12 import actor_scorer, load_actors, sha
from patrol_rsi.m11 import frozen_predictor, seed_digest
from patrol_rsi.m12c1b1 import specs, configs, frozen_models, GATES
from patrol_rsi.value_learning import scorer
from audit_m111 import equivalent, distance, independent_certificates, independent_power_events
from audit_m11 import independent_schedule, independent_events
from audit_m12 import independent_public_and_quota
from audit_m12c1 import independent_snapshot
from audit_m12c1b import positive_at_completion, independent_labels, aggregate, need

ORIG_SNAPSHOT,ORIG_PACKET,ORIG_OPPORTUNITIES=runner.snapshot,runner.public_packet,timing.opportunities
ORIG_CHOOSE=timing.TimingPolicy.choose
PACKETS=FRAMES=CANDIDATES=0
CURRENT_USED=False
FROZEN={}
LAST_PACKET=None


def independent_packet(env):
    obs=env.state.observer;temperature=env.current_history if env.mode=='history' else obs.predicted_temperature
    tasks=[]
    for j,task in enumerate(env.tasks):
        rack=task.index if task.kind=='thermal' else task.index*8
        tasks.append(dict(kind=task.kind,service_s=float(task.service_s),deadline_s=float(env.deadlines[j]),
            weight_scaled=float(env.asset_weights[j]/2),last_check_s=float(env.last_inspection[j]),
            already_alarm=bool(env.measured_alert.get(j,False)),temperature_c=float(temperature[rack]),
            last_temperature_c=float(obs.last_temperature[rack]),temperature_age_s=float(env.clock-obs.temperature_stamp[rack]) if obs.temperature_stamp[rack]>=0 else float(env.clock),
            temperature_stamp_s=float(obs.temperature_stamp[rack]),velocity_c_s=float(env.state.velocity[rack]),
            nominal_temperature_c=float(obs.predicted_temperature[rack]),
            contact_c=float(obs.last_contact[task.index]) if task.kind=='electrical' else None,
            flow_kg_s=float(obs.last_flow[task.index]) if task.kind=='cooling' else None,
            sensor_verified=bool(obs.sensor_bias_verified[task.index]) if task.kind=='control' else False))
    agents=[]
    for i,a in enumerate(env.agents):
        tour=list(a['tour'])
        duration=lambda position,j: math.ceil((distance(env.graph,tuple(position),env.tasks[j].position_m)+env.tasks[j].service_s)/5-1e-10)*5.
        edges={str(j):{str(k):duration(env.tasks[j].position_m,k) for k in tour} for j in tour}
        remaining=a.get('remaining_travel_s',0)+a.get('remaining_service_s',0)
        agents.append(dict(agent=i,tour=tour,cursor=int(env.protected_cursor[i]),busy_target=a['task'],
            busy_finish_s=float(env.clock+math.ceil(remaining/5-1e-10)*5) if a['task'] is not None else None,
            detours_used=int(env.detours_used[i]),baseline_completed=int(env.baseline_completed[i]),
            first_costs_s={str(j):duration(a['position'],j) for j in tour},costs_s=edges,
            period_s=sum(edges[str(a)][str(b)] for a,b in zip(tour[-1:]+tour[:-1],tour))))
    return dict(time_s=float(env.clock),tasks=tasks,agents=agents)


def checked_packet(env):
    global PACKETS,LAST_PACKET
    value=ORIG_PACKET(env);equivalent(value,independent_packet(env),'Public planner packet including independent graph costs')
    LAST_PACKET=value;PACKETS+=1;return value


def checked_snapshot(env,score):
    actual=ORIG_SNAPSHOT(env,score);expected=independent_snapshot(env,score)
    equivalent(actual,expected,'Independent current full legal mask')
    return actual


def signal(row,dt):
    if row['kind']=='thermal':
        extrapolated=row['last_temperature_c']+row['velocity_c_s']*min(120.,row['temperature_age_s']+dt)
        temperature=max(row['temperature_c'],max(35.,min(110.,extrapolated)))
        return max(temperature-65.,0.)/20
    if row['kind']=='electrical':
        return max(row['contact_c']-60.,0.)/30
    if row['kind']=='cooling':
        return max(0.,1.-row['flow_kg_s']/33.6)
    return float(row['sensor_verified'])


def independent_opportunities(packet,horizon):
    tasks=packet['tasks'];root=packet['time_s'];found=[]
    for agent in packet['agents']:
        tour=agent['tour'];edge=agent['costs_s'];cursor=agent['cursor'];cost=agent['first_costs_s']
        last={j:max(0.,tasks[j]['last_check_s']) for j in tour};count=agent['baseline_completed'];extras=agent['detours_used']
        t=root
        if agent['busy_target'] is not None:
            j=agent['busy_target'];need(j==tour[cursor],'WAIT baseline pending work')
            t=agent['busy_finish_s'];last[j]=t;cost=edge[str(j)];cursor=(cursor+1)%len(tour);count+=1
            if count==len(tour):
                count=extras=0
        while t<=min(root+horizon,1795.):
            baseline=tour[cursor];base_finish=t+cost[str(baseline)]
            if base_finish>1800:
                break
            def certificate(target):
                due={j:last[j]+tasks[j]['deadline_s'] for j in tour};when=t+cost[str(target)]
                gaps=[due[target]-when]
                if gaps[0]<-1e-9:
                    return None
                due[target]=when+tasks[target]['deadline_s']
                c=(cursor+1)%len(tour) if target==baseline else cursor
                previous=target
                for j in tour[c:]+tour[:c]:
                    when+=edge[str(previous)][str(j)];gaps.append(due[j]-when)
                    if gaps[-1]<-1e-9:
                        return None
                    due[j]=when+tasks[j]['deadline_s'];previous=j
                return min(gaps)
            base_slack=certificate(baseline);need(base_slack is not None,'Independent projected whole-return plan')
            baseline_finishes={};when=t;prev=None
            for j in tour[cursor:]+tour[:cursor]:
                when+=cost[str(j)] if prev is None else edge[str(prev)][str(j)]
                baseline_finishes[j]=min(1800.,when);prev=j
            for j in tour:
                row=tasks[j];finish=t+cost[str(j)]
                if j==baseline or extras>=2 or finish>1800 or row['already_alarm']:
                    continue
                if (last[j]>0 or row['last_check_s']>=0) and t-last[j]<60:
                    continue
                slack=certificate(j)
                if slack is None:
                    continue
                severity=signal(row,finish-root)
                lead=max(baseline_finishes[j]-finish,0.)
                utility=row['weight_scaled']*severity*lead/300-.05*max(cost[str(j)]-cost[str(baseline)],0)/300-.05*max(base_slack-slack,0)/300
                if utility>0:
                    found.append(dict(agent=agent['agent'],task=j,dispatch_time_s=float(t),finish_s=float(finish),
                        signal=float(severity),surrogate_value=float(utility),baseline_target_finish_s=float(baseline_finishes[j]),
                        return_slack_s=float(slack),discounted_value=float(utility*math.exp(-(t-root)/300))))
            last[baseline]=base_finish;cost=edge[str(baseline)];cursor=(cursor+1)%len(tour);count+=1
            if count==len(tour):
                count=extras=0
            t=base_finish
    return found


def checked_opportunities(packet,horizon):
    global CANDIDATES
    actual=ORIG_OPPORTUNITIES(packet,horizon);expected=independent_opportunities(packet,horizon)
    equivalent(actual,expected,'Every projected positive utility and NOW/WAIT schedule')
    CANDIDATES+=len(actual);return actual


def checked_choose(self,groups,packet,used):
    global CURRENT_USED,FRAMES
    need(used==CURRENT_USED,'Global optional budget bookkeeping')
    result=ORIG_CHOOSE(self,groups,packet,used);choices,logs,consumed,diagnostic=result
    expected={g['agent']:g['baseline_next'] if g['candidates'] else None for g in groups}
    method=self.config['method'];gate=self.config.get('not_before_s',0);eligible=[]
    enabled=not used and method!='tour' and gate!='off' and packet['time_s']>=gate
    if enabled and method in ('evidence','lookahead'):
        candidates=independent_opportunities(packet,120 if method=='lookahead' else 0)
        current=[r for r in candidates if r['dispatch_time_s']==packet['time_s']];future=[r for r in candidates if r['dispatch_time_s']>packet['time_s']]
        best=max(current,key=lambda r:(r['surrogate_value'],-r['agent'],-r['task']),default=None)
        later=max(future,key=lambda r:(r['discounted_value'],-r['dispatch_time_s'],-r['agent'],-r['task']),default=None)
        value=best['surrogate_value'] if best else 0.;wait=later['discounted_value'] if later else 0.
        need(abs(diagnostic['now_value']-value)<1e-9 and abs(diagnostic['wait_value']-wait)<1e-9,'Explicit competing now/wait values')
        if best and value>max(0.,wait)+1e-9:
            expected[best['agent']]=best['task'];eligible=[best]
    elif enabled:
        for g in groups:
            base=next((r for r in g['candidates'] if r['task']==g['baseline_next']),None)
            if base is None:
                continue
            rows=[r for r in g['candidates'] if r['is_optional']]
            if method=='cue':
                for r in rows:
                    if r['cue_strength']>=.05 and not r['known_measured_alarm']:
                        eligible.append(((.01+r['cue_strength'])*r['asset_weight_scaled']/(.03+r['eta_over_deadline']),g['agent'],r['task']))
        if method=='net' and self.config.get('threshold',0.)!='off':
            from audit_m12c1b import independent_features,independent_score
            seed=self.config['seed'];pack=FROZEN[seed]['pack'];rows=independent_features(groups)
            values=independent_score(pack,np.stack([r[2] for r in rows])) if rows else []
            eligible=[(float(v),g['agent'],r['task']) for (g,r,_),v in zip(rows,values) if v>float(self.config.get('threshold',0.))]
        if eligible:
            selected=max(eligible,key=lambda r:r[0]);expected[selected[1]]=selected[2]
    need(choices==expected and consumed==bool(eligible),'Independent bounded stop/wait decision')
    need(sum(v!=g['baseline_next'] for g in groups for i,v in choices.items() if i==g['agent'] and v is not None)<=1,'At most one team action')
    CURRENT_USED=used or consumed;FRAMES+=1;return result


def independent_atlas_check(env,groups,expected,buffer):
    # Verify every saved witness count/first/last/cheapest using independent
    # raw reference-noise calculations; no policy has access to this function.
    if not buffer:
        buffer['segments']=[dict(count=0,old=0,new=0,witnesses=[],first_signal={k:None for k in expected['confusion']}) for _ in expected['segments']]
        buffer['confusion']={k:Counter(tp=0,fp=0,fn=0,tn=0) for k in expected['confusion']}
    packet=LAST_PACKET
    raw_gap=1-env.current_packet['public_delivered_it_kw']/np.maximum(env.current_packet['public_requested_it_kw'],1e-12)
    for g in groups:
        for r in g['candidates']:
            if not r['is_optional']:
                continue
            j=r['task'];p=packet['tasks'][j];task=env.tasks[j];finish=r['candidate_finish_s']
            linear=max(35.,min(110.,p['last_temperature_c']+p['velocity_c_s']*min(120.,p['temperature_age_s']+finish-env.clock)))
            values={'old_cue':r['cue_strength'],'new_evidence':signal(p,finish-env.clock),
                'trend':max(linear-65,0)/20 if task.kind=='thermal' else 0.,
                'residual':max(p['last_temperature_c']-p['nominal_temperature_c'],0)/20 if task.kind=='thermal' and p['temperature_age_s']<=120 else 0.,
                'power_gap':float(raw_gap[task.index] if task.kind=='thermal' else raw_gap.reshape(16,8)[task.index].mean())}
            flags={k:v>=.05 and not r['known_measured_alarm'] for k,v in values.items()}
            truth=any(e['interval']['task']==j and e['interval']['onset_s']<finish<=e['interval']['end_s'] for e in expected['segments']) and positive_at_completion(env,j,finish)
            for key,flag in flags.items():
                buffer['confusion'][key]['tp' if truth and flag else 'fp' if flag else 'fn' if truth else 'tn']+=1
            for i,entry in enumerate(expected['segments']):
                v=entry['interval'];finish=r['candidate_finish_s']
                if v['task']==j and v['onset_s']-120<=env.clock<=v['end_s']:
                    for key,flag in flags.items():
                        if flag and buffer['segments'][i]['first_signal'][key] is None:
                            buffer['segments'][i]['first_signal'][key]=env.clock
                if r['task']==v['task'] and v['onset_s']<finish<=v['end_s'] and positive_at_completion(env,r['task'],finish):
                    old=r['cue_strength']>=.05 and not r['known_measured_alarm'];new=signal(p,finish-env.clock)>=.05 and not r['known_measured_alarm']
                    b=buffer['segments'][i];b['count']+=1;b['old']+=int(old);b['new']+=int(new)
                    b['witnesses'].append({'intervention':dict(dispatch_time_s=g['time_s'],agent=g['agent'],task=r['task']),
                        'finish_s':finish,'completion_cost_s':finish-g['time_s'],'minimum_return_slack_s':r['minimum_return_slack_s'],
                        'old_cued':old,'new_cued':new})
    if env.clock%30==0:
        saved=next(v for v in expected['temperature'] if v['time_s']==env.clock)
        rows=[r for r in packet['tasks'] if r['kind']=='thermal']
        obs=env.state.observer;age=np.array([r['temperature_age_s'] for r in rows]);last=np.array([r['last_temperature_c'] for r in rows]);velocity=np.array([r['velocity_c_s'] for r in rows])
        current={'time_s':env.clock,'truth_c':env.trace.arrays['rack_temperature_c'][int(env.clock/5)-1].tolist(),
            'fresh':[r['temperature_stamp_s']==env.clock for r in rows],'hold':last.tolist(),
            'linear':np.clip(last+velocity*np.minimum(age,120),35,110).tolist(),'nominal':obs.predicted_temperature.tolist(),
            'history':env.current_history.tolist(),'age_s':age.tolist(),'velocity_c_s':velocity.tolist()}
        equivalent(saved,current,'Same-history public inputs and evaluator current-temperature arrays')


def main():
    global CURRENT_USED,FROZEN
    p=argparse.ArgumentParser()
    for key in ('results','actors','estimator-weights','estimator-manifest','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--validation',type=Path)
    args=p.parse_args();torch.set_num_threads(1);folder=args.results
    read=lambda p:json.loads(p.read_bytes())
    index=read(folder/'case-index.json');summary=read(folder/'summary.json');lookup={r['case']:r for r in index}
    atlas=read(folder/'atlas-before-branches.json') if summary['stage']!='validation' else {}
    branches=read(folder/'branch-index.json') if atlas else [];by_branch={r['branch_case']:r for r in branches}
    actors,fingerprints=load_actors(args.actors);actor=actor_scorer(actors[0]);predictor=frozen_predictor(args.estimator_weights,args.estimator_manifest)
    models=frozen_models();FROZEN=models;records={};processes=read(folder/'process-index.json');seen=set();dispatches=certs=0
    runner.snapshot,runner.public_packet=checked_snapshot,checked_packet;timing.opportunities=checked_opportunities;timing.TimingPolicy.choose=checked_choose
    try:
        for number,row in enumerate(index):
            path=folder/row['path'];need(sha(path)==row['sha256'] and path.stat().st_size==row['bytes'],'Raw case bytes')
            original=json.loads(gzip.decompress(path.read_bytes()));dep=row['deployment'];CURRENT_USED=False;observer=None;buffer={}
            if row['case'] in atlas:
                expected=atlas[row['case']]
                class Observer:
                    def observe(self,env,groups):
                        independent_atlas_check(env,groups,expected,buffer)
                observer=Observer()
            config=dep['timing_config'];seed=config.get('seed');score=scorer(models[seed]['pack']) if seed is not None else None
            if seed is not None:
                need(dep['model_id']==models[seed]['model_id'] and config['threshold']==models[seed]['threshold'],'Exact old net model and old selected threshold')
            replay,env=runner.run(row['specification'],predictor,actor,dep['public_controller'],score=score,config=config,
                threshold=dep['threshold'],model_id=dep['model_id'],observer=observer,
                intervention=dep['evaluator_intervention'],probe_enabled=dep['probe_enabled'])
            equivalent(replay,original,'Actual check/wait whole-loop replay')
            independent_schedule(original,env);independent_events(original,env);passed,count=independent_certificates(original,env)
            independent_power_events(env,original);independent_public_and_quota(original,env)
            certs+=passed;dispatches+=count;need(original['metrics']['selected_detours']<=1,'One extra check globally')
            if row['case'] in atlas:
                expected_samples={}
                equivalent({k:dict(v) for k,v in buffer['confusion'].items()},expected['confusion'],'All positive and negative cue instances')
                for actual,saved in zip(buffer['segments'],expected['segments']):
                    need((actual['count'],actual['old'],actual['new'])==(saved['legal_witness_instances'],saved['old_cued_witnesses'],saved['new_cued_witnesses']),'All valid witness counts')
                    witnesses=actual['witnesses']
                    identity=lambda w:(w['intervention']['dispatch_time_s'],w['intervention']['agent'])
                    values={'first':min(witnesses,key=identity) if witnesses else None,
                        'last':max(witnesses,key=identity) if witnesses else None,
                        'cheapest':min(witnesses,key=lambda w:(w['completion_cost_s'],*identity(w))) if witnesses else None}
                    for k,v in values.items():
                        equivalent(v,saved[k],'Predeclared sampled witness '+k)
                        if v is not None:
                            key=tuple(v['intervention'].values())
                            expected_samples.setdefault(key,{'intervention':v['intervention'],'sources':[]})['sources'].append({'segment_index':saved['segment_index'],'kind':k})
                    equivalent(actual['first_signal'],saved['event_signal_first_s'],'Event window signal first times')
                equivalent(list(expected_samples.values()),expected['samples'],'Complete source-only branch selection before outcomes')
            if row['case'] in by_branch:
                source=by_branch[row['case']];base=records[source['base_case']]
                equivalent(source['intervention'],dep['evaluator_intervention'],'Exact independent evaluator intervention')
                equivalent(source['labels'],independent_labels(base,original,source['intervention']['task']),'Full-episode target/collateral/net labels')
            info=processes[row['process_key']]
            if row['process_key'] not in seen:
                path=folder/info['path'];need(sha(path)==info['sha256'],'Original physical arrays')
                with np.load(path,allow_pickle=False) as arrays:
                    for k in arrays.files:
                        np.testing.assert_allclose(arrays[k],env.trace.arrays[k],rtol=1e-11,atol=1e-9)
                seen.add(row['process_key'])
            records[row['case']]=original
            if number%16==0:
                print('B1_AUDIT',number+1,'/',len(index),flush=True)
    finally:
        runner.snapshot,runner.public_packet=ORIG_SNAPSHOT,ORIG_PACKET;timing.opportunities=ORIG_OPPORTUNITIES;timing.TimingPolicy.choose=ORIG_CHOOSE
    need(summary['actual_cases']==len(index),'All actual cases')
    if summary['stage']=='validation':
        selected=read(folder/'selection-frozen.json');options=selected['all_options']
        need([v['gate'] for v in options]==list(GATES),'Exactly five frozen gate options')
        for v in options:
            rows=[lookup[k] for k in v['case_ids']]
            equivalent([r['specification'] for r in rows],specs('validation'),'Gate validation split')
            equivalent(v['validation'],aggregate(rows),'Independent gate objective')
        best=min(options,key=lambda v:(v['validation']['missed'],v['validation']['weighted_wait_s'] or 0.,v['validation']['travel_m']))
        need(selected['selected_gate']==best['gate'],'Gate selected on validation only')
        equivalent(selected['all_configs'],configs(best['gate'],models),'Frozen controls and three old net scorers')
    else:
        gate=read(folder/'test-start-gate.json');groups=read(folder/'groups.json')
        if summary['stage']=='formal':
            selected=read(args.validation/'selection-frozen.json')
            need(gate['selection_sha256']==sha(args.validation/'selection-frozen.json') and gate['selection_frozen_at_unix']<gate['started_at_unix'],'Selection hash and time before any test')
            equivalent(gate['frozen_configs'],selected['all_configs'],'All test controls frozen')
        for tag,value in groups.items():
            rows=[lookup[k] for k in value['case_ids']];expected=specs('smoke' if summary['stage']=='smoke' else 'test')
            if tag=='lookahead-nominal':
                expected=[dict(s,estimator='nominal') for s in expected]
            equivalent([r['specification'] for r in rows],expected,'Full test conditions')
            equivalent(value['aggregate'],aggregate(rows),'All group outcome statistics')
        expected_branches={}
        for base,entry in atlas.items():
            for i,sample in enumerate(entry['samples']):
                branch=base.rsplit('--',1)[0]+'--branch-'+str(i);expected_branches[branch]=(base,sample)
        need(set(by_branch)==set(expected_branches),'All and only predeclared branches')
        for key,(base,sample) in expected_branches.items():
            need(by_branch[key]['base_case']==base and all(by_branch[key][k]==v for k,v in sample.items()),'Branch source catalogue')
    for seed,model in actors.items():
        need(seed_digest(model)==fingerprints[str(seed)]['parameters_sha256'],'Old actor unchanged')
    result={'stage':summary['stage'],'all_actual_cases_replayed':len(index),'physical_arrays_verified':len(seen),
        'public_packets_with_independent_graph_costs':PACKETS,'stop_wait_frames_independently_checked':FRAMES,
        'positive_projected_candidate_values_independently_checked':CANDIDATES,'actual_dispatches':dispatches,
        'completed_measurements_and_whole_return_plans':certs,'atlas_baselines':len(atlas),'independent_branches':len(branches),
        'exact_revisit_violations':0,'one_global_check_budget':1,'scope':'Public surrogate check-vs-wait and sampled single-action references, not optimal stopping or global bounds'}
    with args.output.open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print('B1_AUDIT_PASSED',len(index),flush=True)


if __name__=='__main__':
    main()
