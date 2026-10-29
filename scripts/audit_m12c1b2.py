"""Full B2 replay: independent masks, information controls and stop/wait values."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

import patrol_rsi.robust_timing_run as runner
import patrol_rsi.robust_timing as policy
import patrol_rsi.inspection_timing as old_timing
from patrol_rsi.m12 import actor_scorer, load_actors, sha
from patrol_rsi.m11 import frozen_predictor, seed_digest
from patrol_rsi.m12c1b2 import configs, specs, name
from audit_m111 import equivalent, independent_certificates, independent_power_events
from audit_m11 import independent_schedule, independent_events
from audit_m12 import independent_public_and_quota
import audit_m12c1 as masks
from audit_m12c1b import aggregate, need
from audit_m12c1b1 import independent_packet, independent_opportunities, signal

ORIG_SNAPSHOT, ORIG_PACKET = runner.snapshot, runner.public_packet
ORIG_CONTROL, ORIG_VALUES = policy.observation_control, policy.projected_values
ORIG_OLD_VALUES, ORIG_CHOOSE = old_timing.opportunities, policy.RobustTimingPolicy.choose
EXPECTED_ROWS = None
USED = False
FRAMES = PACKETS = PROJECTED = SIGNAL_CONTROLS = 0


def control(packet, mode):
    if mode == 'original':
        return packet
    out = dict(packet, tasks=[dict(r) for r in packet['tasks']])
    if mode == 'masked':
        for row in out['tasks']:
            row['temperature_c'] = row['last_temperature_c'] = row['nominal_temperature_c'] = 50.
            row['velocity_c_s'] = 0.;row['sensor_verified'] = False
            row['contact_c'] = 38. if row['kind'] == 'electrical' else None
            row['flow_kg_s'] = 48. if row['kind'] == 'cooling' else None
        return out
    fields = ('temperature_c','last_temperature_c','temperature_age_s','temperature_stamp_s',
        'velocity_c_s','nominal_temperature_c','contact_c','flow_kg_s','sensor_verified')
    roles = defaultdict(list)
    for j,row in enumerate(packet['tasks']):
        roles[row['kind']].append(j)
    for role,ids in roles.items():
        ordered = sorted(ids,key=lambda j:hashlib.sha256(('b2-permute:'+role+':'+str(j)).encode('ascii')).hexdigest())
        for k,target in enumerate(ordered):
            source = ordered[(k+1)%len(ordered)]
            for field in fields:
                out['tasks'][target][field] = packet['tasks'][source][field]
    return out


def checked_control(packet, mode):
    global SIGNAL_CONTROLS
    actual = ORIG_CONTROL(packet,mode)
    equivalent(actual,control(packet,mode),'Independent professional information transformation')
    SIGNAL_CONTROLS += 1
    return actual


def checked_packet(env):
    global PACKETS
    actual = ORIG_PACKET(env)
    equivalent(actual,independent_packet(env),'Public current planner state and independent graph costs')
    PACKETS += 1
    return actual


def checked_snapshot(env, scorer):
    actual = ORIG_SNAPSHOT(env,scorer)
    equivalent(actual,masks.independent_snapshot(env,scorer),'All current legal candidates and public features')
    return actual


def checked_values(packet, *, conservative=False):
    global EXPECTED_ROWS, PROJECTED
    actual = ORIG_VALUES(packet,conservative=conservative)
    expected = independent_opportunities(packet,120)
    if conservative:
        lowered = []
        for row in expected:
            r = dict(row)
            if r['dispatch_time_s'] > packet['time_s']:
                task = packet['tasks'][r['task']]
                cap = signal(task,r['finish_s']-r['dispatch_time_s'])
                reduced = min(cap,r['signal'])
                value = r['surrogate_value'] - task['weight_scaled']*(r['signal']-reduced)*max(r['baseline_target_finish_s']-r['finish_s'],0)/300
                r.update(signal=reduced,surrogate_value=value,
                    discounted_value=value*math.exp(-(r['dispatch_time_s']-packet['time_s'])/300))
            if r['surrogate_value'] > 0:
                lowered.append(r)
        expected = lowered
    equivalent(actual,expected,'Every original or conservative projected positive utility')
    EXPECTED_ROWS=expected;PROJECTED+=len(expected)
    return actual


def checked_old_values(packet,horizon):
    global EXPECTED_ROWS,PROJECTED
    actual=ORIG_OLD_VALUES(packet,horizon);expected=independent_opportunities(packet,horizon)
    equivalent(actual,expected,'Independent zero-horizon evidence utility')
    EXPECTED_ROWS=expected;PROJECTED+=len(expected)
    return actual


def checked_choose(self,groups,packet,used):
    global USED,EXPECTED_ROWS,FRAMES
    need(used==USED,'One global budget state')
    EXPECTED_ROWS=None
    result=ORIG_CHOOSE(self,groups,packet,used);choices,logs,consumed,diagnostic=result
    expected={g['agent']:g['baseline_next'] if g['candidates'] else None for g in groups}
    selected=None;method=self.config['method'];now=packet['time_s']
    if not used and method=='cue':
        ranked=[((.01+r['cue_strength'])*r['asset_weight_scaled']/(.03+r['eta_over_deadline']),g['agent'],r['task'])
            for g in groups for r in g['candidates'] if r['is_optional'] and r['cue_strength']>=.05 and not r['known_measured_alarm']]
        if ranked:
            _,i,j=max(ranked,key=lambda r:r[0]);selected=(i,j)
    elif not used and method=='blind':
        seed=self.config['blind_seed']
        gate=5*(1+int(hashlib.sha256(('b2-blind-time:'+str(seed)).encode('ascii')).hexdigest()[:16],16)%300)
        need(diagnostic['blind_trigger_s']==gate,'Public blind trigger independent of process phase')
        options=[(g['agent'],r['task']) for g in groups for r in g['candidates'] if r['is_optional']]
        if now>=gate and options:
            selected=min(options,key=lambda pair:hashlib.sha256(('b2-blind-action:%d:%g:%d:%d'%(seed,now,*pair)).encode('ascii')).hexdigest())
    elif not used and method in ('evidence','lookahead'):
        need(EXPECTED_ROWS is not None,'Independent public projected values were computed')
        current=[r for r in EXPECTED_ROWS if r['dispatch_time_s']==now]
        future=[r for r in EXPECTED_ROWS if r['dispatch_time_s']>now]
        best=max(current,key=lambda r:(r['surrogate_value'],-r['agent'],-r['task']),default=None)
        later=max(future,key=lambda r:(r['discounted_value'],-r['dispatch_time_s'],-r['agent'],-r['task']),default=None)
        value=best['surrogate_value'] if best else 0.;wait=later['discounted_value'] if later else 0.
        need(abs(diagnostic['now_value']-value)<1e-9 and abs(diagnostic['wait_value']-wait)<1e-9,'Competing now and wait utilities')
        equivalent(diagnostic['future_best'],later,'Actual future best reference')
        if best and value>max(0.,wait)+1e-9:
            selected=(best['agent'],best['task'])
    if selected is not None:
        expected[selected[0]]=selected[1]
    need(choices==expected and consumed==(selected is not None),'Independent check/wait or blind action')
    need(sum(v!=g['baseline_next'] for g in groups for i,v in choices.items() if i==g['agent'] and v is not None)<=1,'One team action')
    USED=used or consumed;FRAMES+=1
    return result


def main():
    global USED
    p=argparse.ArgumentParser()
    for key in ('results','actors','estimator-weights','estimator-manifest','output'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1);folder=args.results
    read=lambda p:json.loads(p.read_bytes())
    index=read(folder/'case-index.json');summary=read(folder/'summary.json')
    frozen=read(folder/'controls-frozen-before-test.json')
    expected_specs=specs(summary['chunk'],summary['smoke'])
    need(frozen['configurations']==configs() and frozen['conditions']==expected_specs,'All predeclared controls and phase/seed conditions')
    need([(r['case'],r['tag'],r['specification']) for r in index]==
        [(name(s,k),k,s) for k in configs() for s in expected_specs],'All exact case IDs, ordering and conditions')
    actors,fingerprints=load_actors(args.actors);actor=actor_scorer(actors[0]);predictor=frozen_predictor(args.estimator_weights,args.estimator_manifest)
    inputs=read(folder/'immutable-inputs.json');need(inputs['estimator_sha256']==sha(args.estimator_weights),'Frozen estimator bytes')
    processes=read(folder/'process-index.json');seen=set();dispatches=certificates=0;max_energy=max_electric=0.
    runner.snapshot,runner.public_packet=checked_snapshot,checked_packet
    policy.observation_control,policy.projected_values=checked_control,checked_values
    old_timing.opportunities=checked_old_values;policy.RobustTimingPolicy.choose=checked_choose
    try:
        for number,row in enumerate(index):
            path=folder/row['path'];need(sha(path)==row['sha256'] and path.stat().st_size==row['bytes'],'Raw original case bytes')
            original=json.loads(gzip.decompress(path.read_bytes()));config=configs()[row['tag']]
            need(row['deployment']['timing_config']==config,'Frozen actual deployment configuration')
            USED=False
            replay,env=runner.run(row['specification'],predictor,actor,config['method'],config=config)
            equivalent(replay,original,'Actual closed-loop B2 replay')
            independent_schedule(original,env);independent_events(original,env)
            passed,count=independent_certificates(original,env)
            independent_power_events(env,original);independent_public_and_quota(original,env)
            dispatches+=count;certificates+=passed
            need(original['metrics']['tasks_violating_exact_gap']==0 and original['metrics']['selected_detours']<=1,'Every whole-episode coverage/budget')
            optional=[r for r in original['workflow_decisions'] if r['task']!=r['baseline_next']]
            equivalent(row['optional_action'],optional[0] if optional else None,'Original selected action summary')
            waits=[r for r in original['timing_decisions'] if r['decision']=='wait_value' and (r['now_value'] or 0)>0]
            need(row['positive_now_wait_frames']==len(waits) and row['total_wait_value_frames']==sum(r['decision']=='wait_value' for r in original['timing_decisions']),'All wait/stall summaries')
            need(row['last_positive_wait_s']==(waits[-1]['time_s'] if waits else None),'Final positive-now rejection time')
            process=row['process_key']
            if process not in seen:
                info=processes[process];p=folder/info['path'];need(sha(p)==info['sha256'],'Raw physical array bytes')
                with np.load(p,allow_pickle=False) as pack:
                    for k in pack.files:
                        np.testing.assert_allclose(pack[k],env.trace.arrays[k],rtol=1e-11,atol=1e-9)
                    c=env.config;scalar=pack['process_scalars']
                    energy=c.chip_capacity_kj_k*pack['rack_temperature_c'].sum(1)+c.secondary_capacity_kj_k*pack['secondary_supply_c'].sum(1)+c.primary_capacity_per_pod_kj_k*c.pods*pack['primary_supply_c']+c.contact_capacity_kj_k*pack['contact_temperature_c'].sum(1)
                    residual=energy-np.r_[env.trace.metadata['initial_energy_kj'],energy[:-1]]-5*(scalar[:,2]-scalar[:,4])
                    electric=scalar[:,2]-scalar[:,5]-scalar[:,6]-scalar[:,7]-scalar[:,8]
                    need(abs(residual).max()<1e-4 and abs(electric).max()<1e-6,'Independent physical balances')
                    max_energy=max(max_energy,float(abs(residual).max()));max_electric=max(max_electric,float(abs(electric).max()))
                seen.add(process)
            if number%16==0:
                print('B2_AUDIT',number+1,'/',len(index),flush=True)
    finally:
        runner.snapshot,runner.public_packet=ORIG_SNAPSHOT,ORIG_PACKET
        policy.observation_control,policy.projected_values=ORIG_CONTROL,ORIG_VALUES
        old_timing.opportunities=ORIG_OLD_VALUES;policy.RobustTimingPolicy.choose=ORIG_CHOOSE
    lookup={r['case']:r for r in index}
    for tag,g in read(folder/'groups.json').items():
        rows=[lookup[k] for k in g['case_ids']]
        need([r['specification'] for r in rows]==expected_specs,'Every group common conditions')
        equivalent(g['aggregate'],aggregate(rows),'Independent outcome aggregation')
    for seed,model in actors.items():
        need(seed_digest(model)==fingerprints[str(seed)]['parameters_sha256'],'Old actor parameters unchanged')
    need(summary['actual_cases']==len(index) and summary['physical_configurations']==len(seen),'Complete case and physical scope')
    result={'chunk':summary['chunk'],'smoke':summary['smoke'],'actual_cases_replayed':len(index),
        'physical_array_files_verified':len(seen),'independent_public_candidates':masks.CANDIDATES_CHECKED,
        'independent_public_packets':PACKETS,'independent_stop_wait_frames':FRAMES,'independent_projected_positive_values':PROJECTED,
        'information_transformations_checked':SIGNAL_CONTROLS,'dispatches_and_measurements':dispatches,
        'whole_return_plans_checked':certificates,'exact_gap_violations':0,'global_optional_budget':1,
        'maximum_energy_residual_kj':max_energy,'maximum_electric_residual_kw':max_electric}
    with args.output.open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print('B2_AUDIT_PASSED',len(index),flush=True)


if __name__=='__main__':
    main()
