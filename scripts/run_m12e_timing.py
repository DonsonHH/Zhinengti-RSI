"""Fresh-development timing comparison and bounded offline witness diagnostics."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from patrol_rsi.aligned_patrol import arm_programs
from patrol_rsi.active_inspection import CANDIDATE_FEATURES
from patrol_rsi.b2_actions import evaluate_schedule,branch_schedule,delta_outcome
from patrol_rsi.causal_observation import capture_public
from patrol_rsi.e_stage import verify_stage_identity
from patrol_rsi.estimation_data import CompositeFacility
from patrol_rsi.facility import FacilityConfig
from patrol_rsi.improver_task import PatrolProgram
from patrol_rsi.information_audit import current_legal_geometry,pack_geometry,analytic_continuations,reference_signals
from patrol_rsi.inspection_metrics import task_labels,anomaly_intervals
from patrol_rsi.limited_prediction import read,write,sha,need
from patrol_rsi.m11 import frozen_predictor
from patrol_rsi.m12 import load_actors,actor_scorer
from patrol_rsi.m9 import arrays_digest
from patrol_rsi.patrol_timing import TimingSelector,METHODS,gate_candidate,count_later_roots
from run_m12d_paired import execute,audit_record,usage_details,save_trace,StoredTrace,write_gz

ACTORS=Path('/share/home/renly/renly-patrol-rsi/runs/20261027-m11/policies-v2')
CVAE=Path('/share/home/renly/renly-patrol-rsi/runs/20261028-m111/frozen-estimator.pt')


class IndependentPhaseTrace:
    """Same CompositeFacility equations and sampling; new preregistered phase only."""
    def __init__(self,spec):
        seed,scenario,duration,start=(spec[k] for k in ('seed','scenario','duration_s','fault_start_s'))
        need(type(start) is int and start%5==0 and 95<=start<=895,'Only new independently registered phase grid')
        self.config=FacilityConfig(fault_start_s=start,fault_end_s=start+duration);self.seed=seed;self.scenario=scenario;self.control='feedback'
        plant=CompositeFacility(self.config,seed,scenario,'feedback');keys=('rack_temperature_c','secondary_supply_c','primary_supply_c',
            'contact_temperature_c','rack_flow_kg_s','public_sensor_temperature_c','severity_proxy')
        values={k:[] for k in keys};self.packets=[];scalar=[];initial=plant.stored_energy_kj();energy=electric=0.
        for _ in range(360):
            row=plant.step();self.packets.append(plant.public_telemetry())
            for k in keys:values[k].append(row[k])
            scalar.append([row[k] for k in ('time_s','unserved_it_kw','source_power_kw','stored_energy_kj','rejected_heat_kw','it_power_kw','aux_power_kw','conversion_loss_kw','contact_loss_kw')])
            energy=max(energy,abs(row['energy_residual_kj']));electric=max(electric,abs(row['electric_residual_kw']))
        self.arrays={k:np.stack(v) for k,v in values.items()};self.arrays['process_scalars']=np.asarray(scalar);self.arrays['asset_weights']=plant.criticality.copy();self.fingerprint=arrays_digest(self.arrays)
        self.metadata={'racks':128,'seed':seed,'scenario':scenario,'control':'feedback','fault_duration_s':duration,'fault_start_s':start,'configuration':asdict(self.config),
            'numeric_digest':self.fingerprint,'initial_energy_kj':initial,'maximum_energy_residual_kj':energy,'maximum_electric_residual_kw':electric,
            'peak_source_input_MW':max(v[2] for v in scalar)/1000,'IT_nameplate_MW':32.,'unserved_IT_kwh':sum(v[1] for v in scalar)*5/3600}


def outcome(record):
    intervals=record['anomaly_intervals']
    return {'missed':record['metrics']['missed_segments'],'weighted_wait_s':record['metrics']['consequence_weighted_censored_delay_s'],
        'first_confirmation_s':[v['first_confirmation_s'] for v in intervals],'event_weight_sum':sum(v['weight'] for v in intervals)}


def public_case_row(record,path,audit):
    return {'specification':record['specification'],'method':record['timing_method'],'metrics':record['metrics'],'by_role':record['by_role'],
        'usage':record['usage'],'process_fingerprint':record['process_fingerprint'],'public_telemetry_sequence_SHA256':record['public_telemetry_sequence_SHA256'],
        'decision_frames':record['timing_decision_frames'],'first_root_actor_scores':record['first_root_actor_scores'],'feature_names':list(CANDIDATE_FEATURES),
        'record_path':str(path),'record_SHA256':sha(path),'audit':audit}


def run(spec,method,trace,predictor,scorer,path,*,stop_path,capture=None,forced=None):
    need(not stop_path.exists(),'No new episode after observed peer E1 stop')
    pg=arm_programs()['human_professional'] if method=='human_professional' else PatrolProgram(fallback='ppo') if method=='unconditional_PPO' else PatrolProgram()
    selector=TimingSelector(method,capture=capture,forced=forced)
    write(path.with_suffix('.attempt.json'),{'specification':spec,'method':method,'forced_offline_witness':forced,'charged_episode':True})
    with patch('patrol_rsi.aligned_patrol.choose_program',selector):record,env=execute(spec,pg,trace,predictor,scorer)
    audit=audit_record(record,env);record.update(timing_method=method,timing_decision_frames=env.e1_decision_frames,first_root_actor_scores=env.e1_first_scores)
    record['episode_purpose']='offline_witness' if forced else 'main'
    record['usage']=usage_details(record);record['usage']['actual_actor_network_calls_in_selector']=sum(v['actor_scoring_calls'] for v in env.e1_decision_frames)
    write_gz(path,record)
    row=public_case_row(record,path,audit);row['episode_purpose']=record['episode_purpose']
    return record,env,row


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--seed-index',type=int,required=True);a=p.parse_args()
    os.umask(0o077);torch.set_num_threads(1);stage=a.source_root/'research/2026-10-31-m12e-timing-schema';protocol=stage/'E1-Timing-v1.json';manifest=stage/'process-manifest.json'
    identity=a.run_root/'E1-execution-identity.json';binding=verify_stage_identity(identity,{'E1':protocol,'manifest':manifest,
        'actor0':ACTORS/'history-seed-0/selected.pt','actor1':ACTORS/'history-seed-1/selected.pt','actor2':ACTORS/'history-seed-2/selected.pt','CVAE':CVAE,
        'CVAE_manifest':a.source_root/'research/2026-10-28-m111-coverage/frozen_estimator.json',
        'actor_manifest':a.source_root/'research/2026-10-27-m11-scheduling/evidence/policy-weight-retrieval.json','history':a.run_root/'historical-identities.json'},scope='E1')
    need(a.source_root.resolve()==Path(binding['source_root']).resolve(),'Actual CLI source matches E1 import tree');identity_sha=sha(identity)
    config=read(protocol);need(config['development_manifest_SHA256']==sha(manifest) and a.seed_index in (0,1),'Frozen development only; no final runtime path')
    seed=config['partitions']['development'][a.seed_index];specs=[v for v in read(manifest)['rows'] if v['phase']=='development' and v['seed']==seed]
    need(len(specs)==18,'Nine physical processes/two noises');root=a.run_root/'E1'/f'seed-{seed}';root.mkdir(parents=True,exist_ok=False)
    for name in ('cases','sources','public','diagnostics'): (root/name).mkdir()
    write(root/'started.json',{'seed':seed,'protocol_SHA256':sha(protocol),'manifest_SHA256':sha(manifest),'execution_identity_SHA256':identity_sha,'job_id':os.environ['SLURM_JOB_ID']})
    history=read(a.run_root/'historical-identities.json');need(seed not in history['seed_groups'],'No historical generation seed reused')
    predictor=frozen_predictor(CVAE,a.source_root/'research/2026-10-28-m111-coverage/frozen_estimator.json');actors,_=load_actors(ACTORS);scorer=actor_scorer(actors[0])
    rows=[];reports=[];processes={};attempts=0;begin=time.perf_counter()
    for ordinal,spec in enumerate(specs):
        need(not(a.run_root/'E1-STOP.json').exists(),'Peer E1 failure stops remaining development work')
        key='--'.join(str(spec[k]) for k in ('seed','scenario','duration_s','fault_start_s'));folder=root/'sources'/key
        if key not in processes:
            need(not(a.run_root/'E1-STOP.json').exists(),'No new physical process after peer E1 stop')
            write(root/'sources'/(key+'--generation-attempt.json'),{'specification':spec,'charged_physical_generation':True})
            trace=IndependentPhaseTrace(spec)
            need(trace.fingerprint not in history['numeric_fingerprints'] and trace.fingerprint not in processes.values(),'Fresh unique numeric physical process')
            save_trace(trace,folder);processes[key]=trace.fingerprint
        trace=StoredTrace(folder);geometry_rows=[];candidate_public=[];top1={};capture_count=0
        public_path=root/'public'/f'condition-{ordinal:02d}.jsonl.gz'
        with gzip.open(public_path,'xt',encoding='utf-8') as stream:
            def capture(env,groups,hypothetical):
                nonlocal capture_count
                bundle=capture_public(env);local=current_legal_geometry(bundle);expected={(v['agent'],v['task']) for g in groups for v in g['public'] if v['task']!=v['baseline_next']}
                need(expected=={(v['agent'],v['task']) for v in local},'Independent public geometry equals actual executor legal optional set')
                offset=len(geometry_rows);lookup={(v['agent'],v['task']):offset+i for i,v in enumerate(local)}
                fields={(v['agent'],v['task']):(v,float(g['scores'][k])) for g in groups for k,v in enumerate(g['public'])}
                for item in local:
                    value,score=fields[item['agent'],item['task']];candidate_public.append({'eligible':gate_candidate(value),'public':value,'actor_score':score})
                for method,value in hypothetical.items():
                    if value:top1.setdefault(method,[]).append(lookup[value['agent'],value['task']])
                geometry_rows.extend(local);stream.write(json.dumps({'public':bundle,'hypothetical_public_Top1':hypothetical},allow_nan=False)+'\n');capture_count+=1
            records={};condition_rows=[]
            for method in METHODS:
                path=root/'cases'/f'condition-{ordinal:02d}--{method}.json.gz';attempts+=1
                record,env,row=run(spec,method,trace,predictor,scorer,path,stop_path=a.run_root/'E1-STOP.json',capture=capture if method=='Tour' else None)
                records[method]=record;rows.append(row);condition_rows.append(row)
        need(capture_count==359,'Every Tour decision root5..1795 captured')
        geometry=pack_geometry(geometry_rows);np.savez_compressed(root/'public'/f'condition-{ordinal:02d}--geometry.npz',**geometry)
        receipt=root/'public'/f'condition-{ordinal:02d}--decisions-sealed.json'
        write(receipt,{'before_offline_oracle_evaluation':True,'public_trace_SHA256':sha(public_path),'all5_actual_action_log_SHA256':{m:hashlib.sha256(json.dumps(records[m]['action_log'],sort_keys=True).encode()).hexdigest() for m in METHODS},
            'hypothetical_public_Top1_indices':top1,'candidate_public':candidate_public,'public_only_no_labels':True})
        # Only now does the diagnostic read event labels and future physical arrays.
        base=records['Tour'];physical,excess=task_labels(SimpleNamespace(trace=trace,config=env.config,tasks=env.tasks))
        intervals=anomaly_intervals(physical,excess,env.tasks,env.asset_weights);need(len(intervals)==len(base['anomaly_intervals']),'Common physical event eligibility')
        analytic=analytic_continuations(base,geometry,reference_signals(trace.arrays),measurement_positive=physical)
        delta=analytic['missed_delta'];wait=analytic['weighted_wait_delta_s'];net=-np.nan_to_num(wait,nan=0)-.05*geometry['shift_s']
        useful=(analytic['lost_confirmations']==0)&(delta<=0)&((delta<0)|(net>1e-9))
        good=np.flatnonzero(useful);best=int(min(good,key=lambda i:(delta[i],-net[i],geometry['dispatch_time_s'][i],geometry['agent'][i],geometry['task'][i]))) if len(good) else None
        public_options=[(i,order,m) for order,m in enumerate(('evidence_gate_PPO','evidence_gate_urgency')) for i in top1.get(m,[]) if useful[i]]
        first=min(public_options,key=lambda v:(geometry['dispatch_time_s'][v[0]],v[1],geometry['agent'][v[0]],geometry['task'][v[0]])) if public_options else None
        wanted={}
        if best is not None:wanted.setdefault(best,[]).append('offline_best')
        if first is not None:wanted.setdefault(first[0],[]).append('earliest_public_Top1')
        witnesses=[]
        for number,(i,tags) in enumerate(wanted.items()):
            need(attempts<126,'Perseed90main+36witness episode cap')
            forced={k:(int(geometry[k][i]) if k in ('agent','task') else float(geometry[k][i])) for k in ('agent','task','dispatch_time_s')}
            path=root/'cases'/f'condition-{ordinal:02d}--oracle-witness-{number}.json.gz';attempts+=1
            record,_,row=run(spec,'Tour',trace,predictor,scorer,path,stop_path=a.run_root/'E1-STOP.json',forced=forced);rows.append(row)
            actual=delta_outcome(outcome(base),outcome(record),float(geometry['shift_s'][i]))
            predicted={'missed_delta':int(delta[i]),'weighted_wait_delta_s':float(wait[i]) if np.isfinite(wait[i]) else None,
                'gained_confirmations':int(analytic['gained_confirmations'][i]),'lost_confirmations':int(analytic['lost_confirmations'][i]),
                'first_confirmation_s':analytic['first_confirmation_s'][i].tolist()}
            actual_times=[v['first_confirmation_s'] if v['first_confirmation_s'] is not None else -1 for v in record['anomaly_intervals']]
            matches=all(actual[k]==predicted[k] for k in ('missed_delta','gained_confirmations','lost_confirmations')) and actual_times==predicted['first_confirmation_s']
            matches=matches and (actual['weighted_wait_delta_s'] is None and predicted['weighted_wait_delta_s'] is None or actual['weighted_wait_delta_s'] is not None and predicted['weighted_wait_delta_s'] is not None and np.isclose(actual['weighted_wait_delta_s'],predicted['weighted_wait_delta_s'],atol=1e-8))
            witnesses.append({'tags':tags,'geometry_index':int(i),'forced_current_legal_check':forced,'actual':actual,'analytic':predicted,
                'actual_first_confirmation_s':actual_times,'analytic_matches_actual':bool(matches),'record_path':str(path),'record_SHA256':sha(path),'audit':row['audit']})
        method_opportunities={}
        for method in METHODS:
            extra=next((v for v in records[method]['workflow_decisions'] if v['task']!=v['baseline_next']),None)
            later=[i for i in good if extra is None or geometry['dispatch_time_s'][i]>extra['time_s']]
            method_opportunities[method]={'spent_at_s':extra['time_s'] if extra else None,'later_analytic_useful_candidates':len(later),
                'later_public_Top1_useful_roots':count_later_roots((geometry['dispatch_time_s'][i] for i,_,_ in public_options),extra['time_s'] if extra else 0),
                'not_jointly_executable_or_closed_loop_verified':True}
        report={'specification':spec,'process_fingerprint':trace.fingerprint,'public_receipt_before_labels_SHA256':sha(receipt),'roots':359,'legal_optional_candidates':len(delta),
            'analytic_useful_candidates':int(useful.sum()),'public_eligible_analytic_useful_candidates':sum(v['eligible'] and bool(useful[i]) for i,v in enumerate(candidate_public)),
            'public_Top1_analytic_useful_candidates':len({i for i,_,_ in public_options}),'public_Top1_analytic_useful_ranker_references':len(public_options),'actual_witnesses':witnesses,
            'public_Top1_actual_beneficial':any('earliest_public_Top1' in v['tags'] and v['actual']['net_beneficial'] for v in witnesses),
            'missed_opportunity_diagnostics':method_opportunities,'physics_and_noisy_telemetry_matched':len({v['process_fingerprint'] for v in condition_rows})==len({v['public_telemetry_sequence_SHA256'] for v in condition_rows})==1}
        need(report['physics_and_noisy_telemetry_matched'],'All5methods exact same physical and noisy public source')
        write(root/'diagnostics'/f'condition-{ordinal:02d}.json',report);reports.append(report)
        print('M12E_TIMING_CONDITION',seed,ordinal,len(delta),int(useful.sum()),len(witnesses),flush=True)
    need(len(processes)==9 and attempts<=126 and sha(identity)==identity_sha,'Fixed source/physical/episode scope')
    write(root/'complete.json',{'complete':True,'seed':seed,'main_episodes':90,'actual_total_episode_attempts':attempts,'witness_episodes':attempts-90,'rows':rows,'diagnostics':reports,
        'numeric_processes':processes,'protocol_SHA256':sha(protocol),'execution_identity_SHA256':identity_sha,'elapsed_s':time.perf_counter()-begin,
        'validation_final_source_access':False,'LLM_calls':0,'weights_updated':False})
    print('M12E_TIMING_SEED_COMPLETE',seed,attempts,flush=True)


if __name__=='__main__':main()
