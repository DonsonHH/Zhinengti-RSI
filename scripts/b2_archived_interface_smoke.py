"""Ten engineering episodes on ONE archived development process; no plant step."""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from patrol_rsi.active_inspection import ProcessTrace
from patrol_rsi.aligned_patrol import arm_programs,run_episode
from patrol_rsi.facility import FacilityConfig
from patrol_rsi.limited_prediction import need,read,sha,write
from patrol_rsi.m11 import frozen_predictor
from patrol_rsi.m12 import load_actors,actor_scorer


class ArchivedTrace(ProcessTrace):
    """Original arrays and public packet fields; constructor never builds plant."""
    def __init__(self,folder,group):
        identity=read(folder/'source-identity-not-features.json');spec=identity['specification']
        self.metadata=read((group/identity['source_process_path']).with_suffix('.json'))
        self.config=FacilityConfig(**self.metadata['configuration']);self.seed=spec['seed'];self.scenario=spec['scenario'];self.control='feedback'
        with np.load(group/identity['source_process_path']) as f:self.arrays={k:f[k] for k in f.files}
        self.fingerprint=identity['numeric_process_fingerprint'];self.packets=[]
        thermal_indices=np.arange(0,128,4);flow_indices=np.arange(0,16,2)
        with gzip.open(folder/'public-states.jsonl.gz','rt',encoding='utf-8') as stream:
            for tick,line in enumerate(stream):
                value=json.loads(line);telemetry=value['telemetry']
                self.packets.append({'time_s':float((tick+1)*5),'temperature_indices':thermal_indices,
                    'temperature_c':self.arrays['public_sensor_temperature_c'][tick,thermal_indices].copy(),
                    **{k:np.asarray(telemetry[k],float) for k in ('public_requested_it_kw','public_delivered_it_kw','pod_pump_command','secondary_supply_c')},
                    'primary_supply_c':telemetry['primary_supply_c'],'flow_pod_indices':flow_indices,
                    'flow_pod_kg_s':self.arrays['rack_flow_kg_s'][tick].reshape(16,8).sum(1)[flow_indices]})
        need(len(self.packets)==359,'Alloriginalpublicpackets')
        # The mission has a final integration boundary without a decision root.
        self.packets.append(self.packets[-1]|{'time_s':1800.,
            'temperature_c':self.arrays['public_sensor_temperature_c'][359,thermal_indices].copy()})


def main():
    p=argparse.ArgumentParser()
    for key in ('source-root','source-inventory','actors','estimator-weights','estimator-manifest','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--prior-engineering-receipt',type=Path)
    args=p.parse_args();os.umask(0o077);torch.set_num_threads(1)
    group=args.source_root/'development/seed-20000';folder=group/'development--20000--coldplate_fouling--180--335--0.2'
    identity=read(folder/'source-identity-not-features.json');inventory={v['path']:v for v in read(args.source_inventory)['files']}
    checked=[]
    for path in (folder/'source-identity-not-features.json',folder/'public-states.jsonl.gz',group/identity['source_process_path'],(group/identity['source_process_path']).with_suffix('.json')):
        need(sha(path)==inventory[str(path)]['sha256'],'Archived originalsourceSHA');checked.append({'path':str(path),'sha256':sha(path)})
    trace=ArchivedTrace(folder,group);predictor=frozen_predictor(args.estimator_weights,args.estimator_manifest)
    actors,actor_info=load_actors(args.actors);score=actor_scorer(actors[0]);args.output.mkdir(exist_ok=False);records={};rows=[]
    for mode in ('history','nominal'):
        for name,program in arm_programs().items():
            with patch('patrol_rsi.improver_task.phased_trace',return_value=trace):
                record=run_episode(identity['specification']|{'estimator':mode,'phase':'B2_engineering_only'},program,predictor,score)
            records[name,mode]=record
            actions=json.dumps(record['action_log'],sort_keys=True).encode()
            checks=json.dumps(record['inspection_events'],sort_keys=True).encode()
            rows.append({'arm':name,'estimator':mode,'workflow':program.to_dict(),'metrics':record['metrics'],'usage':record['usage'],
                'action_log_SHA256':__import__('hashlib').sha256(actions).hexdigest(),'inspection_events_SHA256':__import__('hashlib').sha256(checks).hexdigest(),
                'process_fingerprint':trace.fingerprint})
            print('B2_ENGINEERING_EPISODE',name,mode,record['usage']['actual_PPO_dispatches'],record['metrics']['selected_detours'],flush=True)
    comparisons=[]
    for name in arm_programs():
        h,n=records[name,'history'],records[name,'nominal']
        comparisons.append({'arm':name,'action_frames_different':sum(a!=b for a,b in zip(h['action_log'],n['action_log'])),
            'history_minus_nominal_missed':h['metrics']['missed_segments']-n['metrics']['missed_segments'],
            'history_wait_s':h['metrics']['consequence_weighted_censored_delay_s'],'nominal_wait_s':n['metrics']['consequence_weighted_censored_delay_s']})
    repair=None
    if args.prior_engineering_receipt:
        prior=read(args.prior_engineering_receipt);old={(v['arm'],v['estimator']):v for v in prior['rows']}
        need(all(v['action_log_SHA256']==old[v['arm'],v['estimator']]['action_log_SHA256'] and v['inspection_events_SHA256']==old[v['arm'],v['estimator']]['inspection_events_SHA256'] for v in rows),
            'Receipt-label repair cannot change any engineering action or effective measurement')
        repair={'prior_receipt_SHA256':sha(args.prior_engineering_receipt),'all10_action_and_measurement_hashes_unchanged':True,
            'reason':'After globaloneCHECKused, compulsoryTour decisions mustnot be creditedasactualPPO,evenwhen shadowPPOalsoselectedTour.'}
    write(args.output/'interface-smoke.json',{'scope':'Engineeringonly:10episodes onealreadyknown developmentphysicalprocess. Notformalcomparison,noscientifictaskorRSIclaim.',
        'terminal_packet_limit':'No1800sdecisionroot archived. Carry1795nonthermaltelemetryand1800sensorarray atterminalupdateonly;all10arms useidenticalpacket. NotanexactfullTourreplayor formalnewenvironmentresult.',
        'sourcefiles':checked,'actors':actor_info,'CVAE_SHA256_before_and_after':sha(args.estimator_weights),'rows':rows,
        'CVAE_deployment_cross_interventions':comparisons,'receipt_correctness_repair':repair,'new_physical_processes':0,'plant_step_calls':0,'new_model_training':0,'real_LLM_calls':0,
        'phases_opened':['development'],'final_access':False,'job_id':os.environ.get('SLURM_JOB_ID')})


if __name__=='__main__':main()
