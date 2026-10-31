"""Known-development semantic/parity check; no new physics or Qwen generation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from patrol_rsi.f_identity import verify_identity
from patrol_rsi.limited_prediction import read,write,sha,need
from patrol_rsi.m11 import frozen_predictor
from patrol_rsi.m12 import load_actors,actor_scorer
from patrol_rsi.patrol_timing import gate_candidate
from patrol_rsi.timing_skill import TimingSkill
from patrol_rsi.timing_skill_features import from_legacy_row
from timing_skill_execution import run_policy,skill_policy
from run_m12d_paired import StoredTrace,write_gz

ACTORS=Path('/share/home/renly/renly-patrol-rsi/runs/20261027-m11/policies-v2')
CVAE=Path('/share/home/renly/renly-patrol-rsi/runs/20261028-m111/frozen-estimator.pt')
OLD=Path('/share/home/renly/renly-patrol-rsi/runs/20261031-m12e')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);os.umask(0o077);stage=a.source_root/'research/2026-10-31-m12f-timing-skill';cases_path=stage/'cases-v1.json'
    need(a.source_root.resolve()==Path(__file__).resolve().parents[1],'Actual CLI source is executing source')
    study=read(stage/'M12F-Study-v1.json');cases=read(cases_path)
    need(study['protocol']=='M12F-TimingSkill-Study-v1' and study['baseline']==cases['baseline']=='39d06ce5edb4dc7f7fff18bedc76909350a946d4','Exact F study and known cases baseline')
    need(study['cases_SHA256']==sha(cases_path),'Actual cases match reviewed study')
    known=cases['known_development']
    need(len(known)==20 and len({v['case_id'] for v in known})==20,'Exactly20unique frozen known cases')
    identity=a.run_root/'F0-execution-identity.json'
    verify_identity(identity,{'study':stage/'M12F-Study-v1.json','cases':cases_path,'CVAE':CVAE,
        'CVAE_manifest':a.source_root/'research/2026-10-28-m111-coverage/frozen_estimator.json',
        'actor_manifest':a.source_root/'research/2026-10-27-m11-scheduling/evidence/policy-weight-retrieval.json',
        **{f'actor{i}':ACTORS/f'history-seed-{i}/selected.pt' for i in range(3)}},'F0')
    identity_sha=sha(identity)
    root=a.run_root/'F0';root.mkdir(exist_ok=False);(root/'records').mkdir();begin=time.perf_counter();initial=TimingSkill.initial_e1()
    write(root/'started.json',{'identity_SHA256':identity_sha,'job_id':os.environ['SLURM_JOB_ID'],'scope':'Known E1 development parity only; no newindependentperformance claim','Qwen_calls':0})
    parity_count=0;different=0;source_checks=[]
    for seed in (33000,33001):
        for condition in range(18):
            release=read(a.source_root/'research/2026-10-31-m12e-timing-schema/evidence/run/E1'/f'seed-{seed}/public-release'/f'condition-{condition:02d}--public-selections-release.json')
            path=Path(release['original_private_path']);need(path.resolve().is_relative_to((OLD/'E1').resolve()) and sha(path)==release['original_private_SHA256'],'Original E1 development public source authority')
            for item in read(path)['candidate_public']:
                row=item['public'];expected=gate_candidate(row);actual=initial.action(from_legacy_row(row))=='CHECK'
                parity_count+=1;different+=expected!=actual
            source_checks.append({'path':str(path),'sha256':sha(path)})
    need(parity_count==298152 and different==0,'Initial TimingSkill exact E1 public predicate parity')
    predictor=frozen_predictor(CVAE,a.source_root/'research/2026-10-28-m111-coverage/frozen_estimator.json');actors,_=load_actors(ACTORS);scorer=actor_scorer(actors[0]);rows=[]
    for number,case in enumerate(known):
        spec=case['specification'];need(spec['seed'] in (33000,33001) and spec['phase']=='development','Only known olddevelopment, never final input')
        folder=Path(case['trace_folder']);need(folder.resolve().is_relative_to((OLD/'E1').resolve()) and sha(folder/'complete.json')==case['source_meta_SHA256'],'Known process metadata untouched')
        trace=StoredTrace(folder);need(trace.fingerprint==case['process_fingerprint'],'Known numeric process fingerprint')
        start=time.perf_counter();path=root/'records'/f'case-{number:02d}.json.gz'
        write(path.with_suffix('.attempt.json'),{'case_id':case['case_id'],'purpose':'F0 initial closed-loop equivalence','charged_environment_episode':True})
        record,env,audit=run_policy(spec,skill_policy(initial),trace,predictor,scorer)
        write_gz(path,record)
        write(path.with_suffix('.completed.json'),{'record_SHA256':sha(path),'action_SHA256':record['action_log_SHA256'],'audit':audit,'charged_environment_episode':True})
        metric_differences=[]
        for key,value in case['reference_E1_metrics'].items():
            if key in ('decision_CPU_seconds','history_estimator_inference_s'):continue
            actual=record['metrics'][key]
            equal=actual==value if value is None or not isinstance(value,float) else abs(actual-value)<=1e-8*max(1,abs(value))
            if not equal:metric_differences.append({'metric':key,'actual':actual,'expected':value})
        action_equal=record['action_log_SHA256']==case['reference_E1_action_SHA256']
        write(path.with_suffix('.parity.json'),{'action_hash_equal':action_equal,'metric_differences':metric_differences})
        need(action_equal and not metric_differences,'Initial E1 action/metric equivalence; actual charged record and mismatch receipt preserved')
        rows.append({'case_id':case['case_id'],'record_path':str(path),'record_SHA256':sha(path),'action_SHA256':record['action_log_SHA256'],
            'metrics':record['metrics'],'audit':audit,'public_cards':len(record['causal_experience_cards']),'episode_wall_s':time.perf_counter()-start})
        print('M12F_F0_EQUIVALENT',number,rows[-1]['episode_wall_s'],flush=True)
    need(sha(identity)==identity_sha,'Frozen identity unchanged throughout F0')
    write(root/'complete.json',{'complete':True,'initial_E1_candidate_parity':parity_count,'predicate_mismatches':different,'charged_known_closed_loop_episodes':len(rows),
        'all_action_hashes_and_metrics_equal':True,'rows':rows,'public_input_sources':source_checks,'identity_SHA256':identity_sha,'new_physical_processes':0,'Qwen_calls':0,
        'max_episode_wall_s':max(v['episode_wall_s'] for v in rows),'total_wall_s':time.perf_counter()-begin,'old_results_modified':False,'old_final_accessed':False})
    print('M12F_F0_COMPLETE',parity_count,len(rows),flush=True)


if __name__=='__main__':main()
