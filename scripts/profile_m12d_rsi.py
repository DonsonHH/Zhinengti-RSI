"""Measure same-executor engineering cost on the saved smoke trace only."""
import argparse
from pathlib import Path
import time

import torch

from patrol_rsi.aligned_patrol import arm_programs
from patrol_rsi.limited_prediction import read,write,sha
from patrol_rsi.m11 import frozen_predictor
from patrol_rsi.m12 import load_actors,actor_scorer
from run_m12d_paired import StoredTrace,execute,audit_record


def main():
    p=argparse.ArgumentParser()
    for n in ('source-root','run-root'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);r=a.run_root;s=a.source_root
    trace=StoredTrace(r/'smoke/seed-30990/sources/30990--coldplate_fouling--180--335')
    spec=read(r/'smoke/seed-30990/complete.json')['rows'][0]['specification']
    predictor=frozen_predictor(Path('/share/home/renly/renly-patrol-rsi/runs/20261028-m111/frozen-estimator.pt'),s/'research/2026-10-28-m111-coverage/frozen_estimator.json')
    actors,_=load_actors(Path('/share/home/renly/renly-patrol-rsi/runs/20261027-m11/policies-v2'));scorer=actor_scorer(actors[0]);rows=[]
    for arm,program in arm_programs().items():
        start=time.perf_counter();record,env=execute(spec,program,trace,predictor,scorer);wall=time.perf_counter()-start
        start=time.perf_counter();audit_record(record,env);audit=time.perf_counter()-start
        rows.append({'arm':arm,'single_episode_wall_s':wall,'independent_audit_wall_s':audit,'same_executor':True})
    q=read(r/'qwen-preflight/preflight.json')
    episode=max(v['single_episode_wall_s']+v['independent_audit_wall_s'] for v in rows)
    generation=q['generation_s']*768/q['output_tokens']
    bound=episode*1728/2+generation*26+q['model_load_and_verify_s']+120
    write(r/'rsi-cost-profile.json',{'scope':'Engineering timings on alreadycompleted smokeprocess. NotnewRSI adaptation or taskgain. Generation extrapolation is uncertain,not a runtime guarantee.',
        'rows':rows,'one_trial_max_nonfinal_environment_episode_cap':1728,'proposed_CPU_workers':2,
        'max_measured_episode_plus_audit_s':episode,'conservative_linear_768token_generation_s':generation,'formal_model_call_cap_per_trial':26,
        'projected_trial_GPU_reservation_s':bound,'frozen_one_allocation_limit_s':3600,
        'projected_within_one_allocation':bound<=3600,'real_model_generations':0,'new_physical_processes':0,'profile_engineering_episodes':5})
    print('M12D_RSI_COST_PROFILE',bound,'seconds',flush=True)


if __name__=='__main__':main()
