"""Frozen five-arm campaign identities, grouped metrics and final-open gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .aligned_patrol import arm_programs
from .limited_prediction import need,read,sha,write

PAIRED_SHA='dbea4a9b83cbbe4475d3343492646328c6a1c737f22ff79a785c8f2fb17f914a'
RSI_SHA='3bd4e2478c50f152898ecf73d1359dbc246e5c3501425daa034b215328edbe76'
ARMS=('Tour','frozen_PPO','human_professional','fixed_improver_retained','revised_PPO_evidence')


def exclude_history(seed,fingerprint,registry):
    need(seed not in registry['seed_groups'] and fingerprint not in registry['numeric_fingerprints'],'New source overlaps an old process/seed')


def verify_identity(identity,required=None):
    keys={'paired','rsi','manifest','review','gate','history','actor0','actor1','actor2','CVAE','CVAE_manifest'}
    need(set(identity['bound_inputs'])==keys and identity['paired_SHA256']==PAIRED_SHA and identity['RSI_SHA256']==RSI_SHA,'Closed execution identity')
    indexed={v['path']:v['sha256'] for v in identity['artifacts']}
    need(len(indexed)==len(identity['artifacts']) and set(identity['bound_inputs'].values())<=set(indexed),'All bound inputs covered by unique hashes')
    if required:
        need(all(str(Path(value).resolve())==identity['bound_inputs'][key] for key,value in required.items()),'Actual CLI paths match bound source/model/gate/review')
    root=Path(identity['source_root'])
    need(root.resolve()==Path(__file__).resolve().parents[2],'Identity source root equals actual imported code tree')
    for group in ('src','scripts','tests'):
        files=list((root/group).rglob('*.py'));need(files,'Executed source tree present')
        need(all(str(p.resolve()) in indexed for p in files),'All executed source/tests included in identity')
    for path,digest in indexed.items():need(sha(Path(path))==digest,'Executed source/models/gate/review/history bytes unchanged')
    return indexed


def protocols(paired,rsi):
    need(sha(paired)==PAIRED_SHA and sha(rsi)==RSI_SHA,'Frozen protocol bytes mismatch;stop for review')
    p,r=read(paired),read(rsi)
    need(p['arms']=={k:v.to_dict() for k,v in arm_programs().items()} and tuple(p['arms'])==ARMS,'Exact fixed five programs')
    need(p['partition']=={'development':[31000,31001],'validation':[31020,31021],'final_test':[31040,31041,31042,31043]},'Exact frozen phase seeds')
    return p,r


def manifest(protocol):
    rows=[]
    for phase,seeds in protocol['partition'].items():
        for seed in seeds:
            for scenario,fixtures in protocol['physical_fixtures'].items():
                for duration,onset in fixtures:
                    physical=f'{seed}--{scenario}--{duration}--{onset}'
                    for noise in protocol['noise_c']:
                        spec={'phase':phase,'seed':seed,'scenario':scenario,'duration_s':duration,'fault_start_s':onset,
                            'noise_c':noise,'estimator':'history','control':'feedback','resources':'provisioned','variant':'open'}
                        rows.append({'id':physical+'--'+str(noise),'phase':phase,'seed':seed,'physical_key':physical,'specification':spec})
    need(len(rows)==144 and len({v['physical_key'] for v in rows})==72,'9physicalprocesses and2noises perseed')
    return {'protocol_SHA256':PAIRED_SHA,'rows':rows,'formal_episode_cap':720,'final_unopened':True}


def aggregate(rows):
    result={}
    for arm in ARMS:
        own=[v for v in rows if v['arm']==arm];need(own,'Every arm must be present')
        seedrows=[]
        for seed in sorted({v['seed'] for v in own}):
            current=[v for v in own if v['seed']==seed];processes={}
            for v in current:processes.setdefault(v['physical_key'],[]).append(v)
            need(len(processes)==9 and all(len(v)==2 and {r['noise_c'] for r in v}=={.2,1.} for v in processes.values()),'All9processes/two pairednoise conditions')
            means=[];wait=[]
            for group in processes.values():
                need(group[0]['process_fingerprint']==group[1]['process_fingerprint'],'Noise versions share physical arrays')
                means.append(np.mean([v['metrics']['missed_segments'] for v in group]))
                waits=[v['metrics']['consequence_weighted_censored_delay_s'] for v in group]
                need((waits[0] is None)==(waits[1] is None),'Model-independent event eligibility')
                if waits[0] is not None:wait.append(float(np.mean(waits)))
            seedrows.append({'seed':seed,'macro_missed_count':float(np.mean(means)),
                'macro_wait_s':float(np.mean(wait)) if wait else None,'eventbearing_processes':len(wait),
                'processes':len(processes),'conditions':len(current),
                'missed_noise_repeated_sum':sum(v['metrics']['missed_segments'] for v in current),
                'events_noise_repeated_sum':sum(v['metrics']['anomaly_segments'] for v in current),
                'mean_travel_m':float(np.mean([v['metrics']['travel_m'] for v in current])),
                'mean_service_s':float(np.mean([v['metrics']['service_s'] for v in current])),
                'mean_idle_s':float(np.mean([v['metrics']['idle_s'] for v in current])),
                'mean_optional_checks':float(np.mean([v['metrics']['selected_detours'] for v in current])),
                'actual_PPO_dispatches':sum(v['usage']['actual_PPO_dispatches'] for v in current),
                'PPO_multichoice_dispatches':sum(v['usage']['PPO_multichoice_dispatches'] for v in current),
                'total_dispatches':sum(v['usage']['dispatches'] for v in current),
                'estimator_calls':sum(v['usage']['estimator_calls'] for v in current)})
        result[arm]={'seeds':seedrows,'macro_missed_count':float(np.mean([v['macro_missed_count'] for v in seedrows])),
            'macro_wait_s':float(np.mean([v['macro_wait_s'] for v in seedrows if v['macro_wait_s'] is not None])),
            'conditions':len(own),'missed_noise_repeated_sum':sum(v['metrics']['missed_segments'] for v in own),
            'safety_ok':all(v['metrics']['tasks_violating_exact_gap']==0 and v['metrics']['invalid_assignments']==0 and v['metrics']['duplicate_reservations']==0 and v['metrics']['selected_detours']<=1 and v['metrics']['final_professional_coverage']==1. for v in own),
            'all_replays_and_independent_audits_passed':all(v['audit']['passed'] for v in own),
            'mean_optional_checks':float(np.mean([v['metrics']['selected_detours'] for v in own])),
            'mean_travel_m':float(np.mean([v['metrics']['travel_m'] for v in own])),
            'mean_service_s':float(np.mean([v['metrics']['service_s'] for v in own])),
            'mean_idle_s':float(np.mean([v['metrics']['idle_s'] for v in own]))}
    return result


def validation_gate(aggregates,*,sources_ok,budget_ok):
    revised=aggregates['revised_PPO_evidence'];checks={}
    for name in ('Tour','human_professional'):
        base=aggregates[name];perseed={v['seed']:v for v in base['seeds']}
        no_increase=all(v['macro_missed_count']<=perseed[v['seed']]['macro_missed_count'] for v in revised['seeds'])
        delta=revised['macro_missed_count']-base['macro_missed_count']
        relative=(base['macro_wait_s']-revised['macro_wait_s'])/base['macro_wait_s'] if base['macro_wait_s'] else None
        checks[name]={'no_seed_miss_increase':no_increase,'macro_missed_delta':delta,'relative_macro_wait_gain':relative,
            'passed':no_increase and (delta<0 or delta==0 and relative is not None and relative>=.05)}
    safety=all(v['safety_ok'] and v['all_replays_and_independent_audits_passed'] for v in aggregates.values())
    passed=bool(safety and sources_ok and budget_ok and all(v['passed'] for v in checks.values()))
    return {'passed':passed,'status':'validation_passed_seal_before_final' if passed else 'NO-GO_stop_final_sealed',
        'comparators':checks,'safety_and_audit_ok':safety,'source_disjointness_ok':sources_ok,'budget_ok':budget_ok,
        'no_substitute_no_extra_arm':True,'protocol_SHA256':PAIRED_SHA}


def require_final_seal(path,manifest_path):
    value=read(path)
    need(value['gate']['passed'] and value['protocol_SHA256']==PAIRED_SHA and value['manifest_SHA256']==sha(manifest_path),
        'Final remains sealed unless reviewed formal validation passed')
    need(value['RSI_SHA256']==RSI_SHA,'Same immutable RSI protocol context')
    need(value['arms']=={k:v.to_dict() for k,v in arm_programs().items()},'All five original nominal programs sealed')
    run=Path(path).parent;identity_path=run/'execution-identity.json';identity=read(identity_path)
    indexed=verify_identity(identity,{'manifest':manifest_path})
    artifacts={v['path']:v['sha256'] for v in value['artifacts']}
    required=[identity_path,run/'validation-complete.json',run/'resource-check-before-final.json']
    required += [run/'validation'/f'seed-{seed}'/'complete.json' for seed in (31020,31021)]
    need(all(str(p) in artifacts and sha(p)==artifacts[str(p)] for p in required),'Final seal covers identity,actual both validation seeds,aggregate and resource')
    need(all(artifacts.get(p)==digest for p,digest in indexed.items()),'Final seal contains every originally bound artifact')
    actual=read(run/'validation-complete.json')
    need(actual['complete'] and actual['gate']==value['gate'] and actual['formal_episodes']==180,'Actual180record validation gate equals seal')
    for item in value['artifacts']:need(sha(Path(item['path']))==item['sha256'],'Frozen validation/model/source artifacts unchanged')
    return value
