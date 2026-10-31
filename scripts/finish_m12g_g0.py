"""Join completed G0 batches and make the preregistered local-mechanism decision."""
import argparse
from pathlib import Path
from patrol_rsi.limited_prediction import read,write,sha,need
from patrol_rsi.g_timing_patch import digest
from patrol_rsi.timing_learning import strict_gain,strictly_better_key
from patrol_rsi.timing_skill import TimingSkill


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();root=a.run_root.resolve()
    need(not(root/'G-STOP.json').exists(),'No completed scientific result after source/runtime/safety stop')
    identity_path=root/'G-execution-identity.json';identity=read(identity_path);identity_sha=sha(identity_path)
    need(Path(identity['run_root']).resolve()==root,'Frozen G root')
    catalog=read(Path(identity['bound_inputs']['candidates']));cases=read(Path(identity['bound_inputs']['cases']))['cases']
    paths=[root/f'G0/batch-{i}/complete.json' for i in (0,1)];batches=[read(p) for p in paths]
    need(all(v['complete'] and v['identity_SHA256']==identity_sha and v['actual_environment_evaluations']==240 for v in batches),'Both real batches completed under same identity')
    results=[v for b in batches for v in b['results']]
    need([v['candidate']['id'] for v in results]==[v['id'] for v in catalog['policies']] and len(results)==24,'Exactly all24slots in frozen order')
    expected={digest(v['specification']) for v in cases}
    f=read(Path(identity['bound_inputs']['F_search_complete']));refs=f['evaluation_ledger']['results']
    initial_policy={'executor':'timing-skill-v1','ranker':'urgency','skill':TimingSkill.initial_e1().to_dict()}
    initial=next(v for v in refs if digest(v['policy'])==digest(initial_policy));supported=[]
    for result in results:
        need(len(result['rows'])==20 and {digest(v['specification']) for v in result['rows']}==expected and all(v['audit']['passed'] for v in result['rows']),'Every slot has20correct qualified/audited actual cases')
        for row in result['rows']:need(sha(Path(row['record_path']))==row['record_SHA256'],'Actual raw record bytes retained')
        need(result['gain_vs_initial']==strict_gain(result['statistics'],initial['statistics']),'Same frozen gain formula')
        if result['gain_vs_initial'] and result['eligible_local_mechanism']:supported.append(result)
    best=None
    for result in supported:
        if best is None or strictly_better_key(result['statistics'],best['statistics']):best=result
    value={'complete':True,'identity_SHA256':identity_sha,'sealed_batches':[{'path':str(p),'sha256':sha(p)} for p in paths],
        'actual_environment_evaluations':480,'candidate_slots':24,'all_slots_actual':True,'new_physical_processes':0,'model_calls':0,
        'local_mechanism_supported':bool(supported),'supported_local_candidates':[v['candidate']['id'] for v in supported],
        'selected_local_reference':best['candidate'] if best else None,'initial_reference_statistics':initial['statistics'],
        'results':results,'scope':'Repeated-known-development diagnostic only, not independent skill/experience gain.',
        'H01_only_gain_does_not_open_precheck':True,'no_25th_candidate':True}
    write(root/'G0-complete.json',value)
    print('G0_COMPLETE',len(results),480,bool(supported),value['supported_local_candidates'],flush=True)


if __name__=='__main__':main()
