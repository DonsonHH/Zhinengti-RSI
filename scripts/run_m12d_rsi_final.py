"""Final fixed mappings after all three real RSI searches have sealed."""
import argparse
import multiprocessing as mp
from pathlib import Path

import torch

from patrol_rsi.improver_task import PatrolProgram
from patrol_rsi.limited_prediction import read,write,sha,need
from patrol_rsi.paired_campaign import protocols,verify_identity
from patrol_rsi.rsi_campaign import cases,macro,GROUPS,validate_mapping
from run_m12d_rsi import Store,init_worker,ACTORS,CVAE


def main():
    p=argparse.ArgumentParser()
    for n in ('source-root','run-root'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--trial',type=int,required=True);a=p.parse_args();torch.set_num_threads(1)
    need(a.trial in (0,1,2),'Three fixed generation repetitions')
    source=a.source_root;prior=source/'research/2026-10-30-m12c1b4b2-alignment';paired,rsi=protocols(prior/'MultiAgent-Patrol-Paired-v1.json',prior/'RSI-Controlled-Meta-v1.json')
    seal=read(a.run_root/'rsi-final-seal.json')
    need(seal['all_three_adaptations_frozen'] and seal['approved'],'No final before all3 complete')
    for v in seal['artifacts']:need(sha(Path(v['path']))==v['sha256'],'Same sealed mappings and sources')
    identity_path=a.run_root/'rsi-execution-identity.json';identity=read(identity_path)
    verify_identity(identity,{'paired':prior/'MultiAgent-Patrol-Paired-v1.json','rsi':prior/'RSI-Controlled-Meta-v1.json',
        'actor0':ACTORS/'history-seed-0/selected.pt','actor1':ACTORS/'history-seed-1/selected.pt',
        'actor2':ACTORS/'history-seed-2/selected.pt','CVAE':CVAE,
        'CVAE_manifest':source/'research/2026-10-28-m111-coverage/frozen_estimator.json'})
    indexed={v['path']:v['sha256'] for v in seal['artifacts']}
    required=[identity_path,a.run_root/'rsi-resource-before-final.json']+[a.run_root/f'rsi-trial-{trial}'/name for trial in range(3) for name in ('adapt-validation-complete.json','validation-mapping-seal.json')]
    need(all(indexed.get(str(path))==sha(path) for path in required),'All3actual mapping seals plus identity present and unchanged')
    for trial in range(3):
        receipt=read(a.run_root/f'rsi-trial-{trial}'/'adapt-validation-complete.json')
        need(receipt['complete'] and receipt['execution_identity_SHA256']==sha(identity_path),'All three adaptations same-source sealed before any final')
        validate_mapping(receipt,read(a.run_root/f'rsi-trial-{trial}'/'validation-mapping-seal.json'),sha(identity_path),identity['RSI_SHA256'])
    root=a.run_root/f'rsi-trial-{a.trial}'
    write(root/'final-started.json',{'seal_SHA256':sha(a.run_root/'rsi-final-seal.json'),'trial':a.trial,'no_model_calls':True})
    with mp.get_context('spawn').Pool(1,initializer=init_worker,initargs=(str(source),)) as pool:
        store=Store(root,source,pool,read(a.run_root/'historical-identities.json'),resume=True)
        mappings=read(root/'validation-mapping-seal.json')['programs'];groups={};results={}
        for name in GROUPS:
            pg=PatrolProgram.from_dict(mappings[name]['actual_program'])
            value=store.cache('fresh_task_final').evaluate(cases('fresh_task_final',paired,rsi),pg);groups[name]=value
            results[name]=macro(store.rows(value))
        write(root/'final-complete.json',{'complete':True,'trial':a.trial,'actual_program_mappings':mappings,'groups':groups,'statistics':results,
            'new_final_actual_cases':[v for v in store.index.values() if v['specification']['phase']=='fresh_task_final'],
            'final_cost':store.ledger()['fresh_task_final'],'seal_SHA256':sha(a.run_root/'rsi-final-seal.json'),
            'real_LLM_calls':0,'final_selected_replacements':0})
    print('M12D_RSI_FINAL_COMPLETE',a.trial,flush=True)


if __name__=='__main__':main()
