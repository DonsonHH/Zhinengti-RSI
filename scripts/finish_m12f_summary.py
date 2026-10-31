"""Produce a final summary only from completed, independently audited F stages."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


def read(path):return json.loads(path.read_bytes())
def need(value,text):
    if not value:raise ValueError(text)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=a.run_root;need(not(r/'F-STOP.json').exists(),'A runtime/source/safety STOP prevents final completion')
    proof=read(r/'F0/complete.json');search=read(r/'F1/search/complete.json');validation=read(r/'F1/validation/complete.json')
    need(proof['complete'] and search['complete'] and validation['complete'],'Actual F0/F1 completion')
    audits={}
    def audit(phase):
        for kind in ('actual','prompts'):
            path=r/f'{phase}-independent-{kind}-audit.json';value=read(path)
            need(value['passed'],phase+' independent '+kind+' audit')
            audits[path.name]={'SHA256':sha(path),'内容':value}
    audit('F1')
    final=None
    if validation['final_open']:
        final=read(r/'F1/final/complete.json');need(final['complete'],'Opened F1 final must finish')
    else:need(not(r/'F1/final').exists() and not(r/'future-sources/F1_final').exists(),'F1 final remains ungenerated/unopened')
    f1_episodes=search['evaluation_ledger']['attempted_episode_reservations']+validation['evaluation_ledger']['attempted_episode_reservations']
    if final:f1_episodes+=final['evaluation_ledger']['attempted_episode_reservations']
    f1={'generation_repeats':3,'real_model_calls':search['model_calls'],'actual_environment_evaluations':f1_episodes,
        'mechanism_gate':search['mechanism_gate'],'validation_TaskGO':validation['TaskGO'],'validation_ExperienceGO':validation['ExperienceGO'],
        'final_opened':final is not None,'final_TaskGO':final['TaskGO'] if final else None,'final_ExperienceGO':final['ExperienceGO'] if final else None,
        'confirmed_TaskGO':bool(final and final['confirmed_TaskGO']),'confirmed_ExperienceGO':bool(final and final['confirmed_ExperienceGO']),
        'validation_comparisons':validation['comparisons'],'final_comparisons':final['comparisons'] if final else None}
    calls=search['model_calls'];episodes=proof['charged_known_closed_loop_episodes']+f1_episodes
    if search['mechanism_gate']['passed']:
        trials=[read(r/f'F2/trial-{trial}/adaptation/complete.json') for trial in range(3)]
        need(all(v['complete'] for v in trials),'All three conditional F2 actual trials complete')
        val2=read(r/'F2/validation/complete.json');need(val2['complete'],'F2 validation complete')
        final2=None
        if val2['MetaGO']:
            final2=read(r/'F2/final-complete.json');need(final2['complete'],'Opened F2 final must finish both batches and four-seed merge')
        else:need(not(r/'F2/final-batch-0').exists() and not(r/'F2/final-batch-1').exists() and not(r/'future-sources/F2_final').exists(),'F2 final remains ungenerated/unopened')
        audit('F2')
        f2_episodes=sum(v['meta']['ledger']['attempted_episode_reservations']+v['adapt_ledger']['attempted_episode_reservations'] for v in trials)
        f2_episodes+=sum(v['attempted_episode_reservations'] for v in val2['ledgers'])
        if final2:
            for batch in (0,1):f2_episodes+=sum(v['attempted_episode_reservations'] for v in read(r/f'F2/final-batch-{batch}/complete.json')['ledgers'])
        f2_calls=sum(v['model_calls'] for v in trials);calls+=f2_calls;episodes+=f2_episodes
        f2={'executed':True,'generation_repeats':3,'real_model_calls':f2_calls,'actual_environment_evaluations':f2_episodes,
            'valid_meta_protocol_proposals':sum(v['meta']['proposal_valid'] for v in trials),'different_meta_protocol_proposals':sum(v['meta']['protocol_changed'] for v in trials),
            'retained_meta_updates':sum(v['meta']['retained'] for v in trials),'validation_MetaGO':val2['MetaGO'],
            'final_opened':final2 is not None,'final_MetaGO':final2['MetaGO'] if final2 else None,
            'confirmed_MetaGO':bool(final2 and val2['MetaGO'] and final2['MetaGO']),
            'validation_comparisons':val2['comparisons'],'final_comparisons':final2['comparisons'] if final2 else None}
    else:
        need(not(r/'F2').exists(),'Mechanism-closed F2 must have no experiment outputs')
        f2={'executed':False,'reason':'Registered F1 mechanism gate did not pass','real_model_calls':0,'actual_environment_evaluations':0,
            'confirmed_MetaGO':False,'scientific_interpretation':'No meta-effect experiment; not evidence against RSI in principle.'}
    need(read(r/'cost/model_calls-counter.json')['reserved']==calls<=75,'Actual all-phase model call budget')
    need(read(r/'cost/episodes-counter.json')['reserved']==episodes<=6000,'Actual all-phase environment budget including charged F0')
    accounting=list(csv.DictReader((r/'resource-sacct-final.psv').open(),delimiter='|'))
    need(accounting and len({v['JobID'] for v in accounting})==len(accounting),'Unique allocation-level accounting')
    terminal={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'}
    need(all(v['State'].split()[0] in terminal for v in accounting),'All registered allocations must be terminal')
    by_job={v['JobID']:v for v in accounting}
    producers=[str(read(path)['job_id']) for path in (r/'launches').glob('*.json')]
    need(all(job in by_job and by_job[job]['State']=='COMPLETED' and by_job[job]['ExitCode']=='0:0' for job in producers),'Every data-producing campaign allocation must complete successfully')
    cpu=sum(int(v['ElapsedRaw'])*int(v['AllocCPUS'])/60 for v in accounting)
    gpu=sum(int(v['ElapsedRaw'])*int(re.search(r'gres/gpu=(\d+)',v['AllocTRES']).group(1)) for v in accounting if 'gres/gpu=' in v['AllocTRES'])
    gpu_jobs=sum('gres/gpu=' in v['AllocTRES'] for v in accounting)
    need(cpu<=2400 and gpu<=21600 and gpu_jobs<=4,'Total preparation/execution/audit allocation caps')
    inventory=read(r/'actual-run-inventory.json');need(inventory['new_storage_bytes']<=6000000000,'Actual new storage cap')
    costs={'all_allocations':accounting,'failed_preparation_or_audit_allocations':sum(v['State']!='COMPLETED' or v['ExitCode']!='0:0' for v in accounting),
        'allocated_CPU_core_minutes':cpu,'allocated_GPU_seconds':gpu,'GPU_allocations':gpu_jobs,
        'actual_model_calls':calls,'actual_environment_evaluations_including_F0':episodes,'storage_bytes':inventory['new_storage_bytes'],
        'cost_scope':'Includes all Slurm gates, tokenizer-only preparations, engineering parity, searches, confirmation and completed audit allocations. Local authoring/checks are separate and not claimed zero-compute.'}
    value={'bounded_stage_complete':True,'stage':'M12-F','source_version':'v6','F0':{'known_episodes':20,'candidate_comparisons':298152,'mismatches':0,'all_gold_trajectories_and_metrics_equal':True},
        'F1':f1,'F2':f2,'actual_costs':costs,'independent_audits':audits,
        'old_E2_NO_GO_preserved':True,'old_final_raw_used_for_new_search':False,'new_neural_training':False,
        'claims_separate':['Task improvement','Combined causal-experience and acceptance-feedback increment','Improver-method/recursive increment'],
        'future_manifest_is_not_actual_data':True,'inventory':inventory}
    with a.output.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'summary':str(a.output),'bounded_stage_complete':True,'F1_confirmed_task':f1['confirmed_TaskGO'],
        'F1_confirmed_experience':f1['confirmed_ExperienceGO'],'F2_executed':f2['executed'],'F2_confirmed_meta':f2['confirmed_MetaGO'],
        'model_calls':calls,'environment_evaluations':episodes}),flush=True)


if __name__=='__main__':main()
