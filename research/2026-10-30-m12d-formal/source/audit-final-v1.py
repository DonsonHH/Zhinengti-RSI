"""Independent arithmetic and immutable-record audit; never exports final rows."""
import argparse
import gzip
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

GROUPS=('Tour','human_fixed','random_search','fixed3','fixed13','controlled_meta')


def read(path):return json.loads(path.read_bytes())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def need(value,message):
    if not value:raise AssertionError(message)
def same(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):return math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-11)
    return a==b


def macro(rows):
    result=[]
    for seed in sorted({v['specification']['seed'] for v in rows}):
        processes={}
        own=[v for v in rows if v['specification']['seed']==seed]
        for row in own:processes.setdefault(row['process_fingerprint'],[]).append(row)
        need(len(processes)==9,'Nine physical processes per seed')
        misses=[];wait=[]
        for pair in processes.values():
            need(len(pair)==2 and {v['specification']['noise_c'] for v in pair}=={.2,1.},'Exactly paired noises')
            misses.append(statistics.mean(v['metrics']['missed_segments'] for v in pair))
            values=[v['metrics']['consequence_weighted_censored_delay_s'] for v in pair]
            need((values[0] is None)==(values[1] is None),'Shared physical event eligibility')
            if values[0] is not None:wait.append(statistics.mean(values))
        result.append({'seed':seed,'macro_missed_count':statistics.mean(misses),'macro_wait_s':statistics.mean(wait)})
    return {'seeds':result,'macro_missed_count':statistics.mean(v['macro_missed_count'] for v in result),
        'macro_wait_s':statistics.mean(v['macro_wait_s'] for v in result),'conditions':len(rows),
        'missed_noise_repeated_sum':sum(v['metrics']['missed_segments'] for v in rows)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();r=a.run_root
    seal=r/'rsi-final-seal.json';sealed=read(seal);need(sealed['approved'] and sealed['all_three_adaptations_frozen'],'All3 pre-final mappings sealed')
    for artifact in sealed['artifacts']:need(sha(Path(artifact['path']))==artifact['sha256'],'All sealed sources, models, mappings and cost receipts unchanged at final audit')
    final_processes=set();nonfinal_processes=set();telemetry={};checks=[];records=0
    summaries=[];claimed_jobs=set();private_receipts=[];resource=read(r/'rsi-resource-before-final.json')
    need(sha(Path(resource['guarded_final_wrapper']['path']))==resource['guarded_final_wrapper']['sha256'],'Same externally sealed final guard bytes')
    for i in range(3):
        root=r/f'rsi-trial-{i}';d=read(root/'final-complete.json');ad=read(root/'adapt-validation-complete.json')
        claim=read(root/'final-one-shot-claim/claim.json')
        need(claim['trial']==i and claim['resolved_run_root']==str(r.resolve()) and claim['seal_SHA256']==sha(seal) and claim['guard_wrapper_SHA256']==resource['guarded_final_wrapper']['sha256'],'Unique claimed final launch matches run, trial, source and global seal')
        need(claim['persistent_no_retry'] and claim['claimed_before_final_source_access'] and claim['Slurm_job_id'] not in claimed_jobs,'Persistent unique one-shot claim per trial')
        claimed_jobs.add(claim['Slurm_job_id'])
        submission=read(root/'final-submission-claim/receipt.json')
        need(submission['Slurm_job_id']==claim['Slurm_job_id'] and submission['trial']==i and submission['seal_SHA256']==sha(seal) and submission['resolved_run_root']==str(r.resolve()),'Same actually submitted and claimed final allocation')
        need(submission['script_verified_while_job_held_before_release'] and sha(Path(submission['actual_batch_script_path']))==submission['actual_batch_script_SHA256']==resource['guarded_final_wrapper']['sha256'],'Actual Slurm executing bytes verified before release')
        private_receipts.append({'path':str(root/'final-complete.json'),'sha256':sha(root/'final-complete.json')})
        need(d['complete'] and d['real_LLM_calls']==d['final_selected_replacements']==0,'Actual fixed final, no new calls or replacements')
        need(d['seal_SHA256']==read(root/'final-started.json')['seal_SHA256']==sha(seal),'Actual same seal at final start and completion')
        need(d['actual_program_mappings']==read(root/'validation-mapping-seal.json')['programs'],'No changed mapping after final')
        need(set(d['groups'])==set(GROUPS),'Exactly six logical groups')
        search_path=root/'independent-search-audit.json';search=read(search_path)
        expected={v['path']:v['sha256'] for v in resource['independent_search_audits']}
        need(search['passed'] and search['adaptation_receipt_SHA256']==sha(root/'adapt-validation-complete.json') and expected[str(search_path)]==sha(search_path),'All actual search evidence matches resource-sealed audit and adaptation bytes')
        for group in GROUPS:
            nominal=ad['programs'][group]['nominal_program']
            validation_rows=[v for v in ad['case_index'] if v['specification']['phase']=='fresh_task_validation' and v['workflow']==nominal]
            need(len(validation_rows)==36 and same(macro(validation_rows),ad['validation'][group]),'Independent all-group validation macro arithmetic')
        index={v['record_path']:v for v in d['new_final_actual_cases']}
        need(len(index)==d['final_cost']['successful_episodes']==d['final_cost']['attempted_episodes']<=432,'Actual charged final episodes')
        nonfinal_processes.update(v['process_fingerprint'] for v in ad['case_index'])
        for path,row in index.items():
            need(sha(Path(path))==row['record_SHA256'] and row['audit']['passed'],'Unchanged independently audited final record')
            raw=json.loads(gzip.decompress(Path(path).read_bytes()));records+=1
            need(raw['specification']==row['specification'] and raw['workflow']==row['workflow'] and raw['metrics']==row['metrics'],'Index equals actual final execution')
            intervals=raw['anomaly_intervals'];weight=sum(v['weight'] for v in intervals)
            wait=sum(v['weight']*v['delay_or_censor_s'] for v in intervals)/weight if weight else None
            need(same(wait,row['metrics']['consequence_weighted_censored_delay_s']),'Raw consequence-weighted censored waiting')
            need(sum(not v['detected'] for v in intervals)==row['metrics']['missed_segments'],'Raw missed event count')
            need(row['metrics']['tasks_violating_exact_gap']==row['metrics']['invalid_assignments']==0 and row['metrics']['final_professional_coverage']==1.,'All fixed execution safety metrics')
            need(row['usage']['global_optional_checks']<=1 and row['audit']['initial_terminal_revisit_verified'] and row['audit']['physical_process_unchanged'],'Quota, head-tail obligations and action-independent physics')
            need(row['specification']['phase']=='fresh_task_final' and row['specification']['seed'] in (32140,32141,32142,32143),'Only frozen fresh final cohort')
            final_processes.add(row['process_fingerprint'])
            key=(row['process_fingerprint'],row['specification']['noise_c']);old=telemetry.setdefault(key,row['public_telemetry_sequence_SHA256'])
            need(old==row['public_telemetry_sequence_SHA256'],'All methods/repeats share physical process and observation noise')
        for group in GROUPS:
            rows=[index[path] for path in d['groups'][group]['case_ids']]
            need(len(rows)==72 and len({v['specification']['seed'] for v in rows})==4,'Four seeds times nine processes times two noises per group')
            need(all(v['workflow']==d['actual_program_mappings'][group]['actual_program'] for v in rows),'Actual sealed program executed')
            need(same(macro(rows),d['statistics'][group]),'Independent final seed-process-noise macro')
            checks.append({'trial':i,'group':group,'conditions':72,'passed':True})
        summaries.append({'trial':i,'actual_final_environment_episodes':len(index),'logical_final_conditions':432,
            'cache_hits':d['final_cost']['cache_hits'],'final_statistics':d['statistics']})
    need(final_processes.isdisjoint(nonfinal_processes) and len(final_processes)==36,'Thirty-six new numeric final processes, no previous task reuse')
    accounting=subprocess.check_output(['sacct','-X','-j',','.join(sorted(claimed_jobs)),'--format=JobID,State,ExitCode,ElapsedRaw,AllocCPUS','-P'],text=True)
    charged=[line.split('|') for line in accounting.strip().splitlines()[1:]]
    need({v[0] for v in charged}==claimed_jobs and all(v[1:3]==['COMPLETED','0:0'] and int(v[4])==2 for v in charged),'All three actual final launch claims joined to completed charged Slurm allocations')
    report={'passed':True,'checks':checks,'raw_actual_records_rehashed_and_arithmetic_checked':records,
        'unique_final_physical_processes':len(final_processes),'generation_repeats':3,'independent_final_seed_groups':4,
        'model_calls_in_this_audit':0,'environment_episodes_in_this_audit':0,'raw_final_rows_exported':False,
        'whole_neural_execution_replayed':False,'scope':'Independent raw record/hash, metric arithmetic, program mapping and existing route/event/certificate audits; no second neural execution.',
        'trial_summaries':summaries,'private_final_receipts':private_receipts,'global_final_seal_SHA256':sha(seal),
        'actual_final_Slurm_accounting_psv':accounting,
        'claimed_final_Slurm_jobs':sorted(claimed_jobs),'five_arm_final_unopened':not(r/'final_test').exists()}
    (r/'rsi-independent-final-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('M12D_RSI_INDEPENDENT_FINAL_AUDIT',records,len(checks),flush=True)


if __name__=='__main__':main()
