"""Account every M12D Slurm allocation, including failed setup, from sacct."""
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path


def read(path):return json.loads(path.read_bytes())
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def need(value,message):
    if not value:raise AssertionError(message)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--before-final',action='store_true');a=p.parse_args();r=a.run_root
    ids={n for f in (r/'jobs').iterdir() for n in re.findall(r'(?<!\d)(130\d{3})(?!\d)',f.name)}
    actual_final_jobs={read(p)['Slurm_job_id'] for p in r.glob('rsi-trial-*/final-one-shot-claim/claim.json')}
    submitted_final_jobs={read(p)['Slurm_job_id'] for p in r.glob('rsi-trial-*/final-submission-claim/receipt.json')}
    ids=sorted(ids|actual_final_jobs|submitted_final_jobs,key=int)
    if not a.before_final:need(len(actual_final_jobs)==3 and actual_final_jobs==submitted_final_jobs,'Exactly three submitted and claimed final allocations charged')
    raw=subprocess.check_output(['sacct','-X','-j',','.join(ids),'--format=JobID,JobName%48,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,Start,End','-P'],text=True)
    rss_raw=subprocess.check_output(['sacct','-j',','.join(ids),'--format=JobID,State,ExitCode,MaxRSS,AveRSS,MaxVMSize','-P'],text=True)
    batch_RSS=[v for v in csv.DictReader(io.StringIO(rss_raw),delimiter='|') if v['JobID'].endswith('.batch')]
    jobs=list(csv.DictReader(io.StringIO(raw),delimiter='|'));need({v['JobID'] for v in jobs}==set(ids),'All actual logged job allocations in sacct')
    expected_failures={'130883','130890','130911'};formal_gpu={'130897','130900','130901'};preflight_gpu={'130883','130885'}
    events=[];CPU=GPU=preflight_seconds=five_arm_CPU=0;gpu_jobs=[]
    for row in jobs:
        need(row['State'] in ('COMPLETED','FAILED'),'Every allocation terminal before cost seal')
        if row['JobID'] in expected_failures:need(row['State']=='FAILED' and row['ExitCode']=='1:0','Preserved known setup/audit infrastructure failure')
        else:need(row['State']=='COMPLETED' and row['ExitCode']=='0:0','No formal execution or audit failure')
        seconds=int(row['ElapsedRaw']);cpu=int(row['AllocCPUS']);CPU+=seconds*cpu/60
        gpu=int(re.search(r'(?:^|,)gres/gpu=(\d+)',row['AllocTRES']).group(1)) if 'gres/gpu=' in row['AllocTRES'] else 0
        if gpu:
            gpu_jobs.append(row['JobID']);need(gpu==1,'One GPU per allocation')
            need(row['JobID'] in formal_gpu|preflight_gpu,'No unregistered fourth formal allocation')
            if row['JobID'] in formal_gpu:
                need(seconds<=3600,'At most one hour per formal GPU allocation');GPU+=seconds
            else:preflight_seconds+=seconds
        if row['JobID'] in {'130882','130884','130886','130887','130888','130889','130892','130893','130894','130898'}:five_arm_CPU+=seconds*cpu/60
        start=dt.datetime.fromisoformat(row['Start']);end=dt.datetime.fromisoformat(row['End'])
        if end>start:events.extend([(start,1,gpu==0),(end,-1,gpu==0)])
    active=activeCPU=peak=peakCPU=0
    for _,change,is_cpu in sorted(events,key=lambda v:(v[0],v[1])):
        active+=change;activeCPU+=change if is_cpu else 0;peak=max(peak,active);peakCPU=max(peakCPU,activeCPU)
    need(set(gpu_jobs)==formal_gpu|preflight_gpu and GPU<=10800 and preflight_seconds<=600,'Exactly three formal GPU allocations and one separately budgeted preflight generation')
    need(CPU<=2400 and five_arm_CPU<=960 and peakCPU<=2,'Conservative inclusive CPU accounting within both study caps')
    call_count=input_tokens=output_tokens=env=0;audits=[];trial_cost=[]
    for i in range(3):
        root=r/f'rsi-trial-{i}';d=read(root/'adapt-validation-complete.json');audit=root/'independent-search-audit.json'
        need(d['complete'] and read(audit)['passed'],'All three real search audits complete')
        n=len(list((root/'calls').glob('*.start.json')))
        need(n==len(list((root/'calls').glob('*.result.json')))==len(d['model_call_receipts']) and n<=26,'Every formal call charged and completed within per-repeat cap')
        call_count+=n;input_tokens+=d['allcalls_input_tokens'];output_tokens+=d['allcalls_output_tokens']
        cost=sum(v['attempted_episodes'] for v in d['costs'].values())
        final=root/'final-complete.json'
        if a.before_final:need(not final.exists() and not(root/'final-started.json').exists(),'No final opened before resource gate')
        else:
            fd=read(final);need(fd['complete'],'All final episodes complete');cost+=fd['final_cost']['attempted_episodes']
        env+=cost;trial_cost.append({'trial':i,'environment_attempts':cost,'model_calls':n})
        audits.append({'path':str(audit),'sha256':sha(audit)})
    need(call_count<=78 and input_tokens<=479232 and output_tokens<=59904 and env<=6480,'Frozen formal call, token and episode budgets')
    size=sum(p.stat().st_size for p in r.rglob('*') if p.is_file());need(size<4_000_000_000,'Complete run directory below RSI source cap')
    reserved_env=1296 if a.before_final else 0;reserved_CPU=500 if a.before_final else 0;reserved_bytes=512_000_000 if a.before_final else 0
    need(env+reserved_env<=6480 and CPU+reserved_CPU<=2400 and size+reserved_bytes<4_000_000_000,'Remaining fixed final allocation fits caps')
    guard=r/'m12d_guarded_final.sbatch';need(guard.is_file(),'External one-shot final launch wrapper staged')
    report={'within_caps':True,'all_formal_jobs_completed_exit0':True,'known_setup_failures_preserved':sorted(expected_failures),
        'failure_reasons':{'130883':'Missing transformers before any generation','130890':'Profile import path failed before generation or new physics','130911':'Final audit completed record checks but compute-node sacct has no accounting route; recovered with hashed live mu01 accounting receipt, no experiment rerun'},
        'allocation_count':len(jobs),'jobs':jobs,'max_concurrent_allocations':peak,'max_concurrent_CPU_only_allocations':peakCPU,
        'Slurm_batch_memory_receipts':batch_RSS,'memory_scope':'Slurm sampled MaxRSS/AveRSS per batch step; not stage-specific total RAM or CUDA peak memory.',
        'allocated_CPU_minutes_inclusive_all_M12D':CPU,'five_arm_allocated_CPU_minutes_including_regression_and_audit':five_arm_CPU,
        'formal_GPU_allocations':3,'formal_GPU_seconds':GPU,'preflight_GPU_seconds_including_failed_startup':preflight_seconds,
        'actual_formal_model_calls':call_count,'separate_actual_preflight_model_calls':1,'input_tokens':input_tokens,'output_tokens':output_tokens,
        'actual_environment_attempts':env,'trials':trial_cost,'source_storage_bytes_all_run_files':size,
        'remaining_reserved_environment_episodes':reserved_env,'remaining_reserved_CPU_minutes':reserved_CPU,'remaining_reserved_storage_bytes':reserved_bytes,
        'independent_search_audits':audits,'guarded_final_wrapper':{'path':str(guard),'sha256':sha(guard)},
        'five_arm_final_generated':(r/'final_test').exists(),'before_RSI_final':a.before_final,
        'cost_definition':'ElapsedRaw times allocated CPUs; one GPU per formal allocation. All failed setup, regression, profiling and auditing allocations included in conservative CPU total.'}
    target=r/('rsi-resource-before-final.json' if a.before_final else 'resource-ledger.json')
    need(not target.exists(),'Immutable resource receipt; never overwrite pre-final seal inputs')
    write(target,report);(r/('resource-sacct-before-final.psv' if a.before_final else 'resource-sacct.psv')).write_text(raw,encoding='utf-8')
    (r/('resource-memory-before-final.psv' if a.before_final else 'resource-memory.psv')).write_text(rss_raw,encoding='utf-8')
    if not a.before_final:write(r/'storage-final.json',{'bytes_all_new_run_files':size,'source_storage_cap_bytes':4_000_000_000,'within_cap':True,'models_external_not_copied':True})
    print('M12D_RESOURCE_GATE',a.before_final,len(jobs),GPU,CPU,env,size,flush=True)


if __name__=='__main__':main()
