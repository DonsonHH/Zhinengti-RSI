"""Reconcile actual Slurm allocations and disjoint generation/audit budgets."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from patrol_rsi.fresh_opportunity import frozen_inputs, need, read, write, sha


def clock_seconds(value):
    days,clock=(value.split('-',1) if '-' in value else ('0',value))
    parts=list(map(float,clock.split(':')))
    return int(days)*86400+sum(v*60**i for i,v in enumerate(reversed(parts)))


def rss_bytes(value):
    if not value:return 0
    unit=value[-1].upper()
    return float(value[:-1])*{'K':1024,'M':1024**2,'G':1024**3,'T':1024**4}[unit] if unit in 'KMGT' else float(value)


def verified_interval(row):
    need(row.get('Start') not in (None,'','None','Unknown') and row.get('End') not in (None,'','None','Unknown'),
        'Executed allocation requires real Start and End; missing is not zero concurrency')
    start,end=datetime.fromisoformat(row['Start']),datetime.fromisoformat(row['End'])
    seconds=(end-start).total_seconds()
    need(seconds>=0 and abs(seconds-float(row['ElapsedRaw']))<=1.,'Actual allocation timestamps agree with elapsed duration')
    return start,end


def review(ledger_path,data_root,protocol,manifest):
    rows=list(csv.DictReader(ledger_path.read_text().splitlines(),delimiter='|'))
    allocations=[v for v in rows if '.' not in v['JobID']]
    generated={}
    for item in manifest:
        key=(item['split'],item['group_seed'])
        generated[key]=data_root/item['split']/f"seed-{item['group_seed']}"
    total=gen=audit=actual=0.;completed=[];cancelled=[];seen_groups=set(); intervals=[]
    for row in allocations:
        state=row['State'];elapsed=float(row['ElapsedRaw']);cpus=int(row['AllocCPUS'] or 0)
        if state.startswith('CANCELLED'):
            need(elapsed==0 and row['JobID'].split('_')[0] in ('130648','130652','130660','130661','130662','130663','130683','130688','130713','130714'), 'Only documented not-started export/audit-version allocations cancelled')
            cancelled.append(row);continue
        need(state=='COMPLETED' and row['ExitCode']=='0:0','Every executed allocation completed0:0')
        need(row['Partition']=='cu' and cpus==2 and elapsed<=3600 and 'gres/gpu' not in row['AllocTRES'],'Frozen CPU partition/resources/time, no GPU')
        need(row['ReqMem'] in ('8Gn','8192Mn','8G','8192M'),'Frozen8GB allocation memory')
        charge=elapsed*cpus/60;total+=charge
        actual+=clock_seconds(row['TotalCPU'])
        generation_s=0.
        if row['JobName']=='rsi-b4b1a-gen':
            matches=[]
            for key,folder in generated.items():
                start=read(folder/'run-start.json')
                if start['job_id'] in (row['JobID'],row['JobIDRaw']) or start['job_id']==row['JobID'].split('_')[0] and str(start['array_task_id'])==row['JobID'].split('_')[-1]:
                    matches.append((key,folder))
            need(len(matches)==1,'Every actual generation allocation bound to one exact frozen seed')
            key,folder=matches[0];need(key not in seen_groups,'No duplicate generated seed group')
            seen_groups.add(key);done=read(folder/'complete.json')
            need(done['conditions']==20 and len(set(done['completed_condition_ids']))==20,'All exact20 conditions complete')
            generation_s=min(elapsed,done['wall_s']);gen+=generation_s*cpus/60
        audit+=charge-generation_s*cpus/60
        completed.append(dict(row,allocated_CPU_minutes=charge,generation_CPU_minutes=generation_s*cpus/60,
            audit_and_overhead_CPU_minutes=charge-generation_s*cpus/60))
        start,end=verified_interval(row)
        intervals.append((start,1));intervals.append((end,-1))
    need(len(seen_groups)==64,'All64 exact generation groups present in completed Slurm ledger')
    concurrency=peak=0
    for _,change in sorted(intervals,key=lambda v:(v[0],v[1])):
        concurrency+=change;peak=max(peak,concurrency)
    need(peak<=2,'Frozen maximum2 simultaneous allocations')
    need(gen<=protocol['resources']['generation_max_allocated_CPU_minutes'] and audit<=protocol['resources']['evaluation_and_audit_max_allocated_CPU_minutes'], 'Separate generation2880 and audit480 actual allocated CPU-minute caps')
    need(abs(total-gen-audit)<1e-8,'Allocation charges partition exactly, no hidden or double counted CPU')
    return {'status':'passed','ledger_SHA256':sha(ledger_path),'allocations':len(completed),'cancelled_unstarted_records':cancelled,
        'completed_allocations':completed,'allocated_CPU_minutes':total,'generation_allocated_CPU_minutes':gen,
        'audit_overhead_allocated_CPU_minutes':audit,'generation_cap_CPU_minutes':2880,'audit_cap_CPU_minutes':480,
        'measured_TotalCPU_seconds':actual,'peak_RSS_bytes':max((rss_bytes(v['MaxRSS']) for v in rows),default=0),
        'maximum_concurrent_allocations':peak,'GPU_allocations':0,'generated_groups':64,
        'charging_rule':'Within combined generation+verification allocation, recorded generation wall_s gets its2 allocated CPUs; remaining elapsed and every gate/recheck/report allocation count as audit/overhead. Charges sum to actual Slurm total.',
        'predictor_fits':0,'PPO_updates':0,'CVAE_updates':0,'LLM_calls':0}


def main():
    p=argparse.ArgumentParser()
    for n in ('ledger','data-root','protocol','manifest','output'):
        p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args();protocol,manifest=frozen_inputs(args.protocol,args.manifest)
    result=review(args.ledger,args.data_root,protocol,manifest)
    write(args.output,result)
    print('FRESH_RESOURCES_PASSED',result['allocated_CPU_minutes'],result['audit_overhead_allocated_CPU_minutes'],flush=True)


if __name__=='__main__':main()
