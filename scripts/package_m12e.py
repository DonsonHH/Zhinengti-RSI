"""Publish E0 originals and E1 development summaries; keep bulky arrays private."""
import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path


def read(p):return json.loads(p.read_bytes())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.run_root;o=a.output;o.mkdir(exist_ok=False)
    need=read(r/'E1-independent-audit.json');assert need['passed'] and not(r/'E1-STOP.json').exists()
    e0=read(r/'E0/complete.json');assert e0['complete'] and e0['real_model_calls']==24
    ids=sorted({re.search(r'-(\d+)\.(?:out|err)$',x.name).group(1) for x in (r/'jobs').iterdir() if re.search(r'-(\d+)\.(?:out|err)$',x.name)},key=int)
    raw=subprocess.check_output(['sacct','-X','-j',','.join(ids),'--format=JobID,JobName%48,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,Start,End','-P'],text=True)
    memory=subprocess.check_output(['sacct','-j',','.join(ids),'--format=JobID,State,ExitCode,MaxRSS,AveRSS,MaxVMSize','-P'],text=True)
    jobs=list(csv.DictReader(io.StringIO(raw),delimiter='|'));assert {v['JobID'] for v in jobs}==set(ids) and all(v['State']=='COMPLETED' and v['ExitCode']=='0:0' for v in jobs)
    cpu=sum(int(v['ElapsedRaw'])*int(v['AllocCPUS'])/60 for v in jobs);gpu=[v for v in jobs if 'gres/gpu=' in v['AllocTRES']]
    assert len(gpu)==1 and int(gpu[0]['ElapsedRaw'])<=1800 and cpu<=960
    assert {read(r/'E1-submission-claims'/f'seed-{seed}/receipt.json')['job_id'] for seed in (33000,33001)}<=set(ids)
    assert read(r/'E0-submission-claim/receipt.json')['job_id']==gpu[0]['JobID']
    episodes=sum(read(r/'E1'/f'seed-{seed}/complete.json')['actual_total_episode_attempts'] for seed in (33000,33001));assert episodes==need['actual_total_episodes']<=252
    allsize=sum(p.stat().st_size for p in r.rglob('*') if p.is_file());e0size=sum(p.stat().st_size for folder in (r/'E0',r/'code-v1',r/'gate-v1') for p in folder.rglob('*') if p.is_file())
    assert allsize<3_000_000_000 and e0size<500_000_000
    ledger={'all_allocations_completed_exit0':True,'allocation_count':len(jobs),'jobs':jobs,'allocated_CPU_minutes_inclusive':cpu,'formal_GPU_allocations':1,'GPU_seconds':int(gpu[0]['ElapsedRaw']),
        'actual_new_model_calls':24,'actual_E1_main_episodes':180,'actual_E1_witness_episodes':episodes-180,'actual_E1_environment_episodes':episodes,
        'actual_E1_physical_processes':18,'new_neural_training_runs':0,'old_final_raw_access':False,'E2_calls_or_episodes':0,
        'source_storage_bytes_before_export':allsize,'E0_source_storage_bytes':e0size,'Slurm_memory_samples':list(csv.DictReader(io.StringIO(memory),delimiter='|')),
        'accounting_scope':'AllnewSlurmallocations including both316/317gates, E0realGPU, E1main/witness andindependentaudit. Storageexcludesinheritedmodelweights.'}
    write(r/'resource-ledger.json',ledger);(r/'resource-sacct.psv').write_text(raw,encoding='utf-8');(r/'resource-memory.psv').write_text(memory,encoding='utf-8')
    files=[]
    for folder in ('E0','gate-v1','gate-v2','E0-submission-claim','E1-submission-claims','jobs'):files.extend(p for p in (r/folder).rglob('*') if p.is_file())
    for name in ('E0-launch-review.json','E1-launch-review.json','E0-launch-conditions.json','E1-launch-conditions.json','E0-execution-identity.json','E1-execution-identity.json',
        'historical-identities.json','E1-independent-audit.json','resource-ledger.json','resource-sacct.psv','resource-memory.psv','m12e_schema.sbatch','m12e_gate.sbatch','m12e_gate_v2.sbatch','m12e_timing.sbatch','m12e_timing_audit.sbatch','audit_m12e_timing.py'):
        files.append(r/name)
    for seed in (33000,33001):
        root=r/'E1'/f'seed-{seed}'
        files.extend(root/name for name in ('started.json','complete.json'));files.extend((root/'diagnostics').glob('*.json'));files.extend((root/'sources').glob('*/complete.json'));files.extend((root/'sources').glob('*--generation-attempt.json'))
        for original in (root/'public').glob('*--decisions-sealed.json'):
            d=read(original);release={k:v for k,v in d.items() if k!='candidate_public'}
            release.update(original_private_path=str(original),original_private_SHA256=sha(original),public_candidate_count=len(d['candidate_public']),
                large_public_candidate_rows_exported=False,public_trace_private_path=str(original.with_name(original.name.replace('--decisions-sealed.json','.jsonl.gz'))))
            dest=o/'E1'/f'seed-{seed}'/'public-release'/original.name.replace('--decisions-sealed','--public-selections-release');dest.parent.mkdir(parents=True,exist_ok=True);write(dest,release)
    for original in files:
        dest=o/original.relative_to(r);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(original.read_bytes())
    inventory=[{'path':p.relative_to(o).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in o.rglob('*') if p.is_file()]
    write(o/'export-inventory.json',{'files':inventory,'private_arrays_raw_episode_files_or_large_public_vectors_exported':False})
    archive=o.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as t:
        for p in o.rglob('*'):
            if p.is_file():t.add(p,arcname=p.relative_to(o).as_posix(),recursive=False)
    print('M12E_FINAL_PUBLIC_EXPORT',len(files),len(inventory),archive.stat().st_size,sha(archive),flush=True)


if __name__=='__main__':main()
