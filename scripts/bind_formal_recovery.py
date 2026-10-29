"""Historical source/log/filesystem-clock binding for failed audit130534.

No pre-failure cryptographic checkpoint is invented. The proof requires
unchanged NFS ctime/mtime before the first dependency-permitted Slurm start.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import tarfile
import time

p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);args=p.parse_args();root=args.run_root
env=dict(os.environ,SLURM_TIME_FORMAT='%s')
raw=subprocess.check_output(['sacct','-X','--duplicates','-j','130532,130534','--format=JobID%30,Submit,Start,End,State,ExitCode','-P'],env=env).decode()
records=list(csv.DictReader(io.StringIO(raw),delimiter='|'));lookup={r['JobID']:r for r in records}
if len(records)!=4 or len(lookup)!=4:raise RuntimeError('Duplicate allocation/requeue record prevents bounded recovery')
dependency_end=max(int(lookup['130532_'+str(i)]['End']) for i in (0,1))
manifest=json.loads((root/'code-v3-manifest.json').read_bytes());archive=root/manifest['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=manifest['sha256']:raise RuntimeError('Old archive differs')
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        if member.isfile():
            data=tar.extractfile(member).read();row=manifest['files'][member.name]
            if hashlib.sha256(data).hexdigest()!=row['sha256'] or data!=(root/'code-v3'/member.name).read_bytes():
                raise RuntimeError('Old execution source differs from archived bytes')
for chunk in (0,1):
    job=lookup['130534_'+str(chunk)];start=int(job['Start']);end=int(job['End']);folder=root/f'development-{chunk}'
    if job['State']!='FAILED' or job['ExitCode']!='1:0' or not 0<=start-dependency_end<=1:
        raise RuntimeError('Failed job was not its first dependency-permitted allocation')
    stdout=root/f'jobs/development-audit-130534_{chunk}.out';stderr=root/f'jobs/development-audit-130534_{chunk}.err'
    lines=stdout.read_text().splitlines()
    if lines.count(f'FORMAL_AUDIT development {chunk} 8')!=1 or lines.count(f'FORMAL_AUDIT development {chunk} 16')!=1 or len(lines)!=2:
        raise RuntimeError('Repeated or unexpected prefix log; cannot establish one replay per prefix case')
    rows=json.loads((folder/'case-index.json').read_bytes());models=json.loads((folder/'model-inputs.json').read_bytes())
    paths=list(folder.glob('*.json'))+[folder/r['path'] for r in rows[:21]]
    paths += [folder/r['path'] for r in json.loads((folder/'process-index.json').read_bytes()).values()]
    paths += [root/'code-v3'/name for name in manifest['files']]
    paths += [root/'code-v3-manifest.json',archive]
    paths += [Path(r['path']) for r in models['actors'].values()]
    paths += [Path('/share/home/renly/renly-patrol-rsi/runs/20261028-m111/frozen-estimator.pt')]
    bindings=[]
    for path in sorted(set(paths)):
        info=path.lstat()
        if not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_ctime_ns>start*10**9 or info.st_mtime_ns>start*10**9:
            raise RuntimeError('Input changed since/beyond failed-audit start: '+str(path))
        bindings.append({'path':str(path),'size':info.st_size,'ctime_ns':info.st_ctime_ns,'mtime_ns':info.st_mtime_ns,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    logs=[]
    for path in (stdout,stderr):
        info=path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_ctime_ns>(end+2)*10**9:raise RuntimeError('Historical failure log changed')
        logs.append({'path':str(path),'size':info.st_size,'ctime_ns':info.st_ctime_ns,'mtime_ns':info.st_mtime_ns,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    result={'job':job,'dependency_latest_end_epoch':dependency_end,'allocation_history':records,
        'allocation_history_raw':raw,'files':bindings,'logs':logs,'created_unix':time.time(),
        'scope':'Historical filesystem-clock/source/log evidence conditional on consistent Slurm/NFS clocks; not a cryptographic checkpoint captured before the failure.',
        'no_requeue_evidence':'One allocation per task, original Submit, start within1s of the first permitted dependency end, one progress prefix only.'}
    output=root/f'development-{chunk}-historical-binding.json'
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('HISTORICAL_BINDING',chunk,len(bindings),flush=True)
