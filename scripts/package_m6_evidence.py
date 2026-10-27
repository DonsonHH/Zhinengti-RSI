"""Package terminal M6 evidence without losing or modifying any JSON value."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block:=handle.read(8*1024*1024):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--code',required=True)
    p.add_argument('--ledger',type=Path,help='Fresh sacct ledger obtained on login node when compute nodes cannot reach accounting')
    a=p.parse_args();root=a.root.resolve();code=root/a.code
    ids='129810,129811,129812,129813,129814,129822,129824'
    required=['129810','129811','129812','129814','129822','129824']+[f'129813_{i}' for i in range(9)]
    ledger=None
    for attempt in range(4):
        ledger=a.ledger.read_text() if a.ledger else subprocess.check_output(['sacct','-j',ids,'--format=JobID,JobName%26,State,ExitCode,Elapsed','-P'],text=True)
        rows={r['JobID']:r for r in csv.DictReader(io.StringIO(ledger),delimiter='|') if '.' not in r['JobID']}
        if all(j in rows and rows[j]['State']=='COMPLETED' and rows[j]['ExitCode']=='0:0' for j in required):break
        if a.ledger or attempt==3:raise RuntimeError('Terminal accounting not verified for all required jobs')
        time.sleep(3)
    delivery=root/'delivery-evidence';delivery.mkdir(exist_ok=False)
    logs=delivery/'jobs';logs.mkdir();storage=[];weights={}
    for variant in ('joint_vae','no_vae','independent_targets'):
        for seed in range(3):
            source=code/f'results/{variant}-seed-{seed}';target=delivery/source.name;target.mkdir()
            summary=source/'summary.json';raw=summary.read_bytes();s=json.loads(raw)
            assert s['episodes']==1200 and not s['test_used_for_selection'] and s['fresh_checkpoint_trace_replay_matches']
            if len(raw)>=8*1024*1024:
                compressed=gzip.compress(raw,compresslevel=6,mtime=0)
                assert gzip.decompress(compressed)==raw
                destination=target/'summary.json.gz';destination.write_bytes(compressed)
            else:
                destination=target/'summary.json';destination.write_bytes(raw)
            storage.append({'original':str(summary.relative_to(root)),'delivered':str(destination.relative_to(delivery)),
                'original_bytes':len(raw),'uncompressed_sha256':hashlib.sha256(raw).hexdigest(),
                'stored_bytes':destination.stat().st_size,'stored_sha256':digest(destination),
                'storage':'lossless_gzip' if destination.suffix=='.gz' else 'json'})
            for name in ('contract.json','learning.jsonl','initial-validation.json'):shutil.copyfile(source/name,target/name)
            checkpoint=source/'best-policy.pt';sha=digest(checkpoint);assert sha==s['checkpoint_sha256']
            weights[str(checkpoint.relative_to(root))]={'bytes':checkpoint.stat().st_size,'sha256':sha}
    for phase,job in [('smoke',129811),('preflight',129812)]:
        source=code/f'results/{phase}-{job}';target=delivery/source.name;target.mkdir()
        for f in source.glob('*'):
            if f.suffix in ('.json','.jsonl'):shutil.copyfile(f,target/f.name)
        for ext in ('out','err'):shutil.copyfile(code/f'{phase}-{job}.{ext}',logs/f'{phase}-{job}.{ext}')
    for f in code.glob('train-129813*.*'):shutil.copyfile(f,logs/f.name)
    for name in ('review-129810','report-129814','weights-129822','final-review-129824'):
        for ext in ('out','err'):shutil.copyfile(root/f'{name}.{ext}',logs/f'{name}.{ext}')
    for name in ('m5-review-reaudit.json','full-chain-audit.json','full-chain-audit-v2.json',
                 'selected-weight-audit.json','selected-weight-audit-v2.json'):shutil.copyfile(root/name,delivery/name)
    first=json.loads((root/'final-report/aggregate.json').read_text());final=json.loads((root/'final-report-v2/aggregate.json').read_text())
    assert first==final
    assert json.loads((root/'full-chain-audit.json').read_text())==json.loads((root/'full-chain-audit-v2.json').read_text())
    assert json.loads((root/'selected-weight-audit.json').read_text())==json.loads((root/'selected-weight-audit-v2.json').read_text())
    shutil.copytree(root/'final-report-v2',delivery/'report')
    (delivery/'slurm-outcomes.psv').write_text(ledger)
    (delivery/'checkpoint-fingerprints.json').write_text(json.dumps(weights,indent=2))
    (delivery/'evidence-storage-manifest.json').write_text(json.dumps({'summaries':storage,'numeric_data_unchanged':True,
        'original_json_bytes_recover_exactly':True,'original_large_json_retained_in_remote_run_directory':True,
        'reader':'scripts/m6_evidence_io.py; json path falls back to identical json.gz'},indent=2))
    archive=root/'final-delivery-evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        t.add(delivery,arcname='evidence')
        for name in weights:t.add(root/name,arcname='private-checkpoints/'+Path(name).parent.name+'/best-policy.pt')
    print('ARCHIVE_COMPLETE',digest(archive),archive.stat().st_size,'LOSSLESS_SUMMARIES',len(storage),flush=True)


if __name__=='__main__':main()
