"""Lossless public delivery, raw remote index and fixed-condition examples."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True)
    args=p.parse_args();run=args.run_root;out=run/'public-evidence';out.mkdir(exist_ok=False)
    origins={}
    def copy(source,relative):
        dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        raw=source.read_bytes();compressed=source.suffix=='.json' and len(raw)>1000000
        if compressed:
            dest=dest.with_name(dest.name+'.gz');dest.write_bytes(gzip.compress(raw,mtime=0))
        else:
            shutil.copyfile(source,dest)
        rel=dest.relative_to(out).as_posix()
        origins[rel]={'remote_source':str(source),'source_bytes':len(raw),'source_sha256':hashlib.sha256(raw).hexdigest(),
            'lossless_gzip_wrapper_added':compressed,'bytes':dest.stat().st_size,'sha256':sha(dest)}
    folders=['smoke-data','smoke-fit','data']+[f'fit-{i}' for i in range(3)]+['test-common']+[f'test-{i}' for i in range(3)]+['heldout']
    storage={'root':str(run),'cases':[],'processes':[]}
    for folder in folders:
        root=run/folder
        for source in sorted(root.rglob('*')):
            if source.is_file() and not any(part in ('cases','process-traces') for part in source.relative_to(root).parts):
                copy(source,folder+'/'+source.relative_to(root).as_posix())
        for row in json.loads((root/'case-index.json').read_text()):
            storage['cases'].append({'folder':folder,'case':row['case'],'path':str(root/row['path']),
                'bytes':row['bytes'],'sha256':row['sha256']})
        for key,row in json.loads((root/'process-index.json').read_text()).items():
            storage['processes'].append({'folder':folder,'key':key,'path':str(root/row['path']),
                'bytes':row['bytes'],'sha256':row['sha256']})
    for source in sorted(run.glob('*-audit.json')):
        copy(source,source.name)
    for source in sorted((run/'jobs').iterdir()):
        if source.is_file():
            copy(source,'jobs/'+source.name)
    for source in sorted((run/'report').iterdir()):
        if source.is_file():
            copy(source,'report/'+source.name)
    example=[]
    for folder,tag in (('test-common','tour'),('test-common','cue_rule'),('test-common','actor'),('test-0','net'),('test-0','target')):
        index=json.loads((run/folder/'case-index.json').read_text())
        row=next(r for r in index if r['tag']==tag and r['specification']['scenario']=='coldplate_fouling' and
            r['specification']['duration_s']==180 and r['specification']['seed']==7900 and r['specification']['noise_c']==.2)
        copy(run/folder/row['path'],'demo/'+folder+'--'+row['case']+'.json.gz');example.append({'folder':folder,'case':row['case']})
        if tag=='tour':
            process=json.loads((run/folder/'process-index.json').read_text())[row['process_key']]
            copy(run/folder/process['path'],'demo/process.npz')
    (out/'demo-selection.json').write_text(json.dumps({'selection':'Fixed coldplate180 / seed7900 / noise0.2, not by score','examples':example},indent=2)+'\n')
    (out/'raw-storage.json').write_text(json.dumps(storage,indent=2)+'\n')
    (out/'origin-manifest.json').write_text(json.dumps(origins,indent=2)+'\n')
    manifest={p.relative_to(out).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'transfer-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=run/'delivery.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for source in sorted(out.rglob('*')):
            if source.is_file():
                tar.add(source,arcname=source.relative_to(out).as_posix())
    receipt={'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'original_files':len(manifest),
        'raw_case_files_retained_remote':len(storage['cases']),'raw_process_files_retained_remote':len(storage['processes']),
        'job_id':os.environ.get('SLURM_JOB_ID')}
    (run/'delivery-storage.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('C1B_PACKAGE_COMPLETE',json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
