"""Lossless B2 compact delivery, frozen examples and remote raw catalogue."""
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
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    root=args.run;out=root/'public-evidence';out.mkdir(exist_ok=False);origins={};storage={'root':str(root),'cases':[],'processes':[]}
    def copy(source,rel):
        target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);raw=source.read_bytes()
        wrapper=source.suffix=='.json' and len(raw)>1000000
        if wrapper:
            target=target.with_name(target.name+'.gz');target.write_bytes(gzip.compress(raw,mtime=0))
        else:
            shutil.copyfile(source,target)
        origins[target.relative_to(out).as_posix()]={'remote_source':str(source),'source_bytes':len(raw),
            'source_sha256':hashlib.sha256(raw).hexdigest(),'lossless_gzip_wrapper_added':wrapper}
    for name in ['smoke']+[f'chunk-{i}' for i in range(8)]:
        folder=root/name
        for source in sorted(folder.glob('*.json')):
            copy(source,name+'/'+source.name)
        for r in json.loads((folder/'case-index.json').read_bytes()):
            storage['cases'].append({'folder':name,'case':r['case'],'path':str(folder/r['path']),'bytes':r['bytes'],'sha256':r['sha256']})
        for key,r in json.loads((folder/'process-index.json').read_bytes()).items():
            storage['processes'].append({'folder':name,'process_key':key,'path':str(folder/r['path']),'bytes':r['bytes'],'sha256':r['sha256']})
    for source in sorted(root.glob('*-audit.json')):
        copy(source,source.name)
    for source in sorted((root/'report').iterdir()):
        if source.is_file():
            copy(source,'report/'+source.name)
    # The package's own live stdout is collected after it finishes, separately.
    for source in sorted((root/'jobs').iterdir()):
        if source.is_file() and not source.name.startswith('package-'):
            copy(source,'jobs/'+source.name)
    index=json.loads((root/'chunk-2/case-index.json').read_bytes());examples=[]
    for tag in ('tour','evidence_now','lookahead','lookahead_permuted','lookahead_conservative','blind_0'):
        row=next(r for r in index if r['tag']==tag and r['specification']['scenario']=='aux_feeder_loss+cdu_fouling' and
            r['specification']['duration_s']==180 and r['specification']['seed']==8300 and
            r['specification']['fault_start_s']==335 and r['specification']['noise_c']==.2)
        copy(root/'chunk-2'/row['path'],'demo/'+row['case']+'.json.gz');examples.append(row['case'])
        if tag=='tour':
            process=json.loads((root/'chunk-2/process-index.json').read_bytes())[row['process_key']]
            copy(root/'chunk-2'/process['path'],'demo/process.npz')
    (out/'demo-selection.json').write_text(json.dumps({'selection':'Fixed coupled180 seed8300 phase335 noise0.2, not selected by outcome','cases':examples},indent=2)+'\n')
    (out/'raw-storage.json').write_text(json.dumps(storage,indent=2)+'\n')
    (out/'origin-manifest.json').write_text(json.dumps(origins,indent=2)+'\n')
    manifest={p.relative_to(out).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'transfer-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=root/'delivery.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for source in sorted(out.rglob('*')):
            if source.is_file():
                tar.add(source,arcname=source.relative_to(out).as_posix())
    receipt={'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'original_files':len(manifest),
        'raw_cases':len(storage['cases']),'raw_process_files':len(storage['processes']),'job_id':os.environ.get('SLURM_JOB_ID')}
    (root/'delivery-storage.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('B2_PACKAGE_COMPLETE',json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
