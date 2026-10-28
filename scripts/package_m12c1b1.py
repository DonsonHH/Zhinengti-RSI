"""Original B1 delivery with lossless JSON wrappers and raw remote catalogue."""
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
        dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);raw=source.read_bytes()
        wrapper=source.suffix=='.json' and len(raw)>1000000
        if wrapper:
            dest=dest.with_name(dest.name+'.gz');dest.write_bytes(gzip.compress(raw,mtime=0))
        else:
            shutil.copyfile(source,dest)
        origins[dest.relative_to(out).as_posix()]={'remote_source':str(source),'source_sha256':hashlib.sha256(raw).hexdigest(),
            'source_bytes':len(raw),'lossless_gzip_wrapper_added':wrapper}
    for name in ('smoke','validation','formal'):
        folder=root/name
        for source in sorted(folder.glob('*.json')):
            copy(source,name+'/'+source.name)
        index=json.loads((folder/'case-index.json').read_bytes())
        for row in index:
            storage['cases'].append({'folder':name,'case':row['case'],'path':str(folder/row['path']),'bytes':row['bytes'],'sha256':row['sha256']})
        for key,row in json.loads((folder/'process-index.json').read_bytes()).items():
            storage['processes'].append({'folder':name,'key':key,'path':str(folder/row['path']),'bytes':row['bytes'],'sha256':row['sha256']})
    for source in sorted(root.glob('*-audit.json')):
        copy(source,source.name)
    for source in sorted((root/'jobs').iterdir()):
        if source.is_file():
            copy(source,'jobs/'+source.name)
    for source in sorted((root/'report').iterdir()):
        if source.is_file():
            copy(source,'report/'+source.name)
    index=json.loads((root/'formal/case-index.json').read_bytes());demo=[]
    for tag in ('tour','cue_immediate','cue_gate','evidence_now','lookahead'):
        row=next(r for r in index if r['tag']==tag and r['specification']['scenario']=='coldplate_fouling' and
            r['specification']['duration_s']==180 and r['specification']['seed']==8200 and r['specification']['noise_c']==.2)
        copy(root/'formal'/row['path'],'demo/'+row['case']+'.json.gz');demo.append(row['case'])
        if tag=='tour':
            process=json.loads((root/'formal/process-index.json').read_bytes())[row['process_key']]
            copy(root/'formal'/process['path'],'demo/process.npz')
    (out/'demo-selection.json').write_text(json.dumps({'selection':'Fixed coldplate180 seed8200 noise0.2, not by outcome','cases':demo},indent=2)+'\n')
    (out/'raw-storage.json').write_text(json.dumps(storage,indent=2)+'\n')
    (out/'origin-manifest.json').write_text(json.dumps(origins,indent=2)+'\n')
    manifest={p.relative_to(out).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'transfer-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=root/'delivery.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for source in sorted(out.rglob('*')):
            if source.is_file():
                tar.add(source,arcname=source.relative_to(out).as_posix())
    receipt={'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'original_files':len(manifest),
        'raw_case_files':len(storage['cases']),'raw_process_files':len(storage['processes']),'job_id':os.environ.get('SLURM_JOB_ID')}
    (root/'delivery-storage.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('B1_PACKAGE_COMPLETE',json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
