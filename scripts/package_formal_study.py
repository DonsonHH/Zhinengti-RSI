"""Original compact data, fixed complete examples, all-case remote inventory."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);args=p.parse_args();root=args.run_root
out=root/'public-evidence';out.mkdir(exist_ok=False);origins={};storage={'root':str(root),'cases':[],'processes':[]}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def copy(source,rel):
    target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);data=source.read_bytes()
    wrapper=source.suffix=='.json' and len(data)>1000000
    if wrapper:target=target.with_name(target.name+'.gz');target.write_bytes(gzip.compress(data,mtime=0))
    else:shutil.copyfile(source,target)
    origins[target.relative_to(out).as_posix()]={'remote_source':str(source),'source_bytes':len(data),
        'source_sha256':hashlib.sha256(data).hexdigest(),'lossless_gzip_wrapper_added':wrapper}
for phase,size in (('development',2),('validation',2),('test',4)):
    for i in range(size):
        name=f'{phase}-{i}';folder=root/name
        for path in sorted(folder.glob('*.json')):copy(path,name+'/'+path.name)
        copy(root/(name+'-audit.json'),name+'-audit.json')
        checkpoint=root/(name+'-audit.progress.jsonl')
        if checkpoint.exists():copy(checkpoint,name+'-audit.progress.jsonl')
        history=root/(name+'-historical-binding.json')
        if history.exists():copy(history,name+'-historical-binding.json')
        for r in json.loads((folder/'case-index.json').read_bytes()):
            storage['cases'].append({'folder':name,'case':r['case'],'path':str(folder/r['path']),'bytes':r['bytes'],'sha256':r['sha256']})
        for key,r in json.loads((folder/'process-index.json').read_bytes()).items():
            storage['processes'].append({'folder':name,'key':key,'path':str(folder/r['path']),'bytes':r['bytes'],'sha256':r['sha256'],'numeric_digest':r['numeric_digest']})
for phase in ('development','validation'):copy(root/(phase+'-selection.json'),phase+'-selection.json')
for name in ('gate','code-reviewed-gate/gate','audit-repair-gate/gate'):
    for path in sorted((root/name).glob('*')):
        if path.is_file():copy(path,name+'/'+path.name)
for path in sorted((root/'report').glob('*')):
    if path.is_file():copy(path,'report/'+path.name)
for path in sorted((root/'jobs').iterdir()):
    if path.is_file() and not path.name.startswith('package-'):copy(path,'jobs/'+path.name)
# Six full runs fixed before final data: no outcome-based example selection.
examples=[]
for chunk,scenario,start,noise in ((0,'coldplate_fouling',335,.2),(1,'aux_feeder_loss+cdu_fouling',705,1.)):
    folder=root/f'test-{chunk}';index=json.loads((folder/'case-index.json').read_bytes())
    for alias in ('tour','lookahead','revised'):
        row=next(r for r in index if r['alias']==alias and r['specification']['scenario']==scenario
            and r['specification']['fault_start_s']==start and r['specification']['noise_c']==noise)
        copy(folder/row['path'],'demo/'+row['case']+'.json.gz');examples.append(row['case'])
        if chunk==0 and alias=='tour':
            process=json.loads((folder/'process-index.json').read_bytes())[row['process_key']]
            copy(folder/process['path'],'demo/process.npz')
(out/'demo-selection.json').write_text(json.dumps({'scope':'Fixed script before final-test access; not selected by outcome','cases':examples},indent=2)+'\n')
(out/'raw-storage.json').write_text(json.dumps(storage,indent=2)+'\n')
(out/'origin-manifest.json').write_text(json.dumps(origins,indent=2)+'\n')
manifest={path.relative_to(out).as_posix():{'bytes':path.stat().st_size,'sha256':sha(path)} for path in sorted(out.rglob('*')) if path.is_file()}
(out/'transfer-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=root/'delivery.tar.gz'
with tarfile.open(archive,'x:gz') as tar:
    for path in sorted(out.rglob('*')):
        if path.is_file():tar.add(path,arcname=path.relative_to(out).as_posix())
receipt={'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'original_files':len(manifest),
    'raw_cases':len(storage['cases']),'raw_process_files':len(storage['processes'])}
(root/'delivery-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('FORMAL_PACKAGE_DONE',json.dumps(receipt),flush=True)
