"""Download pinned open weights on a login node, verifying official LFS hashes."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('--metadata',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--transport',choices=['hf-mirror','modelscope'],default='hf-mirror')
parser.add_argument('--workers',type=int,default=3)
args=parser.parse_args()
metadata=json.loads(args.metadata.read_text())
args.output.mkdir(parents=True,exist_ok=True)
revision=metadata['sha']
files=[x for x in metadata['siblings'] if x['rfilename'].endswith(('.safetensors','.json','.txt','.jinja')) or x['rfilename']=='LICENSE']

def checksum(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        while chunk:=handle.read(8*1024*1024):digest.update(chunk)
    return digest.hexdigest()

def fetch(item):
    name=item['rfilename'];destination=args.output/name
    if Path(name).name!=name:raise ValueError('Nested model file not allowed')
    expected=item.get('lfs',{}).get('sha256')
    if destination.exists():
        sha=checksum(destination)
        if expected and sha!=expected:raise ValueError('Existing model hash mismatch: '+name)
        if item.get('size') is not None and destination.stat().st_size!=item['size']:raise ValueError('Existing model size mismatch: '+name)
        if not expected and item.get('blobId'):
            content=destination.read_bytes()
            blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
            if blob!=item['blobId']:raise ValueError('Existing official Git blob mismatch: '+name)
        return {'file':name,'bytes':destination.stat().st_size,'sha256':sha}
    partial=Path(str(destination)+'.partial')
    url=(f'https://hf-mirror.com/{metadata["id"]}/resolve/{revision}/{name}' if args.transport=='hf-mirror'
         else f'https://modelscope.cn/models/{metadata["id"]}/resolve/master/{name}')
    if not partial.exists() or partial.stat().st_size!=item.get('size'):
        subprocess.run(['curl','-sS','-L','--fail','--retry','2','--connect-timeout','20','--max-time','1800','-C','-',url,'-o',str(partial)],check=True)
    sha=checksum(partial)
    if expected and sha!=expected:raise ValueError('Downloaded model hash mismatch: '+name)
    if item.get('size') is not None and partial.stat().st_size!=item['size']:raise ValueError('Model size mismatch')
    if not expected and item.get('blobId'):
        content=partial.read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if blob!=item['blobId']:raise ValueError('Downloaded official Git blob mismatch: '+name)
    partial.rename(destination)
    result={'file':name,'bytes':destination.stat().st_size,'sha256':sha}
    print(json.dumps(result),flush=True)
    return result

with ThreadPoolExecutor(max_workers=args.workers) as pool:records=list(pool.map(fetch,files))
(args.output/'verified_manifest.json').write_text(json.dumps({'id':metadata['id'],'revision':revision,'official_metadata_source':'huggingface.co/api/models','download_transport':args.transport,'files':records},indent=2))
print('MODEL_STAGE_COMPLETE',flush=True)
