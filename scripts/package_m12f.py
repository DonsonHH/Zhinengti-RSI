"""Create a new immutable F source snapshot; never replace an existing version."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('version');parser.add_argument('--scope',choices=('F0','all'),default='all')
    a=parser.parse_args()
    if not re.fullmatch(r'code-v[1-9][0-9]*',a.version):raise ValueError('Explicit new source version')
    root=Path(__file__).resolve().parents[1];stage=root/'research/2026-10-31-m12f-timing-skill'
    base=root/'research/2026-10-31-m12e-timing-schema/source/code-v2.tar.gz'
    if digest(base)!='9a229784c276a4f1df66c3ac50ac892ec2b3da5c2925d0d3c101783b0a663450':raise ValueError('Immutable base archive')
    target=root/'.work/m12f'/a.version;target.mkdir(parents=True,exist_ok=False)
    with tarfile.open(base) as archive:archive.extractall(target,filter='data')
    for directory,pattern in (('src','*.py'),('scripts','*.py'),('tests','*.py'),('cluster','*.sbatch')):
        for file in (root/directory).rglob(pattern):
            if a.scope=='F0' and file.name in ('timing_learning.py','test_timing_learning.py'):continue
            output=target/file.relative_to(root);output.parent.mkdir(parents=True,exist_ok=True)
            data=file.read_bytes()
            if file.suffix=='.sbatch':data=data.replace(b'\r\n',b'\n')
            output.write_bytes(data)
    inputs=[stage/name for name in ('M12F-Study-v1.json','cases-v1.json')]
    if a.scope=='all':inputs.append(stage/'prompt-templates-v1.json')
    inputs+=list((root/'research/2026-10-31-m12e-timing-schema/evidence/run/E1').glob('seed-*/public-release/*--public-selections-release.json'))
    if len(inputs)!=(38 if a.scope=='F0' else 39):raise ValueError('All36 archived E1 public-input authorities and current prompt templates required')
    for file in inputs:
        output=target/file.relative_to(root);output.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,output)
    path=stage/'source'/f'{a.version}.tar.gz';files={}
    with path.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as archive:
            for file in sorted(target.rglob('*')):
                if not file.is_file():continue
                relative=file.relative_to(target).as_posix();files[relative]={'bytes':file.stat().st_size,'sha256':digest(file)}
                archive.add(file,arcname=relative,recursive=False)
    manifest={'baseline':'39d06ce5edb4dc7f7fff18bedc76909350a946d4','scope':a.scope,'archive':str(path.relative_to(root)),
        'sha256':digest(path),'bytes':path.stat().st_size,'files':files}
    dest=stage/'evidence'/f'source-manifest-{a.version.removeprefix("code-")}.json'
    with dest.open('x',encoding='utf-8') as stream:json.dump(manifest,stream,indent=2);stream.write('\n')
    print(json.dumps({'root':str(target),'archive':str(path),'manifest':str(dest),'sha256':manifest['sha256'],'files':len(files)}),flush=True)


if __name__=='__main__':main()
