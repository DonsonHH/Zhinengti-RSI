"""New immutable G source using the exact archived Fv6 backbone."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('version');a=p.parse_args()
    if not a.version.startswith('v') or not a.version[1:].isdigit():raise ValueError('Explicit source snapshot version')
    root=Path(__file__).resolve().parents[1];stage=root/'research/2026-11-01-m12g-local';base=root/'research/2026-10-31-m12f-timing-skill/source/code-v6.tar.gz'
    if sha(base)!='036d9ae1e91c17ef2ce015531da052a7cc2570d6062b7b64a7f415ec7dacf141':raise ValueError('Immutable F backbone archive')
    dest=root/'.work/m12g'/f'code-{a.version}';dest.mkdir(parents=True,exist_ok=False)
    with tarfile.open(base) as t:t.extractall(dest,filter='data')
    inputs=[*list((root/'src/patrol_rsi').glob('g_*.py')),*list((root/'tests').glob('test_g_*.py')),
        root/'scripts/prepare_m12g.py',root/'scripts/package_m12g.py',root/'scripts/g_timing_execution.py',root/'scripts/run_m12g_g0.py',root/'scripts/run_m12g_precheck.py',root/'scripts/finish_m12g_g0.py',
        *list((root/'cluster').glob('m12g_*.sbatch')),*list(stage.glob('*.json'))]
    for file in inputs:
        out=dest/file.relative_to(root);out.parent.mkdir(parents=True,exist_ok=True);data=file.read_bytes()
        if file.suffix=='.sbatch':data=data.replace(b'\r\n',b'\n')
        out.write_bytes(data)
    archive=stage/'source'/f'code-{a.version}.tar.gz';files={}
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as t:
            for file in sorted(dest.rglob('*')):
                if file.is_file():
                    name=file.relative_to(dest).as_posix();files[name]={'sha256':sha(file),'bytes':file.stat().st_size};t.add(file,arcname=name,recursive=False)
    value={'baseline':'a5ab5880e33e3df3cd9aeffc3a90a63c40273c41','sha256':sha(archive),'files':files,'archive_bytes':archive.stat().st_size}
    path=stage/'evidence'/f'source-manifest-{a.version}.json'
    with path.open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2);f.write('\n')
    print('G_SOURCE',a.version,len(files),value['sha256'],str(dest),flush=True)


if __name__=='__main__':main()
