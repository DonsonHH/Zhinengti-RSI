"""Hash-indexed original evidence collection for the paired improver study."""
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
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--jobs', type=Path, required=True)
    args = p.parse_args()
    run = args.run
    output = run / 'public-evidence'
    output.mkdir(exist_ok=False)
    def copy(source, rel):
        dest = output / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    jobs = json.loads(args.jobs.read_text())
    jobs['package'] = int(os.environ['SLURM_JOB_ID'])
    for t in range(3):
        root = run / f'trial-{t}'
        for name in ('contract.json', 'summary.json', 'immutable-inputs.json', 'case-index.json',
                'process-index.json', 'selections-frozen.json', 'journal.jsonl', 'test-groups.json'):
            copy(root / name, f'trial-{t}/{name}')
        for directory in ('meta', 'proposals', 'selections'):
            for source in sorted((root / directory).glob('*.json')):
                copy(source, f'trial-{t}/{directory}/{source.name}')
        for name in (f'audit-{t}.json', f'audit-{t}.cases.json', f'metadata-{t}.json'):
            copy(run / name, name)
    demo = json.loads((run / 'report/demo-selection.json').read_text())
    for row in demo:
        copy(run / f"trial-{row['trial']}" / row['path'], f"demo/{row['case']}.json.gz")
    processes = json.loads((run / 'trial-0/process-index.json').read_text())
    example = next((key, info) for key, info in processes.items() if info['process']['seed'] == 6601
        and info['process']['scenario'] == 'coldplate_fouling' and info['process']['fault_duration_s'] == 180)
    copy(run / 'trial-0' / example[1]['path'], 'demo/process.npz')
    for source in (run / 'report').iterdir():
        if source.is_file():
            copy(source, 'report/' + source.name)
    for key, job in jobs.items():
        if not isinstance(job, int):
            continue
        for ext in ('out', 'err'):
            patterns = (f'*-{job}.{ext}', f'*-{job}_*.{ext}')
            for pattern in patterns:
                for source in run.glob(pattern):
                    copy(source, 'jobs/' + source.name)
    for source in run.glob('*.sbatch'):
        copy(source, 'jobs/' + source.name)
    for key in ('smoke_v1', 'smoke_v2', 'smoke_v3'):
        job = jobs[key]
        folder = run / f'smoke-{job}'
        for name in ('summary.json', 'contract.json', 'case-index.json'):
            copy(folder / name, f'{key}/{name}')
        for directory in ('proposals', 'meta'):
            for source in (folder / directory).glob('*.json'):
                copy(source, f'{key}/{directory}/{source.name}')
    for name in ('development-gate.json', 'smoke-audit.json', 'm12a-objective-review.json'):
        copy(run / name, name)
    (output / 'job-identities.json').write_text(json.dumps(jobs, indent=2) + '\n')
    raw = {'trials': {}, 'scope': 'All full cases/physical arrays retained remotely; public full records are predeclared examples only'}
    for t in range(3):
        root = run / f'trial-{t}'
        index = json.loads((root / 'case-index.json').read_text())
        process = json.loads((root / 'process-index.json').read_text())
        raw['trials'][str(t)] = {'cases': [{k: row[k] for k in ('case', 'sha256', 'bytes')} | {'path': str(root / row['path'])}
                for row in index],
            'processes': [{'key': key, 'path': str(root / row['path']), 'sha256': row['sha256'], 'bytes': row['bytes']}
                for key, row in process.items()]}
    (output / 'raw-storage.json').write_text(json.dumps(raw, indent=2) + '\n')
    manifest = {p.relative_to(output).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
        for p in sorted(output.rglob('*')) if p.is_file()}
    (output / 'transfer-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    archive = run / 'delivery.tar.gz'
    with tarfile.open(archive, 'x:gz') as tar:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                tar.add(p, arcname=p.relative_to(output).as_posix())
    receipt = {'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size, 'original_files': len(manifest)}
    (run / 'delivery-storage.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('M12B_PACKAGE_COMPLETE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
