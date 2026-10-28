"""Collect original M12 evidence with a transfer manifest; preserve full raw data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--jobs', type=Path, required=True)
    args = p.parse_args()
    run, pilot = args.run, args.run / 'pilot-v3'
    output = run / 'public-evidence'
    output.mkdir(exist_ok=False)
    def copy(source, relative):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in ('contract.json', 'summary.json', 'immutable-inputs.json', 'case-index.json', 'process-index.json',
            'selections-frozen.json', 'test-groups.json', 'journal.jsonl'):
        copy(pilot / name, name)
    for directory in ('proposals', 'selections'):
        for source in sorted((pilot / directory).glob('*.json')):
            copy(source, f'{directory}/{source.name}')
    for source in sorted((run / 'report').iterdir()):
        if source.is_file():
            copy(source, 'report/' + source.name)
    for name in ('full-audit.json', 'smoke-independent-audit.json', 'development-gate-v3.json'):
        copy(run / name, name)
    for attempt in ('smoke-v2', 'smoke-v3'):
        for name in ('summary.json', 'case-index.json', 'immutable-inputs.json'):
            copy(run / attempt / name, f'{attempt}/{name}')
        for source in (run / attempt / 'proposals').glob('*.json'):
            copy(source, f'{attempt}/proposals/{source.name}')
    jobs = json.loads(args.jobs.read_text())
    jobs['package'] = int(os.environ['SLURM_JOB_ID'])
    for job in jobs.values():
        if not isinstance(job, int):
            continue
        for ext in ('out', 'err'):
            for source in run.glob(f'*-{job}.{ext}'):
                copy(source, 'jobs/' + source.name)
    for name in ('audit-smoke.sbatch', 'audit-report.sbatch', 'package.sbatch'):
        copy(run / name, 'jobs/' + name)
    (output / 'job-identities.json').write_text(json.dumps(jobs, indent=2) + '\n')
    demo_selection = json.loads((run / 'report/demo-selection.json').read_text())
    index = {r['case']: r for r in json.loads((pilot / 'case-index.json').read_text())}
    for key in {r['case'] for r in demo_selection.values()}:
        copy(pilot / index[key]['path'], f'demo/{key}.json.gz')
    process_index = json.loads((pilot / 'process-index.json').read_text())
    process_key = 'coldplate_fouling--180--6301'
    copy(pilot / process_index[process_key]['path'], 'demo/process.npz')
    storage = {'remote_root': str(pilot),
        'cases': [{'case': r['case'], 'path': str(pilot / r['path']), 'sha256': r['sha256'], 'bytes': r['bytes']}
            for r in index.values()],
        'processes': [{'key': key, 'path': str(pilot / row['path']), 'sha256': row['sha256'], 'bytes': row['bytes']}
            for key, row in process_index.items()],
        'downloaded_scope': 'All compact case metrics, proposals, selections and logs; predeclared full demos and one process NPZ. Full cases/process arrays remain remotely indexed.'}
    (output / 'raw-storage.json').write_text(json.dumps(storage, indent=2) + '\n')
    manifest = {p.relative_to(output).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
        for p in sorted(output.rglob('*')) if p.is_file()}
    (output / 'transfer-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    archive = run / 'delivery.tar.gz'
    with tarfile.open(archive, 'x:gz') as tar:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(output).as_posix())
    storage = {'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
        'original_files': len(manifest), 'package_job': jobs['package']}
    (run / 'delivery-storage.json').write_text(json.dumps(storage, indent=2) + '\n')
    print('M12_PACKAGE_COMPLETE', json.dumps(storage), flush=True)


if __name__ == '__main__':
    main()
