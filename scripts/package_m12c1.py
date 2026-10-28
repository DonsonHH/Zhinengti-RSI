"""Preserve opportunity probes, raw one-action branches and original evidence."""
import argparse
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
    run, output = args.run, args.run / 'public-evidence'
    output.mkdir(exist_ok=False)
    def copy(source, rel):
        target = output / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise RuntimeError('Inconsistent duplicate evidence destination')
        shutil.copyfile(source, target)
    jobs = json.loads(args.jobs.read_text())
    jobs['package'] = int(os.environ['SLURM_JOB_ID'])
    formal = run / 'formal-v1'
    for name in ('contract.json', 'summary.json', 'immutable-inputs.json', 'case-index.json', 'process-index.json',
            'branch-index.json', 'main-stage-complete.json'):
        copy(formal / name, name)
    for name in ('summary.json', 'case-index.json', 'branch-index.json', 'contract.json', 'immutable-inputs.json'):
        copy(run / 'smoke-v1' / name, 'smoke/' + name)
    for name in ('development-gate.json', 'smoke-audit.json', 'formal-audit.json'):
        copy(run / name, name)
    for source in (run / 'report').iterdir():
        if source.is_file():
            copy(source, 'report/' + source.name)
    demo = json.loads((run / 'report/demo-selection.json').read_text())
    index = {r['case']: r for r in json.loads((formal / 'case-index.json').read_text())}
    examples = set(demo['control_cases'].values())
    if demo['branch_case'] is not None:
        examples.add(demo['branch_case'])
    for case in examples:
        copy(formal / index[case]['path'], f'demo/{case}.json.gz')
    process = json.loads((formal / 'process-index.json').read_text())
    copy(formal / process[demo['process_key']]['path'], 'demo/process.npz')
    for job in jobs.values():
        for extension in ('out', 'err'):
            for source in run.glob(f'*-{job}.{extension}'):
                copy(source, 'jobs/' + source.name)
    for source in run.glob('*.sbatch'):
        copy(source, 'jobs/' + source.name)
    (output / 'job-identities.json').write_text(json.dumps(jobs, indent=2) + '\n')
    storage = {'root': str(formal),
        'cases': [{'case': row['case'], 'path': str(formal / row['path']), 'sha256': row['sha256'], 'bytes': row['bytes']}
            for row in index.values()],
        'processes': [{'key': key, 'path': str(formal / row['path']), 'sha256': row['sha256'], 'bytes': row['bytes']}
            for key, row in process.items()],
        'scope': 'All compact case/probe/branch records, examples and logs downloaded; complete raw cases/process arrays remain remotely indexed.'}
    (output / 'raw-storage.json').write_text(json.dumps(storage, indent=2) + '\n')
    manifest = {p.relative_to(output).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
        for p in sorted(output.rglob('*')) if p.is_file()}
    (output / 'transfer-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    archive = run / 'delivery.tar.gz'
    with tarfile.open(archive, 'x:gz') as tar:
        for source in sorted(output.rglob('*')):
            if source.is_file():
                tar.add(source, arcname=source.relative_to(output).as_posix())
    receipt = {'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size, 'original_files': len(manifest)}
    (run / 'delivery-storage.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('C1A_PACKAGE_COMPLETE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
