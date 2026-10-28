"""Preserve original information-study evidence with a transport hash ledger."""
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
            raise RuntimeError('Different originals mapped to the same delivery path')
        shutil.copyfile(source, target)
    jobs = json.loads(args.jobs.read_text())
    jobs['package'] = int(os.environ['SLURM_JOB_ID'])
    storage = {'trials': {}, 'scope': 'Full case and process files remain on the cluster; only predeclared full demos copied here'}
    for t in range(3):
        root = run / f'trial-{t}'
        for name in ('contract.json', 'summary.json', 'immutable-inputs.json', 'case-index.json', 'process-index.json',
                'selections-frozen.json', 'journal.jsonl', 'test-groups.json', 'deployment-interventions.json'):
            copy(root / name, f'trial-{t}/{name}')
        for sub in ('proposals', 'selections'):
            for source in (root / sub).glob('*.json'):
                copy(source, f'trial-{t}/{sub}/{source.name}')
        for name in (f'audit-{t}.json', f'audit-{t}.cases.json', f'metadata-{t}.json'):
            copy(run / name, name)
        index = json.loads((root / 'case-index.json').read_text())
        processes = json.loads((root / 'process-index.json').read_text())
        storage['trials'][str(t)] = {'cases': [{'case': r['case'], 'path': str(root / r['path']),
                'sha256': r['sha256'], 'bytes': r['bytes']} for r in index],
            'processes': [{'key': k, 'path': str(root / v['path']), 'sha256': v['sha256'], 'bytes': v['bytes']}
                for k, v in processes.items()]}
    for source in (run / 'report').iterdir():
        if source.is_file():
            copy(source, 'report/' + source.name)
    demo = json.loads((run / 'report/demo-selection.json').read_text())
    for row in demo:
        copy(run / f"trial-{row['trial']}" / row['path'], f"demo/{row['case']}.json.gz")
    process_index = json.loads((run / 'trial-0/process-index.json').read_text())
    example = next(v for v in process_index.values() if v['process']['seed'] == 7401 and
        v['process']['scenario'] == 'coldplate_fouling' and v['process']['fault_duration_s'] == 180)
    copy(run / 'trial-0' / example['path'], 'demo/process.npz')
    for key, job in jobs.items():
        if not isinstance(job, int):
            continue
        for ext in ('out', 'err'):
            for pattern in (f'*-{job}.{ext}', f'*-{job}_*.{ext}'):
                for source in run.glob(pattern):
                    copy(source, 'jobs/' + source.name)
    for source in run.glob('*.sbatch'):
        copy(source, 'jobs/' + source.name)
    smoke = run / f"smoke-{jobs['smoke']}"
    for name in ('summary.json', 'case-index.json', 'contract.json', 'immutable-inputs.json'):
        copy(smoke / name, 'smoke/' + name)
    for source in (smoke / 'proposals').glob('*.json'):
        copy(source, 'smoke/proposals/' + source.name)
    for name in ('development-gate.json', 'smoke-audit.json', 'metadata-smoke.json', 'm12b-transfer-review.json'):
        copy(run / name, name)
    (output / 'job-identities.json').write_text(json.dumps(jobs, indent=2) + '\n')
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
    print('M12C_PACKAGE_COMPLETE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
