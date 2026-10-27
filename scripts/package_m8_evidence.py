"""Lossless M8 evidence and private selected-weight transfer package."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--required-jobs', required=True)
    args = parser.parse_args()
    ledger = args.ledger.read_text()
    rows = {row['JobID']: row for row in csv.DictReader(io.StringIO(ledger), delimiter='|')}
    for job in args.required_jobs.split(','):
        if rows.get(job, {}).get('State') != 'COMPLETED' or rows[job]['ExitCode'] != '0:0':
            raise RuntimeError(f'Terminal state not successful: {job}')
    evidence = args.root / 'delivery-evidence'
    evidence.mkdir(exist_ok=False)
    weights = {}
    for directory in sorted((args.code / 'results').iterdir()):
        if not directory.is_dir():
            continue
        summary = json.loads(gzip.decompress((directory / 'summary.json.gz').read_bytes()))
        target = evidence / directory.name
        target.mkdir()
        for path in sorted(directory.iterdir()):
            if path.suffix in ('.gz', '.json', '.jsonl'):
                shutil.copyfile(path, target / path.name)
        weight = directory / 'best-policy.pt'
        if weight.exists():
            assert digest(weight) == summary['checkpoint_sha256']
            weights[directory.name] = {'sha256': digest(weight), 'bytes': weight.stat().st_size,
                                       'remote': str(weight), 'private': True}
    logs = evidence / 'jobs'
    logs.mkdir()
    for path in args.root.iterdir():
        if path.is_file() and path.suffix in ('.out', '.err', '.sbatch'):
            shutil.copyfile(path, logs / path.name)
    for name in ('m7-review-reaudit.json', 'development-gate.json', 'full-chain-audit.json', 'actual-weight-audit.json'):
        shutil.copyfile(args.root / name, evidence / name)
    shutil.copytree(args.root / 'final-report', evidence / 'report')
    (evidence / 'slurm-outcomes-before-packaging.psv').write_text(ledger)
    (evidence / 'checkpoint-fingerprints.json').write_text(json.dumps(weights, indent=2))
    files = {path.relative_to(evidence).as_posix(): {'sha256': digest(path), 'bytes': path.stat().st_size}
             for path in sorted(evidence.rglob('*')) if path.is_file()}
    (evidence / 'transfer-files.json').write_text(json.dumps(files, indent=2))
    archive = args.root / 'final-delivery-evidence.tar.gz'
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive, 'w:gz') as output:
        output.add(evidence, arcname='evidence')
        for name in weights:
            output.add(args.code / 'results' / name / 'best-policy.pt', arcname=f'private-checkpoints/{name}/best-policy.pt')
    print('M8_PACKAGE', digest(archive), archive.stat().st_size, 'FILES', len(files), 'PRIVATE_WEIGHTS', len(weights), flush=True)


if __name__ == '__main__':
    main()
