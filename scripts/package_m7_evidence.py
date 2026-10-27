"""Package terminal M7 evidence using a login-generated Slurm ledger."""
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
    parser.add_argument('--array-id', required=True)
    parser.add_argument('--review-id', required=True)
    parser.add_argument('--smoke-id', required=True)
    parser.add_argument('--report-id', required=True)
    args = parser.parse_args()
    ledger = args.ledger.read_text()
    states = {row['JobID']: row for row in csv.DictReader(io.StringIO(ledger), delimiter='|')}
    required = [args.review_id, args.smoke_id, args.report_id] + [f'{args.array_id}_{index}' for index in range(27)]
    for job in required:
        if states.get(job, {}).get('State') != 'COMPLETED' or states[job]['ExitCode'] != '0:0':
            raise RuntimeError(f'Incomplete required job {job}')
    target = args.root / 'delivery-evidence'
    target.mkdir(exist_ok=False)
    for directory in sorted((args.code / 'results').iterdir()):
        if not directory.is_dir():
            continue
        summary = json.loads(gzip.decompress((directory / 'summary.json.gz').read_bytes()))
        for group in summary['groups'].values():
            if not summary['smoke'] and len(group['cases']) != 96:
                raise RuntimeError('Incomplete final conditions')
        destination = target / directory.name
        destination.mkdir()
        for path in sorted(directory.iterdir()):
            if path.suffix in ('.json', '.gz'):
                shutil.copyfile(path, destination / path.name)
    logs = target / 'jobs'
    logs.mkdir()
    for path in args.root.iterdir():
        if path.is_file() and path.suffix in ('.out', '.err', '.sbatch'):
            shutil.copyfile(path, logs / path.name)
    for name in ('m6-review-reaudit.json', 'development-gate.json', 'full-chain-audit.json'):
        shutil.copyfile(args.root / name, target / name)
    shutil.copytree(args.root / 'final-report', target / 'report')
    (target / 'slurm-outcomes-before-packaging.psv').write_text(ledger)
    files = {str(path.relative_to(target)): {'bytes': path.stat().st_size, 'sha256': digest(path)}
             for path in sorted(target.rglob('*')) if path.is_file()}
    (target / 'transfer-files.json').write_text(json.dumps(files, indent=2))
    archive = args.root / 'final-delivery-evidence.tar.gz'
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive, 'w:gz') as output:
        output.add(target, arcname='evidence')
    print('M7_PACKAGE', digest(archive), archive.stat().st_size, 'FILES', len(files), flush=True)


if __name__ == '__main__':
    main()
