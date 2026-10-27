"""Package M9 public traces/plots; retain full numerical arrays in run storage."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--smoke', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--required-jobs', required=True)
    args = parser.parse_args()
    ledger = args.ledger.read_text()
    jobs = {row['JobID']: row for row in csv.DictReader(io.StringIO(ledger), delimiter='|')}
    for job in args.required_jobs.split(','):
        if jobs.get(job, {}).get('State') != 'COMPLETED' or jobs[job]['ExitCode'] != '0:0':
            raise RuntimeError(f'Job not terminal-successful: {job}')
    destination = args.root / 'delivery-evidence'
    destination.mkdir(exist_ok=False)
    for name in ('contract.json', 'topology.json', 'case-index.json', 'summary.json', 'demo.json.gz'):
        shutil.copyfile(args.results / name, destination / name)
    shutil.copytree(args.results / 'cases', destination / 'cases')
    smoke = destination / args.smoke.name; smoke.mkdir()
    for name in ('contract.json', 'summary.json', 'case-index.json'):
        shutil.copyfile(args.smoke / name, smoke / name)
    logs = destination / 'jobs'; logs.mkdir()
    for path in args.root.iterdir():
        if path.is_file() and path.suffix in ('.out', '.err', '.sbatch'):
            shutil.copyfile(path, logs / path.name)
    for name in ('development-gate.json', 'm8-review-reaudit.json', 'm8-review-actual-weights.json', 'full-chain-audit.json'):
        shutil.copyfile(args.root / name, destination / name)
    shutil.copytree(args.root / 'final-report', destination / 'report')
    (destination / 'slurm-outcomes-before-packaging.psv').write_text(ledger)
    raw = []
    for entry in json.loads((args.results / 'case-index.json').read_text()):
        info = entry['private_numeric_file']; path = Path(info['path'])
        if sha(path) != info['sha256'] or path.stat().st_size != info['bytes']:
            raise RuntimeError('Full numeric array fingerprint mismatch')
        raw.append({'case': entry['case'], 'path': str(path), 'sha256': info['sha256'], 'bytes': info['bytes'],
                    'numeric_arrays_sha256': entry['numeric_arrays_sha256']})
    (destination / 'raw-numeric-storage.json').write_text(json.dumps({'files': raw,
        'total_bytes': sum(row['bytes'] for row in raw), 'all_numeric_arrays_replayed_by_independent_job': True,
        'retention': 'Full NPZ arrays retained under run results/private-traces; not duplicated in Git or this transfer; exact demo arrays and all scalar traces included'}, indent=2))
    files = {path.relative_to(destination).as_posix(): {'bytes': path.stat().st_size, 'sha256': sha(path)}
             for path in sorted(destination.rglob('*')) if path.is_file()}
    (destination / 'transfer-files.json').write_text(json.dumps(files, indent=2))
    archive = args.root / 'final-delivery-evidence.tar.gz'
    if archive.exists(): raise FileExistsError(archive)
    with tarfile.open(archive, 'w:gz') as output: output.add(destination, arcname='evidence')
    print('M9_PACKAGE', sha(archive), archive.stat().st_size, 'FILES', len(files), 'RETAINED_FULL_TRACES', len(raw), flush=True)


if __name__ == '__main__': main()
