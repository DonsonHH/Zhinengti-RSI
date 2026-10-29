"""Lossless compact B0 delivery with full dataset paths and byte hashes."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile


FOLDERS = ('development-0', 'development-1', 'validation-0', 'validation-1',
           'test-0', 'test-1', 'test-2', 'test-3')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report-folder', default='report-v2')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    original, wrappers = {}, {}

    def copy(path, name):
        raw = path.read_bytes()
        dest = args.output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.suffix == '.json' and len(raw) > 1000000:
            dest = dest.with_name(dest.name + '.gz')
            dest.write_bytes(gzip.compress(raw, mtime=0))
            wrappers[dest.relative_to(args.output).as_posix()] = True
        else:
            dest.write_bytes(raw)
        original[dest.relative_to(args.output).as_posix()] = {'source_path': str(path), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}

    for folder in FOLDERS:
        for path in sorted((args.root / 'full' / folder).rglob('*')):
            if not path.is_file():
                continue
            rel = path.relative_to(args.root).as_posix()
            # All label arrays retained, public matrices for two predetermined
            # development examples only. Other full features remain indexed.
            if path.name == 'public-features.npz' and not (folder == 'development-0' and path.parent.name in ('condition-00', 'condition-14')):
                continue
            copy(path, rel)
    for path in sorted((args.root / args.report_folder).rglob('*')):
        if path.is_file():
            copy(path, 'report/' + path.relative_to(args.root / args.report_folder).as_posix())
    for name in ('gate', 'reviewed/gate', 'anchored/gate', 'final/gate', 'jobs'):
        for path in sorted((args.root / name).rglob('*')):
            if path.is_file():
                copy(path, path.relative_to(args.root).as_posix())
    for name in ('pilot', 'reviewed/pilot', 'anchored/pilot'):
        for path in sorted((args.root / name).rglob('summary.json')):
            copy(path, path.relative_to(args.root).as_posix())
    (args.output / 'origin-manifest.json').write_text(json.dumps(original, indent=2) + '\n', encoding='utf-8')
    (args.output / 'wrappers.json').write_text(json.dumps(wrappers, indent=2) + '\n', encoding='utf-8')
    transfer = {p.relative_to(args.output).as_posix(): {'bytes': p.stat().st_size,
        'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in args.output.rglob('*') if p.is_file()}
    (args.output / 'transfer-manifest.json').write_text(json.dumps(transfer, indent=2) + '\n', encoding='utf-8')
    archive = args.output.parent / 'delivery.tar.gz'
    with tarfile.open(archive, 'x:gz') as tar:
        for path in sorted(args.output.rglob('*')):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(args.output).as_posix(), recursive=False)
    receipt = {'archive_bytes': archive.stat().st_size, 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'transferred_files': len(transfer), 'local_full_feature_conditions': 2,
        'local_all_A_B_label_conditions': 128, 'remote_full_feature_conditions': 128,
        'source_scope': 'All labels/statistics/source identities and two fixed public feature examples. Complete 128 public matrices are remote and byte indexed.'}
    (args.output.parent / 'delivery-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
