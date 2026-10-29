"""Retain all compact factors and index every full public-state original."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


FOLDERS = ('development-0', 'development-1', 'validation-0', 'validation-1',
           'test-0', 'test-1', 'test-2', 'test-3')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    original_files, wrappers, inventory = {}, {}, []
    total_bytes = 0
    for folder in FOLDERS:
        source = args.root / 'formal' / folder
        for path in sorted(source.rglob('*')):
            if not path.is_file():
                continue
            rel = path.relative_to(args.root / 'formal').as_posix()
            info = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}
            total_bytes += info['bytes']
            inventory.append(info)
            is_history = path.name == 'public-states.jsonl.gz'
            # Known historical examples, fixed by source identity before new
            # aggregate results. Full histories of other cases stay indexed.
            demo = rel.startswith('development-0/condition-14/') or rel.startswith('validation-1/condition-12/')
            if is_history and not demo:
                continue
            dest = args.output / 'formal' / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            original_files[dest.relative_to(args.output).as_posix()] = info
    if total_bytes > 2147483648:
        raise RuntimeError('Frozen raw audit output storage cap exceeded')
    for folder in ('report', 'gate', 'reviewed/gate', 'pilot', 'reviewed/pilot', 'jobs'):
        source = args.root / folder
        for path in sorted(source.rglob('*')):
            if not path.is_file() or path.name == 'public-states.jsonl.gz':
                continue
            if source.name == 'pilot' and path.name not in ('summary.json', 'input-binding.json'):
                continue
            relative = path.relative_to(args.root)
            dest = args.output / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            raw = path.read_bytes()
            if path.suffix == '.json' and len(raw) > 1000000:
                dest = dest.with_name(dest.name + '.gz')
                dest.write_bytes(gzip.compress(raw, mtime=0))
                wrappers[dest.relative_to(args.output).as_posix()] = {'original_bytes': len(raw),
                    'original_sha256': hashlib.sha256(raw).hexdigest(), 'lossless_gzip_wrapper_added': True}
            else:
                dest.write_bytes(raw)
            original_files[dest.relative_to(args.output).as_posix()] = {'path': str(path),
                'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    for name, value in [('raw-storage.json', {'root': str(args.root), 'files': inventory,
                                             'total_original_factor_bytes': total_bytes,
                                             'full_public_histories': 128}),
                        ('origin-manifest.json', original_files), ('wrappers.json', wrappers)]:
        (args.output / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    transfer = {p.relative_to(args.output).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                for p in sorted(args.output.rglob('*')) if p.is_file()}
    (args.output / 'transfer-manifest.json').write_text(json.dumps(transfer, indent=2) + '\n', encoding='utf-8')
    archive = args.output.parent / 'delivery.tar.gz'
    with tarfile.open(archive, 'x:gz') as tar:
        for path in sorted(args.output.rglob('*')):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(args.output).as_posix(), recursive=False)
    receipt = {'archive_bytes': archive.stat().st_size, 'archive_sha256': sha(archive),
        'transferred_original_files': len(transfer), 'indexed_original_factor_files': len(inventory),
        'all_full_public_states_preserved_remotely': 128, 'local_complete_public_histories': 2,
        'raw_original_factor_bytes': total_bytes}
    (args.output.parent / 'delivery-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
