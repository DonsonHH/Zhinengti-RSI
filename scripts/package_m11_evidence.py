"""Package completed M11 evidence; full cases/process arrays remain indexed."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def xml_tokens(node):
    return (node.tag, sorted((k, ' '.join(v.split())) for k, v in node.attrib.items()),
            ' '.join((node.text or '').split()), tuple(xml_tokens(child) for child in node))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    ids = json.loads((root / 'job-identities.json').read_text())
    audit = json.loads((root / f"full-audit-{ids['audit_report']}.json").read_text())
    assert audit['all_test_episodes_replayed'] == 1136 and audit['actual_selected_policy_weights_read'] == 6
    results = root / f"evaluation-{ids['evaluate']}"
    report = root / f"report-{ids['audit_report']}"
    out = root / 'public-evidence'
    out.mkdir(exist_ok=False)
    for name in ('jobs', 'report', 'demo', 'training-v1', 'training-v2', 'capacity-v1', 'capacity-v2'):
        (out / name).mkdir()
    copies = {
        root / 'job-identities.json': 'job-identities.json',
        root / 'development-gate.json': 'development-gate-v1.json',
        root / 'development-gate-v2.json': 'development-gate-v2.json',
        root / 'training-gate-v2.json': 'training-gate-v2.json',
        root / f"m10-full-review-{ids['review']}.json": 'm10-full-review.json',
        root / f"full-audit-{ids['audit_report']}.json": 'full-audit.json',
        root / 'resource-profile.json': 'resource-profile.json',
        results / 'contract.json': 'contract.json',
        results / 'summary.json': 'evaluation-summary.json',
        results / 'process-index.json': 'process-index.json',
        root / 'review-m10.sbatch': 'jobs/review-m10.sbatch',
        root / 'profile-resources.py': 'jobs/profile-resources.py',
        root / 'aisles-profile.py': 'jobs/aisles-profile.py',
        root / 'profile-resources.sbatch': 'jobs/profile-resources.sbatch',
        root / 'audit-report.sbatch': 'jobs/audit-report.sbatch',
        root / 'package.sbatch': 'jobs/package.sbatch',
        root / 'ledger-before-package.psv': 'slurm-before-package.psv',
    }
    for version, job, folder in ((1, ids['smoke'], 'capacity-v1'), (2, ids['smoke_v2'], 'capacity-v2')):
        for file in ('feasibility.json', 'aisles-open.json', 'aisles-restricted.json'):
            copies[root / f'capacity-{job}' / file] = folder + '/' + file
        copies[root / f'smoke-model-{job}' / 'training.json'] = f'development-training-v{version}.json'
    for name, prefix in (('review', 'review'), ('smoke', 'smoke'), ('resource_profile', 'profile'),
                         ('smoke_v2', 'smoke-v2'), ('evaluate', 'evaluate'), ('audit_report', 'audit-report')):
        for extension in ('out', 'err'):
            filename = f"{prefix}-{ids[name]}.{extension}"
            copies[root / filename] = 'jobs/' + filename
    weights = []
    for version, folder, key, prefix in ((1, 'policies', 'train_array', 'train'), (2, 'policies-v2', 'train_array_v2', 'train-v2')):
        for mode in ('nominal', 'history'):
            for seed in (0, 1, 2):
                training = root / folder / f'{mode}-seed-{seed}'
                copies[training / 'training.json'] = f'training-v{version}/{mode}-seed-{seed}.json'
                record = json.loads((training / 'training.json').read_text())
                path = training / 'selected.pt'
                weights.append({'version': version, 'estimator': mode, 'seed': seed,
                                'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size,
                                'training_record_sha256': sha(training / 'training.json'),
                                'used_for_formal_tests': version == 2})
                task = seed + (3 if mode == 'history' else 0)
                for extension in ('out', 'err'):
                    filename = f'{prefix}-{ids[key]}_{task}.{extension}'
                    copies[root / filename] = 'jobs/' + filename
    for source, relative in copies.items():
        shutil.copyfile(source, out / relative)
    original = (results / 'case-index.json').read_bytes()
    (out / 'case-index.json.gz').write_bytes(gzip.compress(original, mtime=0))
    index = json.loads(original)
    processes = json.loads((results / 'process-index.json').read_text())
    raw_storage = {'case_files': [{'case': row['case'], 'path': str(results / row['record']),
                                  'bytes': row['bytes'], 'sha256': row['sha256']} for row in index],
                   'process_arrays': [dict(row, path=str(results / row['path'])) for row in processes.values()],
                   'policy_checkpoints': weights,
                   'temperature_estimator': {'path': str(root / 'frozen-estimator.pt'),
                                             'sha256': sha(root / 'frozen-estimator.pt'),
                                             'bytes': (root / 'frozen-estimator.pt').stat().st_size},
                   'case_index_uncompressed_sha256': hashlib.sha256(original).hexdigest(),
                   'all_formal_case_files_and_process_arrays_read_by_independent_audit': True,
                   'retrieved_scope': 'All compact per-case metrics and hashes; four full demo cases and one full process NPZ only'}
    (out / 'raw-storage.json').write_text(json.dumps(raw_storage, indent=2))
    normalizations = []
    for source in report.iterdir():
        destination = out / 'report' / source.name
        raw = source.read_bytes()
        if source.suffix == '.svg':
            clean = b'\n'.join(line.rstrip() for line in raw.splitlines()) + b'\n'
            assert xml_tokens(ET.fromstring(raw)) == xml_tokens(ET.fromstring(clean))
            destination.write_bytes(clean)
            if clean != raw:
                normalizations.append({'file': 'report/' + source.name, 'original_sha256': sha(source),
                    'normalized_sha256': sha(destination), 'XML_token_content_identical': True})
        else:
            shutil.copyfile(source, destination)
    (out / 'text-normalization.json').write_text(json.dumps(normalizations, indent=2))
    selected = json.loads((report / 'demo-selection.json').read_text())
    lookup = {row['case']: row for row in index}
    for case in selected['case_names']:
        row = lookup[case]
        shutil.copyfile(results / row['record'], out / 'demo' / (row['specification']['method'] + '.json.gz'))
    process = processes[lookup[selected['case_names'][0]]['process_key']]
    shutil.copyfile(results / process['path'], out / 'demo/process.npz')
    manifest = {}
    for path in sorted(out.rglob('*')):
        if path.is_file():
            manifest[str(path.relative_to(out))] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    (out / 'transfer-manifest.json').write_text(json.dumps(manifest, indent=2))
    archive = root / 'delivery.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(out, arcname='evidence')
    result = {'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive),
              'original_files': len(manifest), 'formal_episodes': len(index),
              'complete_unique_process_arrays': len(processes),
              'final_policy_weights': 6, 'superseded_policy_weights': 6}
    (root / 'delivery-storage.json').write_text(json.dumps(result, indent=2))
    print('M11_PACKAGE_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
