"""Compact startup evidence; complete pilot records remain hash indexed."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokens(node):
    return (node.tag, sorted((k, ' '.join(v.split())) for k, v in node.attrib.items()),
            ' '.join((node.text or '').split()), tuple(tokens(c) for c in node))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root
    ids = json.loads((root / 'job-identities.json').read_text())
    audit = json.loads((root / f"full-audit-{ids['audit_report']}.json").read_text())
    assert audit['all_pilot_cases_replayed'] == 128
    assert audit['protected_cases'] == 96 and audit['maximum_protected_exact_gap_ratio'] <= 1
    results = root / f"pilot-{ids['pilot']}"
    report = root / f"report-{ids['audit_report']}"
    out = root / 'public-evidence'
    out.mkdir(exist_ok=False)
    for name in ('jobs', 'report', 'demo'):
        (out / name).mkdir()
    copies = {
        root / 'job-identities.json': 'job-identities.json',
        root / 'development-gate.json': 'development-gate.json',
        root / 'm11-targeted-review.json': 'm11-targeted-review.json',
        root / f"full-audit-{ids['audit_report']}.json": 'full-audit.json',
        results / 'contract.json': 'contract.json',
        results / 'summary.json': 'pilot-summary.json',
        results / 'case-index.json': 'case-index.json',
        results / 'process-index.json': 'process-index.json',
        root / f"development-{ids['development']}" / 'case-index.json': 'development-case-index.json',
        root / f"development-{ids['development']}" / 'summary.json': 'development-summary.json',
        root / 'review-m11.py': 'jobs/review-m11.py',
        root / 'review.sbatch': 'jobs/review.sbatch',
        root / 'audit-report.sbatch': 'jobs/audit-report.sbatch',
        root / 'package.sbatch': 'jobs/package.sbatch',
        root / 'ledger-before-package.psv': 'slurm-before-package.psv'
    }
    for key, prefix in (('review', 'review'), ('development', 'development'),
                        ('pilot', 'pilot'), ('audit_report', 'audit-report')):
        for extension in ('out', 'err'):
            name = f'{prefix}-{ids[key]}.{extension}'
            copies[root / name] = 'jobs/' + name
    for source, relative in copies.items():
        shutil.copyfile(source, out / relative)
    normalizations = []
    for source in report.iterdir():
        target = out / 'report' / source.name
        raw = source.read_bytes()
        if source.suffix == '.svg':
            clean = b'\n'.join(line.rstrip() for line in raw.splitlines()) + b'\n'
            assert tokens(ET.fromstring(raw)) == tokens(ET.fromstring(clean))
            target.write_bytes(clean)
            normalizations.append({'file': 'report/' + source.name,
                'original_sha256': sha(source), 'normalized_sha256': sha(target),
                'XML_token_content_identical': True})
        else:
            shutil.copyfile(source, target)
    (out / 'text-normalization.json').write_text(json.dumps(normalizations, indent=2))
    index = json.loads((results / 'case-index.json').read_text())
    processes = json.loads((results / 'process-index.json').read_text())
    storage = {'cases': [dict(case=r['case'], path=str(results / r['path']), bytes=r['bytes'], sha256=r['sha256']) for r in index],
        'process_arrays': [dict(v, path=str(results / v['path'])) for v in processes.values()],
        'frozen_temperature_estimator': {'path': str(root / 'frozen-estimator.pt'),
                                         'sha256': sha(root / 'frozen-estimator.pt')},
        'all128_cases_and16_process_arrays_read_by_independent_audit': True,
        'retrieval_scope': 'All compact case metrics and return-plan counts; four full demo records and one process NPZ. All complete pilot records remain remotely indexed.'}
    (out / 'raw-storage.json').write_text(json.dumps(storage, indent=2))
    selection = json.loads((report / 'demo-selection.json').read_text())
    lookup = {r['case']: r for r in index}
    assert len(selection['cases']) == 4
    for case in selection['cases']:
        item = lookup[case]
        shutil.copyfile(results / item['path'], out / 'demo' / (item['specification']['selector'] + '.json.gz'))
    process = processes[lookup[selection['cases'][0]]['process_key']]
    shutil.copyfile(results / process['path'], out / 'demo/process.npz')
    manifest = {str(path.relative_to(out)): {'bytes': path.stat().st_size, 'sha256': sha(path)}
                for path in sorted(out.rglob('*')) if path.is_file()}
    (out / 'transfer-manifest.json').write_text(json.dumps(manifest, indent=2))
    archive = root / 'delivery.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(out, arcname='evidence')
    receipt = {'sha256': sha(archive), 'bytes': archive.stat().st_size,
               'original_files': len(manifest), 'pilot_cases': 128, 'new_policy_training': False}
    (root / 'delivery-storage.json').write_text(json.dumps(receipt, indent=2))
    print('M111_STARTUP_PACKAGE_COMPLETE', json.dumps(receipt))


if __name__ == '__main__':
    main()
