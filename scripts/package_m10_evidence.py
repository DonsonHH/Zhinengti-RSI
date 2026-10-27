"""Verify GP-only revision and package compact, auditable M10 evidence."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import xml.etree.ElementTree as ET

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def xml_tokens(node):
    # SVG path coordinates remain identical; collapse only lexical whitespace.
    return (node.tag, sorted((k, ' '.join(v.split())) for k, v in node.attrib.items()),
            ' '.join((node.text or '').split()), tuple(xml_tokens(child) for child in node))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    ids = json.loads((root / 'job-identities.json').read_text())
    train = root / f"training-data-{ids['train']}"
    models = root / f"models-{ids['train']}"
    data = root / f"test-data-{ids['evaluate_v2']}"
    results = root / f"evaluation-{ids['evaluate_v2']}"
    old_data = root / f"test-data-{ids['evaluate']}"
    old_results = root / f"evaluation-{ids['evaluate']}"
    report = root / f"report-{ids['audit_report']}"
    audit = json.loads((root / f"full-audit-{ids['audit_report']}.json").read_text())
    assert audit['all_dataset_cases_regenerated'] == 598
    assert audit['independent_method_condition_metrics'] == 3498
    index = json.loads((data / 'index.json').read_text())
    unchanged_data_keys, unchanged_prediction_keys = set(), set()
    for item in index:
        with np.load(data / item['path'], allow_pickle=False) as new, np.load(old_data / item['path'], allow_pickle=False) as old:
            for key in new.files:
                if 'seconds' not in key and key != 'pred_gp64':
                    np.testing.assert_equal(new[key], old[key])
                    unchanged_data_keys.add(key)
        filename = item['case'] + '.npz'
        with np.load(results / 'private-predictions' / filename, allow_pickle=False) as new, np.load(old_results / 'private-predictions' / filename, allow_pickle=False) as old:
            for key in new.files:
                if key != 'gp64':
                    np.testing.assert_equal(new[key], old[key])
                    unchanged_prediction_keys.add(key)
    out = root / 'public-evidence-v2'
    out.mkdir(exist_ok=False)
    (out / 'jobs').mkdir()
    (out / 'report').mkdir()
    (out / 'demo').mkdir()
    originals = []
    for folder, prefix in ((train, 'training'), (data, 'test')):
        raw = (folder / 'index.json').read_bytes()
        (out / (prefix + '-data-index.json.gz')).write_bytes(gzip.compress(raw, mtime=0))
        originals.append({'phase': prefix, 'remote_index': str(folder / 'index.json'),
                          'uncompressed_index_sha256': hashlib.sha256(raw).hexdigest(),
                          'files': json.loads(raw), 'numeric_arrays_downloaded': False})
    # Full compressed indices retain all measurement events. Compact storage
    # entries avoid duplicating that larger metadata inside the public ledger.
    storage = {'dataset_cases': [], 'prediction_files': [], 'selected_weights': []}
    for record in originals:
        for item in record['files']:
            storage['dataset_cases'].append({'case': item['case'],
                'path': str((train if record['phase'] == 'training' else data) / item['path']),
                'bytes': item['bytes'], 'sha256': item['sha256'], 'numeric_digest': item['numeric_digest']})
    for item in json.loads((results / 'prediction-index.json').read_text()):
        storage['prediction_files'].append(dict(item, path=str(results / item['path'])))
    training = json.loads((models / 'training.json').read_text())
    for item in training['training_records']:
        path = models / item['checkpoint']
        storage['selected_weights'].append({'method': item['method'], 'seed': item['seed'],
                                            'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    storage['all_arrays_and_selected_weights_read_by_audit'] = True
    storage['array_download_scope'] = 'Two exact demonstration arrays only; full data/predictions retained remotely'
    (out / 'raw-storage.json').write_text(json.dumps(storage, indent=2))
    (out / 'data-index-storage.json').write_text(json.dumps(
        [{k: v for k, v in r.items() if k != 'files'} for r in originals], indent=2))
    (out / 'version-proof.json').write_text(json.dumps({
        'cases_compared': 318, 'v1_and_v2_common_data_keys_exact': sorted(unchanged_data_keys),
        'v1_and_v2_nonGP_predictions_exact': sorted(unchanged_prediction_keys),
        'all_six_CVAE_outputs_identical_after_GP_revision': True,
        'superseded_metrics_remote_path': str(old_results / 'metrics.json'),
        'superseded_metrics_sha256': sha(old_results / 'metrics.json'),
        'reason': 'GP sampling and observation-derived mean corrected before formal score inspection; retained v1 proves all other predictions unchanged'}, indent=2))
    copies = {
        root / 'job-identities.json': 'job-identities.json',
        root / 'development-gate.json': 'development-gate.json',
        root / 'training-gate.json': 'training-gate.json',
        root / f"m9-full-review-{ids['review']}.json": 'm9-full-review.json',
        root / f"full-audit-{ids['audit_report']}.json": 'full-audit.json',
        root / f"smoke-models-{ids['smoke']}" / 'training.json': 'development-training.json',
        models / 'training.json': 'training.json',
        results / 'metrics.json': 'metrics.json',
        results / 'summary.json': 'evaluation-summary.json',
        results / 'prediction-index.json': 'prediction-index.json',
        data / 'contract.json': 'contract.json',
        root / 'review-m9.sbatch': 'jobs/review-m9.sbatch',
        root / 'evaluate.sbatch': 'jobs/evaluate-v1.sbatch',
        root / 'evaluate-v2.sbatch': 'jobs/evaluate-v2.sbatch',
        root / 'audit-report.sbatch': 'jobs/audit-report.sbatch',
        root / 'package.sbatch': 'jobs/package.sbatch',
        root / 'package-v2.sbatch': 'jobs/package-v2.sbatch',
    }
    prefixes = {'review': 'review', 'smoke': 'smoke', 'train': 'train', 'evaluate': 'evaluate',
                'evaluate_v2': 'evaluate-v2', 'audit_report': 'audit-report', 'package_failed': 'package'}
    for kind, prefix in prefixes.items():
        for extension in ('out', 'err'):
            name = f'{prefix}-{ids[kind]}.{extension}'
            copies[root / name] = 'jobs/' + name
    for source, relative in copies.items():
        shutil.copyfile(source, out / relative)
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
    selected = next(r for r in index if r['case'] == 'test--nominal--aux_feeder_loss+cdu_fouling--feedback--300')
    shutil.copyfile(data / selected['path'], out / 'demo/data.npz')
    shutil.copyfile(results / 'private-predictions' / (selected['case'] + '.npz'), out / 'demo/predictions.npz')
    (out / 'demo/selection.json').write_text(json.dumps({'case': selected['case'],
        'frame': 29, 'time_s': 900, 'selection': 'Predeclared scenario/seed/time, not selected by performance'}, indent=2))
    # Compute nodes do not have access to the configured local accounting DB.
    # Capture lightweight scheduler metadata on mu01, then consume it offline.
    shutil.copyfile(root / 'ledger-before-package-v2.psv', out / 'slurm-before-packaging.psv')
    manifest = {}
    for path in sorted(out.rglob('*')):
        if path.is_file():
            manifest[str(path.relative_to(out))] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    (out / 'transfer-manifest.json').write_text(json.dumps(manifest, indent=2))
    archive = root / 'delivery-v2.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(out, arcname='evidence')
    result = {'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive),
              'files_before_transfer_manifest': len(manifest), 'all_data_cases': 598,
              'metric_records': 3498, 'same_nonGP_predictions_across_versions': True}
    (root / 'delivery-storage-v2.json').write_text(json.dumps(result, indent=2))
    print('M10_PACKAGE_COMPLETE', json.dumps(result))


if __name__ == '__main__':
    main()
