"""Portable verification of the bounded local queue-diagnosis deliverable."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def read(path):
    return json.loads(path.read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args();root = args.workspace.resolve()
    base = root / 'research/2026-10-29-m12c1b3-collateral'
    frozen = read(base / 'evidence/protected-baseline.json')
    for name, expected in frozen['files'].items():
        raw = (root / name).read_bytes()
        if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
            raise RuntimeError('Preexisting tracked file changed: ' + name)
    final = base / 'evidence/diagnosis-v2'
    # Re-execute the public adapter/operator and independent schedule audit on
    # the exact original records, without any simulator or neural inference.
    with tempfile.TemporaryDirectory(prefix='queue-diagnosis-') as temporary:
        output = Path(temporary) / 'replay'
        subprocess.run([sys.executable, str(root / 'scripts/diagnose_queue_cost.py'),
            '--evidence', str(root / 'research/2026-10-29-m12c1b2-confirm/evidence'),
            '--output', str(output)], cwd=root, check=True, stdout=subprocess.PIPE)
        for name in ('public-queue-diagnostics.json', 'offline-queue-review.json'):
            if read(output / name) != read(final / name):
                raise RuntimeError('Recomputed diagnostic differs: ' + name)
        summary = read(output / 'execution-summary.json')
        saved = read(final / 'execution-summary.json')
        if {k:v for k,v in summary.items() if k != 'seconds'} != {k:v for k,v in saved.items() if k != 'seconds'}:
            raise RuntimeError('Reproduction scope differs')
    public = read(final / 'public-queue-diagnostics.json')
    forbidden = {'scenario', 'seed', 'fault_start_s', 'fault_end_s', 'anomaly_intervals',
        'first_confirmation_s', 'actual_missed_delta', 'offline_outcome', 'future_labels', 'specification', 'original_case'}
    def inspect(value):
        if isinstance(value, dict):
            if forbidden.intersection(value):
                raise RuntimeError('Hidden-evaluation field in public diagnostic')
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)
    inspect(public)
    offline = read(final / 'offline-queue-review.json')
    original_inputs = read(final / 'diagnostic-inputs.json')['cases']
    for inputs, before, after in zip(original_inputs, public['cases'], offline['cases']):
        if not inputs['case_alias'] == before['case_alias'] == after['case_alias']:
            raise RuntimeError('Public/offline aliases do not match')
        for key in ('source', 'baseline_source'):
            # Saved paths can be repository-relative or from the generating
            # machine; use the original basename and public fixture folders.
            source = inputs[key]
            basename = Path(source['path']).name
            paths = list((root / 'research/2026-10-29-m12c1b2-confirm/evidence').glob('*/' + basename))
            if len(paths) != 1 or hashlib.sha256(paths[0].read_bytes()).hexdigest() != source['sha256']:
                raise RuntimeError('Original development source no longer matches')
    for path in base.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if '://' not in target and not target.startswith('#') and not (path.parent / target).exists():
                raise RuntimeError('Broken local document link: ' + target)
    state = read(base / '阶段状态.json')
    if any(state[k] != 0 for k in ('new_rollouts', 'new_neural_training', 'LLM_calls', 'new_Slurm_jobs')):
        raise RuntimeError('Local scope changed')
    if (base / 'evidence/run-manifest.json').exists():
        for name, expected in read(base / 'evidence/run-manifest.json')['files'].items():
            raw = (root / name).read_bytes()
            if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
                raise RuntimeError('Final deliverable bytes changed: ' + name)
    result = {'baseline': frozen['commit'], 'all_protected_files_unchanged': len(frozen['files']),
        'public_and_offline_diagnostics_reproduced_exactly': True,
        'recorded_branches': len(public['cases']), 'independent_completion_checks': saved['independent_Dijkstra_and_actual_completion_checks'],
        'no_hidden_evaluation_keys_in_public_view': True, 'no_new_rollouts_training_or_model_calls': True,
        'scope': 'Local queue reconstruction and source preservation, not new policy or field validation.'}
    with args.output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, ensure_ascii=False, indent=2);f.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
