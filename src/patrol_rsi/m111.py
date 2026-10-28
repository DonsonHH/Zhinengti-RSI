"""M11.1 startup slice: explicit return plans, task evidence, fixed pilot."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .covered_inspection import SELECTORS, run_covered
from .inspection_model import TaskPolicy
from .m11 import frozen_predictor, seed_digest

FIXTURES = (('healthy', 900), ('coldplate_fouling', 60), ('coldplate_fouling', 180),
            ('coldplate_fouling', 900), ('dc_contact_hotspot', 900),
            ('it_feeder_loss', 180), ('it_feeder_loss', 900),
            ('aux_feeder_loss+cdu_fouling', 180))


def specs(phase):
    fixtures = FIXTURES if phase == 'pilot' else (('healthy', 900), ('coldplate_fouling', 180))
    seeds = (5100, 5101) if phase == 'pilot' else (5000,)
    selectors = SELECTORS if phase == 'pilot' else ('cycle', 'risk_certified', 'untrained_certified')
    return [dict(phase=phase, seed=seed, scenario=scenario, duration_s=duration,
                 control='feedback', resources='provisioned', variant='open',
                 estimator=estimator, selector=selector)
            for scenario, duration in fixtures for seed in seeds
            for estimator in ('nominal', 'history') for selector in selectors]


def name(spec):
    return '--'.join(str(spec[k]) for k in ('phase', 'scenario', 'duration_s', 'seed', 'estimator', 'selector'))


def contract():
    return {'stage': 'M11.1 startup prototype', 'new_policy_training': False,
            'protected_ownership': 'Original baseline owner sets for every comparator; no partner takeover',
            'conditions': [list(v) for v in FIXTURES], 'pilot_seeds': [5100, 5101],
            'development_seeds': [5000], 'pilot_cases': 128, 'physical_conditions': 16,
            'selectors': list(SELECTORS), 'estimators': ['nominal', 'history'],
            'untrained_score_seed': 77, 'history_estimator_retrained': False,
            'gate': 'Candidate completion plus complete baseline continuation; repeated cycle within all owned deadlines',
            'exact_gap_metric': 'Initial time0, actual completion timestamps and terminal1800 all included',
            'remote_power': 'Declared calibrated simulation power channels; over10percent deficit for3 consecutive5s readings; cause undiagnosed',
            'unsupported_measurement_channels': ['actual_voltage', 'actual_branch_current'],
            'scope': 'Known static layout, deterministic service/travel; no collision, communication latency or real deployment guarantee'}


def untrained_scorer():
    torch.manual_seed(77)
    model = TaskPolicy()
    model.eval()
    initial = seed_digest(model)
    def score(features):
        with torch.inference_mode():
            return model.actor(torch.as_tensor(features)).squeeze(-1).numpy()
    return score, model, initial


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=('development', 'pilot'), required=True)
    p.add_argument('--estimator-weights', type=Path, required=True)
    p.add_argument('--estimator-manifest', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    predictor = frozen_predictor(args.estimator_weights, args.estimator_manifest)
    scorer, model, initial = untrained_scorer()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'cases').mkdir()
    (args.output / 'process-traces').mkdir()
    index, process_index = [], {}
    for number, spec in enumerate(specs(args.phase)):
        record, env = run_covered(spec, predictor, scorer)
        if seed_digest(model) != initial:
            raise RuntimeError('Untrained scoring model was modified')
        case = name(spec)
        path = args.output / 'cases' / (case + '.json.gz')
        path.write_bytes(gzip.compress((json.dumps(record) + '\n').encode(), mtime=0))
        process = '--'.join(str(spec[k]) for k in ('scenario', 'duration_s', 'seed'))
        if process not in process_index:
            destination = args.output / 'process-traces' / (process + '.npz')
            np.savez_compressed(destination, **env.trace.arrays)
            process_index[process] = {'path': str(destination.relative_to(args.output)),
                'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                'bytes': destination.stat().st_size, 'process': env.trace.metadata}
        index.append({'case': case, 'specification': spec, 'path': str(path.relative_to(args.output)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size,
            'process_key': process, 'metrics': record['metrics'], 'by_role': record['by_role'],
            'service_evidence_event_counts': {kind: sum(e['kind'] == kind for e in record['service_evidence_events'])
                for kind in ('new_measured_alert', 'persistent_recheck', 'recovery_confirmation')}})
        print('M111_CASE_COMPLETE', number + 1, case, 'exact_gap_violations',
              record['metrics']['tasks_violating_exact_gap'], flush=True)
    for filename, content in [('case-index.json', index), ('process-index.json', process_index),
        ('contract.json', contract()), ('summary.json', {'phase': args.phase, 'cases': len(index),
         'unique_processes': len(process_index), 'new_policy_training': False,
         'untrained_model_parameters_sha256': initial, 'job_id': os.environ.get('SLURM_JOB_ID')})]:
        with (args.output / filename).open('x') as f:
            json.dump(content, f, indent=2)
            f.write('\n')


if __name__ == '__main__':
    main()
