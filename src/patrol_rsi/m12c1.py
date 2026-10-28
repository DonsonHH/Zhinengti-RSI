"""C1-A frozen public-cue controls and evaluator-only one-detour diagnosis."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .m12 import actor_scorer, load_actors, sha, write
from .m11 import frozen_predictor, seed_digest
from .m12b import specification, TEST
from .opportunity_run import run_public, run_evaluator_branch

METHODS = ('tour', 'cue_rule', 'actor')


def contract():
    return {'stage': 'M12-C1A local decision-opportunity diagnosis',
        'methods': list(METHODS), 'estimators': ['nominal', 'history'], 'complete_main_cases': 192,
        'physical_processes': 16, 'observation_conditions': 32, 'formal_seeds': [7700, 7701], 'smoke_seed': 7699,
        'noise_c': [.2, 1.], 'probe_tour_cases': 64,
        'branch_rule': 'One independent branch for first feasible optional witness of each tour-missed segment; do not combine branches.',
        'probe_scope': 'Finite one-detour witnesses on original tour states, not a globally optimal schedule, an upper bound or a proof of impossibility.',
        'signal_scope': 'Prespecified public profession cue family plus thermal raw telemetry and model disagreement; absence is not information-theoretic absence.',
        'actor': 'Existing M11 history-seed0 deployed unchanged in both estimation modes',
        'estimator': 'Existing M10 history-seed1 checkpoint', 'new_training': False,
        'LLM_calls': 0, 'improver_updated': False,
        'isolation': 'Policy accepts only explicit public snapshot; hidden label matching and override selection are evaluator-only'}


def specs(phase):
    if phase == 'smoke':
        physical = [specification(phase, a, b, 7699) for a, b in (('healthy', 900), ('coldplate_fouling', 180))]
    elif phase == 'formal':
        physical = [specification(phase, a, b, seed, noise) for a, b in TEST for seed in (7700, 7701) for noise in (.2, 1.)]
    else:
        raise ValueError('Unknown C1-A split')
    return [dict(p, estimator=estimator, controller=method) for p in physical
        for estimator in ('nominal', 'history') for method in METHODS]


def name(spec):
    return '--'.join(str(spec[k]) for k in ('phase', 'scenario', 'duration_s', 'seed', 'noise_c', 'estimator', 'controller'))


def branch_targets(record):
    result = []
    for row in record['evaluator_probe']['segments']:
        if row['baseline_detected'] or row['first_witness'] is None:
            continue
        w = row['first_witness']
        result.append({'segment_index': row['segment_index'], 'intervention': {
            'dispatch_time_s': w['dispatch_time_s'], 'agent': w['agent'], 'task': row['task']}})
    return result


def main():
    p = argparse.ArgumentParser()
    for key in ('output', 'actors', 'estimator-weights', 'estimator-manifest'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--phase', choices=('smoke', 'formal'), required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    predictor = frozen_predictor(args.estimator_weights, args.estimator_manifest)
    models, weights = load_actors(args.actors)
    scorer = actor_scorer(models[0])
    args.output.mkdir(parents=True, exist_ok=False)
    for sub in ('cases', 'process-traces'):
        (args.output / sub).mkdir()
    write(args.output / 'contract.json', contract())
    write(args.output / 'immutable-inputs.json', {'actors': weights, 'estimator_sha256': sha(args.estimator_weights),
        'job_id': os.environ.get('SLURM_JOB_ID'), 'torch': torch.__version__, 'numpy': np.__version__})
    index, processes, baseline_records, branch_index = [], {}, {}, []
    def save(case, spec, record, env, source=None):
        path = args.output / 'cases' / (case + '.json.gz')
        path.write_bytes(gzip.compress((json.dumps(record) + '\n').encode(), mtime=0))
        process = f"{spec['scenario']}--{spec['duration_s']}--{spec['seed']}--{spec['fault_start_s']}"
        if process not in processes:
            destination = args.output / 'process-traces' / (process + '.npz')
            np.savez_compressed(destination, **env.trace.arrays)
            processes[process] = {'path': str(destination.relative_to(args.output)), 'sha256': sha(destination),
                'bytes': destination.stat().st_size, 'process': env.trace.metadata}
        elif processes[process]['process']['numeric_digest'] != env.trace.fingerprint:
            raise RuntimeError('Policy or branch changed physical process')
        row = {'case': case, 'specification': spec, 'path': str(path.relative_to(args.output)),
            'sha256': sha(path), 'bytes': path.stat().st_size, 'process_key': process,
            'metrics': record['metrics'], 'public_selection': record['public_selection'],
            'by_role': record['by_role'], 'anomaly_intervals': record['anomaly_intervals'],
            'event_weight_sum': sum(r['weight'] for r in record['anomaly_intervals']),
            'weighted_delay_sum': sum(r['weight'] * r['delay_or_censor_s'] for r in record['anomaly_intervals']),
            'evaluator_probe': record['evaluator_probe'], 'counterfactual_source': source}
        index.append(row)
        return row
    for number, spec in enumerate(specs(args.phase)):
        case = name(spec)
        record, env = run_public(spec, predictor, scorer, spec['controller'], probe_enabled=spec['controller'] == 'tour')
        save(case, spec, record, env)
        if spec['controller'] == 'tour':
            baseline_records[case] = record
        if number % 12 == 0:
            print('C1A_MAIN', number + 1, '/', len(specs(args.phase)), flush=True)
    write(args.output / 'main-stage-complete.json', {'cases': len(index), 'probe_baselines': len(baseline_records),
        'all_main_cases_finished_before_branch_evaluation': True})
    for base_case, baseline in baseline_records.items():
        for target in branch_targets(baseline):
            spec = dict(baseline['specification'], controller='evaluator_one_detour')
            case = base_case + '--branch--' + str(target['segment_index'])
            record, env = run_evaluator_branch(spec, predictor, scorer, target['intervention'])
            segment = target['segment_index']
            before, after = baseline['anomaly_intervals'][segment], record['anomaly_intervals'][segment]
            if before['task'] != after['task'] or before['onset_s'] != after['onset_s'] or not after['detected']:
                raise RuntimeError('Local witness failed in actual counterfactual execution')
            source = {'baseline_case': base_case, 'segment_index': segment, 'intervention': target['intervention'],
                'scope': 'One independent evaluator-only action; not a deployable policy or combined rescue count'}
            save(case, spec, record, env, source)
            branch_index.append({'case': case, **source, 'target_detected': after['detected'],
                'total_missed_delta': record['metrics']['missed_segments'] - baseline['metrics']['missed_segments'],
                'travel_delta_m': record['metrics']['travel_m'] - baseline['metrics']['travel_m'],
                'service_delta_s': record['metrics']['service_s'] - baseline['metrics']['service_s'],
                'baseline_intervals': baseline['anomaly_intervals'], 'branch_intervals': record['anomaly_intervals']})
    for seed, model in models.items():
        if seed_digest(model) != weights[str(seed)]['parameters_sha256']:
            raise RuntimeError('Frozen actor parameters changed')
    write(args.output / 'case-index.json', index)
    write(args.output / 'process-index.json', processes)
    write(args.output / 'branch-index.json', branch_index)
    write(args.output / 'summary.json', {'phase': args.phase, 'job_id': os.environ.get('SLURM_JOB_ID'),
        'main_cases': len(specs(args.phase)), 'evaluator_branches': len(branch_index), 'all_actual_cases': len(index),
        'physical_processes': len(processes), 'probe_tour_cases': len(baseline_records),
        'all_weights_unchanged': True, 'LLM_calls': 0, 'new_training': False,
        'seconds': time.perf_counter() - started})
    print('C1A_COMPLETE', len(index), 'cases', len(branch_index), 'branches', flush=True)


if __name__ == '__main__':
    main()
