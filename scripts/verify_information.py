"""Independent complete archive, history, and analytic-event verification."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from patrol_rsi.active_inspection import ProcessTrace, graph_for, measured_anomaly
from patrol_rsi.causal_observation import validate_public
from patrol_rsi.facility import FacilityConfig
from patrol_rsi.information_audit import PublicSignals


FOLDERS = ('development-0', 'development-1', 'validation-0', 'validation-1',
           'test-0', 'test-1', 'test-2', 'test-3')


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(value, message):
    if not value:
        raise RuntimeError(message)


def direct_event_evaluation(record, geometry, arrays):
    """Shift each archived inspection explicitly, then measure at its new time.

    Independent of the analytic evaluator's grouped-shift/searchsorted method.
    Original measurement/alarm functions provide the profession-specific test.
    """
    tasks = graph_for(128, 'open').tasks
    config = FacilityConfig()
    trace = SimpleNamespace(arrays=arrays, config=config, seed=record['specification']['seed'])
    n, windows = len(geometry['task']), record['anomaly_intervals']
    reference = np.concatenate((arrays['rack_temperature_c'] > 65.,
        arrays['contact_temperature_c'] > 60.,
        arrays['rack_flow_kg_s'].reshape(360, 16, 8).sum(2) < 33.6,
        np.abs((arrays['public_sensor_temperature_c'] - arrays['rack_temperature_c']).reshape(360, 16, 8).mean(2)) > 1.), axis=1)
    confirms = reference.copy()
    for tick, task in np.argwhere(reference):
        measurement = ProcessTrace.measurement(trace, tasks[int(task)], int(tick))
        confirms[tick, task] = measured_anomaly(tasks[int(task)], measurement)
    first = np.full((n, len(windows)), -1, dtype=np.int16)
    events = record['inspection_events']
    for column, window in enumerate(windows):
        task = window['task']
        original = [v for v in events if v['task'] == task]
        owner = original[0]['agent']
        old_times = np.asarray([v['time_s'] for v in original])[None, :]
        changed = (geometry['agent'][:, None] == owner) & (old_times > geometry['dispatch_time_s'][:, None])
        times = np.where(changed, old_times + geometry['shift_s'][:, None], old_times)
        valid = (times <= 1800.) & (times >= window['onset_s']) & (times <= window['end_s'])
        ticks = np.minimum(359, (times / 5).astype(int) - 1)
        valid &= confirms[ticks, task]
        earliest = np.where(valid, times, np.inf).min(axis=1)
        extra = geometry['finish_s']
        extra_valid = (geometry['task'] == task) & (extra >= window['onset_s']) & (extra <= window['end_s'])
        extra_valid &= confirms[np.minimum(359, (extra / 5).astype(int) - 1), task]
        earliest = np.minimum(earliest, np.where(extra_valid, extra, np.inf))
        first[:, column] = np.where(np.isfinite(earliest), earliest, -1).astype(np.int16)
    detected = first >= 0
    old_detected = np.asarray([v['detected'] for v in windows], dtype=bool)
    misses = (~detected).sum(1) - int((~old_detected).sum())
    weights = np.asarray([v['weight'] for v in windows])
    if weights.sum():
        onset = np.asarray([v['onset_s'] for v in windows])
        end = np.asarray([v['end_s'] for v in windows])
        delay = np.where(detected, first - onset, end - onset)
        base = np.asarray([v['delay_or_censor_s'] for v in windows])
        wait = ((delay - base) * weights).sum(1) / weights.sum()
    else:
        wait = np.full(n, np.nan)
    return first, misses, wait


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--formal-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    counts = {'source_cases': 0, 'public_frames': 0, 'original_public_SHA_frames': 0,
              'analytic_actions': 0, 'all_event_confirmation_labels': 0, 'CVAE_zero_signal_proof_frames': 0}
    per_folder = {}
    for name in FOLDERS:
        folder = args.root / name
        source = args.formal_root / name
        rows = read(source / 'case-index.json')
        processes = read(source / 'process-index.json')
        with np.load(folder / 'geometry.npz') as data:
            geometry = {k: data[k].copy() for k in data.files}
        tours = [v for v in rows if v['alias'] == 'tour']
        need(len(tours) == 16, 'Original Tour condition count')
        for index, tour_row in enumerate(tours):
            path = folder / f'condition-{index:02}'
            summary = read(path / 'summary.json')
            need(summary['condition'] == tour_row['specification'], 'Condition ordering/source identity')
            original_case_path = source / tour_row['path']
            need(sha(original_case_path) == tour_row['sha256'], 'Original Tour case hash')
            tour = read(original_case_path)
            key = tuple(tour_row['specification'].items())
            same_conditions = [v for v in rows if v['specification'] == tour_row['specification']]
            original_records = []
            for row in same_conditions:
                p = source / row['path']
                need(sha(p) == row['sha256'], 'Original method case hash')
                original_records.append(read(p))
            counts['source_cases'] += len(original_records)
            original_stamps = {}
            for record in original_records:
                for stamp in record['causal_snapshot_stamps']:
                    original_stamps.setdefault(stamp['time_s'], []).append(stamp['sha256'])
            frames = 0
            with gzip.open(path / 'public-states.jsonl.gz', 'rt', encoding='utf-8') as stream:
                for line in stream:
                    bundle = json.loads(line)
                    validate_public(bundle)
                    now = bundle['time_s']
                    need(now == (frames + 1) * 5, 'Missing/repeated public frame')
                    digest = hashlib.sha256(json.dumps(bundle, sort_keys=True).encode()).hexdigest()
                    need(all(v == digest for v in original_stamps.get(now, [])), 'Independent original public SHA comparison')
                    counts['original_public_SHA_frames'] += len(original_stamps.get(now, []))
                    # Establish the actual missingness premise and both-endpoint
                    # bound before interpreting equal fixed-history scores.
                    public = PublicSignals(bundle)
                    need(np.isfinite(public.channels['nominal'][:128]).all(), 'No thermal alternate source available')
                    can_change = False
                    for delta in (0., 1800. - now):
                        full, _ = public.at(np.arange(128), delta)
                        drop, _ = public.at(np.arange(128), delta, drop_history=True)
                        can_change |= bool(np.any(np.abs(full - drop) > 1e-12))
                    counts['CVAE_zero_signal_proof_frames'] += int(not can_change)
                    frames += 1
            need(frames == 359, 'Incomplete reconstructed public history')
            counts['public_frames'] += frames
            process = processes[tour_row['process_key']]
            need(sha(source / process['path']) == process['sha256'], 'Original physical array SHA')
            with np.load(source / process['path']) as data:
                arrays = {k: data[k].copy() for k in data.files}
            first, misses, wait = direct_event_evaluation(tour, geometry, arrays)
            with np.load(path / 'analytic-continuations.npz') as oracle:
                need(np.array_equal(first, oracle['first_confirmation_s']), 'Independent all-action event confirmation matrix')
                need(np.array_equal(misses, oracle['missed_delta']), 'Independent all-action whole-mission misses')
                need(np.allclose(wait, oracle['weighted_wait_delta_s'], rtol=1e-10, atol=1e-8, equal_nan=True),
                     'Independent all-action weighted whole-mission wait')
                helpful = (misses < 0) | ((misses == 0) & (wait < -1e-9))
                need(np.array_equal(helpful, oracle['helpful']), 'Independent all-action beneficial labels')
            counts['analytic_actions'] += len(misses)
            counts['all_event_confirmation_labels'] += first.size
        per_folder[name] = {'conditions': len(tours), 'source_cases': len(rows)}
        print('INDEPENDENT_INFORMATION_VERIFIED', name, counts, flush=True)
    need(counts['source_cases'] == 768 and counts['public_frames'] == 128 * 359, 'Full independent coverage')
    result = {'independent_verification': 'passed', 'counts': counts, 'folders': per_folder,
              'scope': 'All 128 public histories checked against archived hashes where recorded; all 768 originals hash-bound; all hard-legal analytic action/event outcomes independently recomputed with original measurement/alarm functions. No policy trial or parameter selection.'}
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('INDEPENDENT_INFORMATION_ALL_PASSED', flush=True)


if __name__ == '__main__':
    main()
