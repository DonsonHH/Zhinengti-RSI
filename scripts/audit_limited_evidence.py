"""Independent source/cache and frozen-score arithmetic audit; no model calls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
import time

import numpy as np

from patrol_rsi.limited_prediction import CELLS, HeadData, load_reviewed, need, read, sha, write


def close(actual, expected, message):
    if actual is None or expected is None:
        need(actual is expected, message)
    else:
        need(bool(np.isclose(actual, expected, rtol=1e-9, atol=1e-12)), message)


def audit_probabilities(data, row, probability):
    """Recompute released quantities without calling the production evaluator."""
    li = 0 if row['label'] == 'A_physical' else 1
    y = np.asarray(data.y[:data.n, li]); w = np.asarray(data.w[:data.n])
    p = np.asarray(probability); w = w / w.sum(); alarm = p > row['threshold']
    summary = row['summary']; q = np.clip(p, 1e-6, 1 - 1e-6)
    close(summary['Brier'], np.sum(w * (p - y) ** 2), 'Independent weighted Brier')
    close(summary['LogLoss'], -np.sum(w * (y * np.log(q) + (1-y) * np.log1p(-q))), 'Independent logloss')
    close(summary['empirical_FPR'], np.mean(alarm[y == 0]), 'Independent empirical FPR')
    order = np.argsort(p, kind='stable')[::-1]
    pp, yy, ww = p[order], y[order], w[order]
    ends = np.r_[np.flatnonzero(np.diff(pp) != 0), len(pp)-1]
    hit = np.cumsum(yy * ww)[ends]; mass = np.cumsum(ww)[ends]
    ap = np.dot(np.diff(np.r_[0., hit]), hit / mass) / hit[-1] if hit[-1] else None
    close(summary['AUPRC'], ap, 'Independent grouped-tie average precision')
    ece = 0.
    for j, value in enumerate(summary['reliability_bins']):
        mask = (p >= j/10) & ((p < (j+1)/10) if j < 9 else (p <= 1))
        weight = w[mask].sum()
        need(value['n'] == int(mask.sum()), 'Independent reliability bin count')
        close(value['weight'], weight, 'Independent reliability mass')
        if weight:
            mean = np.sum(w[mask] * p[mask]) / weight; rate = np.sum(w[mask] * y[mask]) / weight
            close(value['mean_probability'], mean, 'Independent reliability prediction')
            close(value['observed_fraction'], rate, 'Independent reliability outcome')
            ece += weight * abs(mean-rate)
    close(summary['ECE'], ece, 'Independent ECE')
    meta = np.asarray(data.meta[:data.n]); condition_brier = defaultdict(list)
    condition_recall = defaultdict(list); eligibility = defaultdict(list)
    for cid in np.unique(meta[:, 2]):
        mask = meta[:, 2] == cid; own = meta[mask]; seed, process = map(int, own[0, 3:5])
        b = data.y[:data.n, 1][mask]; ids = own[:, 5]
        eligible = set(ids[(b == 1) & (ids > 0)].tolist())
        detected = set(ids[(b == 1) & (ids > 0) & alarm[mask]].tolist())
        condition_brier[seed, process].append(float(np.mean((p[mask]-y[mask])**2)))
        condition_recall[seed, process].append(len(detected)/len(eligible) if eligible else None)
        eligibility[seed, process].append(eligible)
    calculated = []
    for seed in sorted(set(key[0] for key in condition_brier)):
        briers, recalls = [], []
        for key in sorted(condition_brier):
            if key[0] != seed:
                continue
            need(len(condition_brier[key]) == 2 and eligibility[key][0] == eligibility[key][1], 'Exactly paired noise event denominator')
            briers.append(np.mean(condition_brier[key]))
            if condition_recall[key][0] is not None:
                recalls.append(np.mean(condition_recall[key]))
        own = next(v for v in row['seeds'] if v['seed'] == seed)
        close(own['Brier'], np.mean(briers), 'Independent process/seed Brier')
        close(own['event_recall'], np.mean(recalls) if recalls else None, 'Independent event/process/seed recall')
        calculated.append((np.mean(briers), np.mean(recalls) if recalls else None))
    close(summary['seed_macro_Brier'], np.mean([v[0] for v in calculated]), 'Independent macro Brier')
    rr = [v[1] for v in calculated if v[1] is not None]
    close(summary['seed_macro_event_recall'], np.mean(rr) if rr else None, 'Independent macro recall')


def main():
    parser = argparse.ArgumentParser()
    for name in ('protocol', 'review', 'old-manifest', 'source-root', 'source-inventory', 'cache', 'selection', 'final-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); os.umask(0o077)
    protocol, _ = load_reviewed(args.protocol, args.review); started = time.perf_counter()
    need(sha(args.old_manifest) == protocol['original_manifest_SHA256'], 'Original manifest unchanged')
    need(sha(args.source_inventory) == 'ef404d3ecadb8b6e6b503e6bfac11551a32e9a10d66c117c74648f25acae8065', 'Original all-byte inventory')
    originals = read(args.source_inventory)['files']; checked_bytes = 0
    for item in originals:
        path = Path(item['path'])
        need(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Every original byte still unchanged')
        checked_bytes += item['bytes']
    manifest = read(args.old_manifest)['manifest']; fingerprint_splits = defaultdict(set)
    condition_ids = {v['id']: i for i, v in enumerate(manifest)}
    selection = read(args.selection); opened = read(args.cache/'final-open.json')
    need(opened['selection_SHA256'] == sha(args.selection) and opened['opened_unix'] >= selection['frozen_unix'], 'Chronological final seal')
    for artifact in selection['artifacts']:
        need(sha(Path(artifact['path'])) == artifact['sha256'], 'Every selection artifact unchanged after opening')
    phases = []; coverage = []; model_count = 0
    for phase, seeds in protocol['splits'].items():
        data = [HeadData(args.cache/phase/f'cell-{ci}', expected_phase=phase, expected_cell=ci, expected_protocol=sha(args.protocol)) for ci in range(7)]
        offset = [0]*7; processes = {}; seen = set(); local_counts = defaultdict(list)
        for item in (v for v in manifest if v['split'] == phase):
            folder = args.source_root/phase/f"seed-{item['group_seed']}"/item['id']
            identity = read(folder/'source-identity-not-features.json'); fp = identity['numeric_process_fingerprint']
            fingerprint_splits[fp].add(phase)
            key = item['group_seed'], fp, item['specification']['noise_c']
            if key in seen:
                continue
            seen.add(key); process = processes.setdefault(fp, len(processes))
            with np.load(folder/'public-features.npz') as f:
                base, action = f['base'], f['completion']
            with np.load(folder/'targets-evaluator-only.npz') as f:
                aa, bb = f['A'], f['B_action']
            with np.load(folder/'labels-only.npz') as f:
                raw_id, retained_id = f['event_id'], f['retained_event_id']
            events = read(folder/'events-evaluator-only.json')
            for ci, (kind, horizon) in enumerate(CELLS):
                lower, upper = {'thermal': (0,128), 'electrical': (128,144), 'control': (160,176)}[kind]
                hi = (30,60,120).index(horizon)
                legal = bb[:, hi, lower:upper] >= 0
                roots, tasks_local = np.where(legal); tasks = tasks_local + lower; n = len(tasks)
                sl = slice(offset[ci], offset[ci]+n); cohort = data[ci]
                need(np.array_equal(cohort.x[sl, :85], base[roots, tasks]), 'Independent all original public base features')
                need(np.array_equal(cohort.x[sl, 85:], action[roots, hi, tasks]), 'Independent all original completion features')
                need(np.array_equal(cohort.y[sl, 0], aa[roots, hi, tasks]) and np.array_equal(cohort.y[sl, 1], bb[roots, hi, tasks]), 'Independent exact A and legal B target')
                finish = (roots+1)*30+horizon; ticks = finish//5-1
                expected_columns = (roots, tasks, np.full(n, condition_ids[item['id']]), np.full(n,item['group_seed']),
                                    np.full(n,process), retained_id[ticks,tasks], raw_id[ticks,tasks],finish)
                for column, expected in enumerate(expected_columns):
                    need(np.array_equal(cohort.meta[sl,column],expected), 'Independent evaluator-only metadata alignment')
                need(n > 0 and np.allclose(cohort.w[sl], 1/(len(seeds)*18*n), rtol=0, atol=1e-15), 'Independent seed/process/noise/row weights')
                eligible = np.unique(retained_id[ticks,tasks][bb[roots,hi,tasks] == 1]); eligible = eligible[eligible > 0]
                own = [v for v in events if v['kind'] == kind]
                local_counts[ci].append({'noise': item['specification']['noise_c'], 'process': process,
                    'all_physical_raw': len(own), 'all_physical_retained': sum(v['retained_at_least_15s'] for v in own),
                    'eligible_retained': len(eligible), 'valid_rows': n})
                offset[ci] += n
        need(len(seen) == len(seeds)*18 and all(offset[i] == data[i].n for i in range(7)), 'No cache row absent, repeated or padded into evaluation')
        phases.append({'phase':phase,'conditions':len(seen),'numeric_processes':len(processes),'seed_groups':len(seeds),'rows_by_cell':offset})
        for ci, rows in local_counts.items():
            per_noise = []
            for noise in (.2,1.):
                own = [v for v in rows if v['noise'] == noise]
                per_noise.append({'noise_c':noise,'conditions':len(own),
                    **{key:sum(v[key] for v in own) for key in ('all_physical_raw','all_physical_retained','eligible_retained','valid_rows')},
                    'zero_eligible_processes':sum(v['eligible_retained']==0 for v in own),
                    'eligible_count_distribution':dict(Counter(str(v['eligible_retained']) for v in own))})
            process_rows = [v for v in rows if v['noise'] == .2]
            coverage.append({'phase':phase,'cell':ci,'kind':CELLS[ci][0],'horizon_s':CELLS[ci][1],
                'per_noise':per_noise,'physical_processes':len(process_rows),
                'unique_process_all_raw_events':sum(v['all_physical_raw'] for v in process_rows),
                'unique_process_all_retained_events':sum(v['all_physical_retained'] for v in process_rows),
                'unique_process_eligible_retained_events':sum(v['eligible_retained'] for v in process_rows),
                'zero_eligible_processes':sum(v['eligible_retained']==0 for v in process_rows),
                'per_process_eligible_count_distribution':dict(Counter(str(v['eligible_retained']) for v in process_rows)),
                'per_process_all_retained_count_distribution':dict(Counter(str(v['all_physical_retained']) for v in process_rows)),
                'privacy':'Counts and distributions only;no process fingerprint,event ID,final row or episode trace exported.'})
        if phase == 'final_test':
            for ci, cohort in enumerate(data):
                released = read(args.final_root/f'cell-{ci}'/'released-metrics.json')
                for row in released['models']:
                    path = args.final_root/f'cell-{ci}'/(row['id']+'--private-probabilities.npy')
                    need(sha(path) == row['raw_predictions_SHA256'], 'Original once-scored probability bytes')
                    audit_probabilities(cohort,row,np.load(path,mmap_mode='r')); model_count += 1
    need(all(len(v)==1 for v in fingerprint_splits.values()), 'No numeric physical process shared between splits')
    args.output.mkdir(exist_ok=False)
    write(args.output/'independent-evidence-audit.json',{'status':'passed','protocol_SHA256':sha(args.protocol),
        'selection_SHA256':sha(args.selection),'original_files_rehashed':len(originals),'original_bytes_rehashed':checked_bytes,
        'unique_numeric_processes_all_splits':len(fingerprint_splits),'cross_split_process_overlap':0,'phases':phases,
        'final_frozen_models_arithmetic_recomputed':model_count,'source_cache_feature_target_meta_weight_exact':True,
        'final_row_or_event_ID_exports':0,'model_inference_refits_or_threshold_changes':0,'wall_s':time.perf_counter()-started,
        'method':'Independent loops and arithmetic on immutable source and saved probability vectors;does not call fit,predict,productionevaluate or probability metric helpers.'})
    write(args.output/'event-denominator-diagnostics.json',{'protocol_SHA256':sha(args.protocol),'cells':coverage,
        'event_scope':'Within each predeclared cell,per-noise and unique numeric-process counts. Same events can recur across horizons;do not sum cells as independent events.',
        'final_individual_processes_and_event_IDs_exported':False})
    print('LIMITED_INDEPENDENT_EVIDENCE_AUDIT_PASSED',model_count,flush=True)


if __name__ == '__main__':
    main()
