"""Offline evaluator labels and sampling; never imported by deployable policy."""
from __future__ import annotations

import hashlib

import numpy as np

from .active_inspection import measured_anomaly
from .inspection_metrics import task_labels, anomaly_intervals
from .inspection_value import feature_rows


def public_key(g, r):
    return (float(g['time_s']), int(g['agent']), int(r['task']))


def public_hash(key):
    return hashlib.sha256(('%g:%d:%d' % key).encode()).hexdigest()


class Collector:
    def __init__(self):
        self.public, self.quality = {}, []
        self.confusion = dict(tp=0, fp=0, fn=0, tn=0)
        self.completion_segments = None

    def observe(self, env, groups):
        if self.completion_segments is None:
            labels, excess = task_labels(env)
            intervals = anomaly_intervals(labels, excess, env.tasks, env.asset_weights)
            self.completion_segments = {(r['task'], (t + 1) * 5.) for r in intervals
                for t in range(int(r['onset_s'] / 5), int(r['end_s'] / 5))}
        for g, r, x in feature_rows(groups):
            key = public_key(g, r)
            cued = bool(r['cue_strength'] >= .05 and not r['known_measured_alarm'])
            self.public[key] = {'intervention': dict(zip(('dispatch_time_s', 'agent', 'task'), key)),
                'x': x.tolist(), 'cue': cued, 'age': r['features'][0], 'hash': public_hash(key)}
            truth = ((r['task'], r['candidate_finish_s']) in self.completion_segments and
                measured_anomaly(env.tasks[r['task']], env.trace.measurement(env.tasks[r['task']], int(r['candidate_finish_s'] / 5) - 1)))
            self.confusion['tp' if truth and cued else 'fp' if cued else 'fn' if truth else 'tn'] += 1
        if int(env.clock) % 30 == 0:
            obs = env.state.observer
            age = np.where(obs.temperature_stamp >= 0, env.clock - obs.temperature_stamp, env.clock)
            self.quality.append({'time_s': env.clock, 'truth_c': env.trace.arrays['rack_temperature_c'][int(env.clock / 5) - 1].tolist(),
                'fresh': (obs.temperature_stamp == env.clock).tolist(), 'hold': obs.last_temperature.tolist(),
                'linear': np.clip(obs.last_temperature + env.state.velocity * age, 35, 110).tolist(),
                'nominal': obs.predicted_temperature.tolist(), 'history': env.current_history.tolist()})

    def sample(self, record, *, test=False):
        rows = sorted(self.public.values(), key=lambda r: r['hash'])
        chosen = {}
        groups = ([r for r in rows if r['cue']], [r for r in rows if not r['cue'] and r['age'] >= .5],
            [r for r in rows if not r['cue'] and r['age'] < .5])
        if test:
            groups = (groups[0], [r for r in rows if not r['cue']])
        for label, pool, count in zip(('cue', 'high_age', 'other') if not test else ('cue', 'other'), groups, (6, 3, 3) if not test else (2, 2)):
            for r in pool[:count]:
                key = tuple(r['intervention'].values())
                chosen[key] = dict(r, provenance=['natural_' + label])
        if not test:
            teachers = [r for r in record['evaluator_probe']['segments'] if not r['baseline_detected'] and r['first_witness'] is not None][:4]
            for segment in teachers:
                w = segment['first_witness']
                key = (w['dispatch_time_s'], w['agent'], segment['task'])
                if key not in chosen:
                    chosen[key] = dict(self.public[key], provenance=[])
                chosen[key]['provenance'].append('teacher_segment_' + str(segment['segment_index']))
        return list(chosen.values())


def labels(base, branch, target):
    before, after = base['anomaly_intervals'], branch['anomaly_intervals']
    if [(r['task'], r['onset_s'], r['end_s']) for r in before] != [(r['task'], r['onset_s'], r['end_s']) for r in after]:
        raise RuntimeError('Inspection must not change physical event segments')
    recovered = sum(not a['detected'] and b['detected'] for a, b in zip(before, after))
    lost = sum(a['detected'] and not b['detected'] for a, b in zip(before, after))
    target_recovered = sum(not a['detected'] and b['detected'] and a['task'] == target for a, b in zip(before, after))
    target_wait_gain = sum(a['weight'] * (a['delay_or_censor_s'] - b['delay_or_censor_s']) for a, b in zip(before, after) if a['task'] == target)
    target_weight = sum(a['weight'] for a in before if a['task'] == target)
    # The target ablation counts only positive benefits, deliberately excluding collateral losses.
    target_value = target_recovered + max(target_wait_gain / target_weight, 0) / 3000 if target_weight else 0.
    a, b = base['metrics'], branch['metrics']
    wait_gain = (a['consequence_weighted_censored_delay_s'] - b['consequence_weighted_censored_delay_s']) if before else 0.
    travel = b['travel_m'] - a['travel_m']
    net_value = a['missed_segments'] - b['missed_segments'] + wait_gain / 3000 - travel / 100000
    return {'net': net_value, 'target': target_value, 'recovered_segments': recovered, 'collateral_lost_segments': lost,
        'target_recovered_segments': target_recovered, 'missed_delta': b['missed_segments'] - a['missed_segments'],
        'weighted_wait_gain_s': wait_gain, 'travel_delta_m': travel, 'service_delta_s': b['service_s'] - a['service_s']}
