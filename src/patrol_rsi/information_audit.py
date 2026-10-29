"""Read-only, factorized audit of every legal one-check Tour continuation.

Public forecasts and evaluator labels are deliberately separate interfaces.
No plant, learned-model training, or policy rollout is constructed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .causal_observation import task_signal, validate_public
from .inspection_timing import return_slack
from .public_stopping import StopConfig, short_tour_veto


SOURCE_NAMES = ('unavailable', 'history_assimilated', 'nominal', 'measured_bounded_trend',
                'remote_flow', 'qualified_flow', 'qualified_contact', 'qualified_sensor_error')
SOURCE_IDS = {name: i for i, name in enumerate(SOURCE_NAMES)}
KINDS = ('thermal', 'electrical', 'cooling', 'control')


def current_legal_geometry(bundle):
    """Include zero/missing-signal and alarmed targets; only hard masks apply."""
    validate_public(bundle)
    now = bundle['time_s']
    tasks = bundle['tasks']
    rows = []
    for agent in bundle['agents']:
        if agent['busy_target'] is not None:
            continue
        tour, cursor = agent['tour'], agent['cursor']
        base = tour[cursor]
        costs, edges = agent['first_costs_s'], agent['costs_s']
        if now + costs[str(base)] > 1800:
            continue
        last = {j: max(0., tasks[j]['last_check_s']) for j in tour}
        limits = {j: tasks[j]['deadline_s'] for j in tour}
        order = tour[cursor:] + tour[:cursor]
        arrivals, at, previous = {}, now, None
        for j in order:
            at += costs[str(j)] if previous is None else edges[str(previous)][str(j)]
            arrivals[j], previous = at, j
        for j in tour:
            finish = now + costs[str(j)]
            if j == base or agent['detours_used'] >= 2 or finish > 1800:
                continue
            if now - last[j] < 60 and (last[j] > 0 or tasks[j]['last_check_s'] >= 0):
                continue
            slack = return_slack(now, costs, last, tour, cursor, j, limits, edges)
            if slack is None:
                continue
            shift = max(0., costs[str(j)] + edges[str(j)][str(base)] - costs[str(base)])
            others = [k for k in order if k != j]
            first, later, full = [], [], []
            for k in others:
                first_at = arrivals[k]
                visits = []
                while first_at <= 1800:
                    visits.append((first_at, max(0., min(first_at + shift, 1800) - first_at)))
                    first_at += agent['period_s']
                first.append(visits[0][1] if visits else 0.)
                later.append(sum(e * math.exp(-a / 120) for a, e in visits[1:]))
                full.append(sum(e for _, e in visits))
            rows.append({'agent': agent['agent'], 'task': j, 'dispatch_time_s': now,
                         'finish_s': finish, 'tour_target_finish_s': arrivals[j],
                         'target_lead_s': max(0., min(arrivals[j], 1800) - finish),
                         'return_slack_s': slack, 'shift_s': shift,
                         'period_s': agent['period_s'], 'other_tasks': others,
                         'other_arrivals_s': [arrivals[k] for k in others],
                         'first_exposure_s': first, 'later_exposure_factor': later,
                         'full_exposure_s': full})
    return rows


def pack_geometry(rows):
    """One candidate per actual Tour dispatch, reused for every earlier root."""
    width = max((len(row['other_tasks']) for row in rows), default=0)
    arrays = {key: np.asarray([row[key] for row in rows], dtype=np.int32 if key in ('agent', 'task') else np.float64)
              for key in ('agent', 'task', 'dispatch_time_s', 'finish_s', 'tour_target_finish_s',
                          'target_lead_s', 'return_slack_s', 'shift_s', 'period_s')}
    for key in ('other_tasks', 'other_arrivals_s', 'first_exposure_s',
                'later_exposure_factor', 'full_exposure_s'):
        table = np.full((len(rows), width), -1 if key == 'other_tasks' else 0.,
                        dtype=np.int32 if key == 'other_tasks' else np.float64)
        for i, row in enumerate(rows):
            table[i, :len(row[key])] = row[key]
        arrays[key] = table
    arrays['other_mask'] = arrays['other_tasks'] >= 0
    return arrays


@dataclass
class PublicSignals:
    """Only fields from a validated public bundle, never evaluator arrays."""
    bundle: dict

    def __post_init__(self):
        validate_public(self.bundle)
        tasks = self.bundle['tasks']
        self.kind = np.asarray([KINDS.index(v['kind']) for v in tasks])
        self.weight = np.asarray([v['weight_scaled'] for v in tasks])
        self.pod = np.asarray([v['pod'] for v in tasks])
        self.alarm = np.asarray([v['already_alarm'] for v in tasks])
        self.velocity = np.asarray([v['velocity_c_s'] for v in tasks])
        self.channels = {}
        for key in ('history_assimilated', 'nominal', 'temperature_measurement', 'cvae_raw',
                    'flow', 'contact', 'sensor_error'):
            self.channels[key] = np.asarray([v[key]['value'] if v[key]['available'] else np.nan for v in tasks])
        self.age = np.asarray([v['temperature_measurement']['age_s'] if v['temperature_measurement']['available'] else 0. for v in tasks])
        self.flow_source = np.asarray([SOURCE_IDS.get(v['flow']['source'], 0) for v in tasks])

    def thermal_components(self, ids, delta):
        ids, delta = np.broadcast_arrays(ids, delta)
        return {'history_assimilated': self.channels['history_assimilated'][ids],
                'nominal': self.channels['nominal'][ids],
                'cvae_raw': self.channels['cvae_raw'][ids],
                'measured_bounded_trend': np.clip(self.channels['temperature_measurement'][ids] +
                    self.velocity[ids] * np.minimum(120., self.age[ids] + delta), 35., 110.)}

    def at(self, ids, delta=0., *, drop_history=False):
        ids, delta = np.broadcast_arrays(np.asarray(ids, dtype=int), delta)
        pieces = self.thermal_components(ids, delta)
        history = np.full(ids.shape, np.nan) if drop_history else pieces['history_assimilated']
        value = np.where(np.isfinite(history), history, -np.inf)
        source = np.where(np.isfinite(history), 1, 0)
        # Python max((value, source_name)) uses lexical source order on ties.
        for key in ('measured_bounded_trend', 'nominal'):
            temperature = pieces[key]
            take = np.isfinite(temperature) & (temperature >= value)
            value = np.where(take, temperature, value)
            source = np.where(take, SOURCE_IDS[key], source)
        signal = np.where(np.isfinite(value), np.maximum(value - 65., 0.) / 20., np.nan)
        kind = self.kind[ids]
        for k, channel, threshold, scale, src in (
                (1, 'contact', 60., 30., SOURCE_IDS['qualified_contact']),
                (2, 'flow', 33.6, -33.6, None),
                (3, 'sensor_error', 1., 1., SOURCE_IDS['qualified_sensor_error'])):
            raw = self.channels[channel][ids]
            measured = np.asarray(np.abs(raw) > 1, dtype=float) if k == 3 else np.maximum((raw - threshold) / scale, 0.)
            measured = np.where(np.isfinite(raw), measured, np.nan)
            signal = np.where(kind == k, measured, signal)
            current_source = self.flow_source[ids] if src is None else src
            source = np.where(kind == k, np.where(np.isfinite(raw), current_source, 0), source)
        return signal, source.astype(np.int16)

    def pod_support(self, *, drop_history=False):
        ids = np.arange(len(self.kind))
        signal, _ = self.at(ids, drop_history=drop_history)
        result = np.zeros(len(ids))
        for j in ids:
            other = signal[(self.pod == self.pod[j]) & (ids != j)]
            result[j] = min(max(np.nanmax(other) if np.isfinite(other).any() else 0., 0.), 1.) * .05
        return result


def score_candidates(public, geometry, config=StopConfig(), *, drop_history=False):
    """All remaining hard-legal candidates, including excluded/negative ones."""
    now = public.bundle['time_s']
    index = np.flatnonzero(geometry['dispatch_time_s'] >= now)
    target = geometry['task'][index]
    signal, source = public.at(target, geometry['finish_s'][index] - now, drop_history=drop_history)
    others = geometry['other_tasks'][index]
    mask = geometry['other_mask'][index]
    safe = np.maximum(others, 0)
    risk, _ = public.at(safe, np.minimum(geometry['other_arrivals_s'][index] +
                        geometry['shift_s'][index, None], 1800.) - now, drop_history=drop_history)
    corrected = np.where(np.isfinite(risk), risk, config.unknown_risk) + public.pod_support(drop_history=drop_history)[safe]
    weighted = public.weight[safe] * corrected * mask / 300
    first = (weighted * geometry['first_exposure_s'][index]).sum(axis=1)
    future = (weighted * geometry['later_exposure_factor'][index]).sum(axis=1) * math.exp(now / 120)
    benefit = public.weight[target] * signal * geometry['target_lead_s'][index] / 300
    execution = .05 * geometry['shift_s'][index] / 300
    net = benefit - config.collateral * (first + config.future_risk * future) - execution
    positive = np.isfinite(signal) & (signal > 0) & ~public.alarm[target]
    veto = (config.tour_soon_s > 0) & (geometry['tour_target_finish_s'][index] - geometry['dispatch_time_s'][index] <= config.tour_soon_s) & (benefit - first - future - execution <= 0.)
    return {'geometry_index': index, 'target_signal': signal, 'target_source': source,
            'target_benefit': benefit, 'first_round_collateral_cost': first,
            'full_future_risk_cost': future, 'execution_cost': execution,
            'net_score': net, 'positive_signal_eligible': positive,
            'tour_slack_veto': veto, 'admissible': positive & ~veto & (net > config.threshold),
            'current': geometry['dispatch_time_s'][index] == now}


def scalar_candidate(bundle, geometry, i, config=StopConfig(), *, drop_history=False):
    """Exact original Python operation order for winner/tie verification."""
    tasks, root = bundle['tasks'], bundle['time_s']
    def signal(j, delta=0.):
        row = tasks[j]
        if drop_history:
            row = dict(row, history_assimilated=dict(row['history_assimilated'], available=False, value=None))
        return task_signal(row, delta)
    j = int(geometry['task'][i])
    target_signal, target_source = signal(j, geometry['finish_s'][i] - root)
    first = future = exposure = 0.
    cost_rows, unknown = [], []
    shift = float(geometry['shift_s'][i])
    for k, arrival in zip(geometry['other_tasks'][i], geometry['other_arrivals_s'][i]):
        if k < 0:
            continue
        k = int(k)
        risk, src = signal(k, min(arrival + shift, 1800) - root)
        if risk is None:
            unknown.append(k)
        linked = max((signal(z)[0] or 0. for z in range(len(tasks)) if z != k and tasks[z]['pod'] == tasks[k]['pod']), default=0.)
        topology = .05 * min(linked, 1.)
        corrected = (risk if risk is not None else config.unknown_risk) + topology
        visits, at = [], float(arrival)
        while at <= 1800:
            visits.append((at, max(0., min(at + shift, 1800.) - at)))
            at += geometry['period_s'][i]
        initial = visits[0][1] if visits else 0.
        a = tasks[k]['weight_scaled'] * corrected * initial / 300
        b = tasks[k]['weight_scaled'] * corrected * sum(e * math.exp(-(t - root) / 120) for t, e in visits[1:]) / 300
        first += a
        future += b
        exposure += sum(e for _, e in visits)
        cost_rows.append({'task': k, 'delay_s': shift, 'full_delay_exposure_s': sum(e for _, e in visits),
                          'risk_signal': risk, 'missing': risk is None, 'risk_source': src,
                          'topology_correction': topology, 'first_round_cost': a, 'future_risk_cost': b})
    lead = float(geometry['target_lead_s'][i])
    benefit = tasks[j]['weight_scaled'] * target_signal * lead / 300 if target_signal is not None else None
    execution = .05 * shift / 300
    parts = {'target_benefit': benefit, 'first_round_collateral_cost': first,
             'full_future_risk_cost': future, 'execution_cost': execution}
    veto = short_tour_veto(parts, tour_eta_s=geometry['tour_target_finish_s'][i] - geometry['dispatch_time_s'][i],
                          soon_s=config.tour_soon_s) if benefit is not None else False
    return {'agent': int(geometry['agent'][i]), 'task': j,
            'dispatch_time_s': float(geometry['dispatch_time_s'][i]), 'finish_s': float(geometry['finish_s'][i]),
            'tour_target_finish_s': float(geometry['tour_target_finish_s'][i]), 'target_lead_s': lead,
            'target_signal': target_signal, 'target_source': target_source,
            'target_benefit': benefit, 'first_round_collateral_cost': first,
            'full_delay_exposure_s': exposure, 'future_risk_correction': config.future_risk * future,
            'execution_cost': execution, 'full_future_risk_cost': future,
            'net_score': benefit - config.collateral * (first + config.future_risk * future) - execution if benefit is not None else None,
            'tour_slack_veto': veto, 'unknown_collateral_tasks': unknown,
            'return_slack_s': float(geometry['return_slack_s'][i]), 'task_costs': cost_rows}


def audited_best(bundle, geometry, values, current, config):
    flag = values['admissible'] & (values['current'] if current else ~values['current'])
    possible = np.flatnonzero(flag)
    if not len(possible):
        return None, None
    maximum = np.max(values['net_score'][possible])
    close = possible[values['net_score'][possible] >= maximum - 1e-9]
    points = [(int(values['geometry_index'][i]), scalar_candidate(bundle, geometry, int(values['geometry_index'][i]), config)) for i in close]
    return max(points, key=lambda v: (v[1]['net_score'], -v[1]['dispatch_time_s'], -v[1]['agent'], -v[1]['task']))


def rescore_candidates(bundle, geometry, values, config):
    """Change only registered coefficient arithmetic at the same history."""
    index = values['geometry_index']
    benefit = values['target_benefit']
    first, future = values['first_round_collateral_cost'], values['full_future_risk_cost']
    net = benefit - config.collateral * (first + config.future_risk * future) - values['execution_cost']
    veto = (config.tour_soon_s > 0) & (geometry['tour_target_finish_s'][index] - geometry['dispatch_time_s'][index] <= config.tour_soon_s) & (benefit - first - future - values['execution_cost'] <= 0.)
    return dict(values, net_score=net, tour_slack_veto=veto,
                admissible=values['positive_signal_eligible'] & ~veto & (net > config.threshold))


def reference_signals(arrays):
    """Evaluator-only instantaneous professional signals, not public inputs."""
    thermal = np.maximum(arrays['rack_temperature_c'] - 65., 0.) / 20
    electrical = np.maximum(arrays['contact_temperature_c'] - 60., 0.) / 30
    cooling = np.maximum(1 - arrays['rack_flow_kg_s'].reshape(360, 16, 8).sum(2) / 33.6, 0.)
    control = (np.abs((arrays['public_sensor_temperature_c'] - arrays['rack_temperature_c']).reshape(360, 16, 8).mean(2)) > 1).astype(float)
    return np.concatenate((thermal, electrical, cooling, control), axis=1)


def effective_confirmability(arrays, seed):
    """Require both physical positivity and the original calibrated alarm.

    Thermal reference measurements have archived counter-based 0.1C noise.
    Randomness is indexed by old seed/task/time; no new process is sampled.
    """
    reference = reference_signals(arrays)
    confirmable = reference > 0
    for tick, task in np.argwhere(confirmable[:, :128]):
        noise = np.random.default_rng(3000000 + seed * 10000 + int(task) * 361 + int(tick)).normal(0, .1)
        confirmable[tick, task] = arrays['rack_temperature_c'][tick, task] + noise > 65.
    return confirmable


def analytic_continuations(tour_record, geometry, reference, *, measurement_positive=None):
    """Specify CHECK once then Tour; sample only the archived physical trace.

    Inspection changes observations but cannot change the prescribed remaining
    Tour actions or the plant in this frozen task. No alternate policy is run.
    """
    intervals = tour_record['anomaly_intervals']
    confirmable = reference > 0 if measurement_positive is None else measurement_positive
    count, total = len(geometry['task']), len(intervals)
    first = np.full((count, total), np.inf)
    events = tour_record['inspection_events']
    owner = {v['task']: v['agent'] for v in events}
    for column, window in enumerate(intervals):
        task = window['task']
        old_first = window['first_confirmation_s'] if window['detected'] else np.inf
        first[:, column] = old_first
        own = geometry['agent'] == owner[task]
        original_times = np.asarray([v['time_s'] for v in events if v['task'] == task])
        for shift in np.unique(geometry['shift_s'][own]):
            use = np.flatnonzero(own & (geometry['shift_s'] == shift))
            dispatch = geometry['dispatch_time_s'][use]
            future_times = original_times + shift
            in_mission = future_times <= 1800
            positive = confirmable[np.minimum(359, (future_times / 5).astype(int) - 1), task]
            valid = in_mission & positive & (future_times >= window['onset_s']) & (future_times <= window['end_s'])
            old_valid, new_valid = original_times[valid], future_times[valid]
            position = np.searchsorted(old_valid, dispatch, side='right')
            future = np.r_[new_valid, np.inf][position]
            past = np.where(old_first <= dispatch, old_first, np.inf)
            extra_time = geometry['finish_s'][use]
            extra_positive = confirmable[np.minimum(359, (extra_time / 5).astype(int) - 1), task]
            extra = np.where((geometry['task'][use] == task) & extra_positive &
                (extra_time >= window['onset_s']) & (extra_time <= window['end_s']), extra_time, np.inf)
            first[use, column] = np.minimum(np.minimum(past, future), extra)
    detected = np.isfinite(first)
    old_detected = np.asarray([v['detected'] for v in intervals], dtype=bool)
    missed_delta = (~detected).sum(axis=1) - int((~old_detected).sum())
    weights = np.asarray([v['weight'] for v in intervals])
    if weights.sum():
        onset = np.asarray([v['onset_s'] for v in intervals])
        ends = np.asarray([v['end_s'] for v in intervals])
        delay = np.where(detected, first - onset, ends - onset)
        old_delay = np.asarray([v['delay_or_censor_s'] for v in intervals])
        wait_delta = ((delay - old_delay) * weights).sum(axis=1) / weights.sum()
    else:
        wait_delta = np.full(count, np.nan)
    return {'missed_delta': missed_delta.astype(np.int16), 'weighted_wait_delta_s': wait_delta,
            'helpful': (missed_delta < 0) | ((missed_delta == 0) & (wait_delta < -1e-9)),
            'harmful': (missed_delta > 0) | ((missed_delta == 0) & (wait_delta > 1e-9)),
            'lost_confirmations': (old_detected & ~detected).sum(axis=1).astype(np.int16),
            'gained_confirmations': (~old_detected & detected).sum(axis=1).astype(np.int16),
            'first_confirmation_s': np.where(detected, first, -1).astype(np.int16)}
