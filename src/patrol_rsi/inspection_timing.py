"""Public-model check-vs-wait: no simulator trace, scenario or future readings.

Waiting follows the owned tour and preserves the one optional global action.
Forecasts are explicit surrogates; they do not claim calibrated anomaly value.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

from .public_dispatch import choose as tour_choose
from .inspection_value import choose as old_choose


@lru_cache(maxsize=4)
def geometry(graph):
    tasks = graph.tasks
    return tuple((t.kind, t.index, float(t.service_s)) for t in tasks)


def public_packet(env):
    """Whitelist current public observations and known schedule/map quantities."""
    obs = env.state.observer
    temperature = env.current_history if env.mode == 'history' else obs.predicted_temperature
    task_rows = []
    for j, (kind, index, service) in enumerate(geometry(env.graph)):
        rack = index if kind == 'thermal' else index * 8
        task_rows.append({'kind': kind, 'service_s': service, 'deadline_s': float(env.deadlines[j]),
            'weight_scaled': float(env.asset_weights[j] / 2), 'last_check_s': float(env.last_inspection[j]),
            'already_alarm': bool(env.measured_alert.get(j, False)),
            'temperature_c': float(temperature[rack]), 'last_temperature_c': float(obs.last_temperature[rack]),
            'temperature_age_s': float(env.clock - obs.temperature_stamp[rack]) if obs.temperature_stamp[rack] >= 0 else float(env.clock),
            'temperature_stamp_s': float(obs.temperature_stamp[rack]), 'velocity_c_s': float(env.state.velocity[rack]),
            'nominal_temperature_c': float(obs.predicted_temperature[rack]),
            'contact_c': float(obs.last_contact[index]) if kind == 'electrical' else None,
            'flow_kg_s': float(obs.last_flow[index]) if kind == 'cooling' else None,
            'sensor_verified': bool(obs.sensor_bias_verified[index]) if kind == 'control' else False})
    agents = []
    for i, a in enumerate(env.agents):
        plan = env.plans[i];tour = list(plan.tour)
        first = {str(j): plan.travel_service(a['position'], j) for j in tour}
        costs = {str(j): {str(k): plan.cost[(j, k)] for k in tour} for j in tour}
        remaining = a.get('remaining_travel_s', 0.) + a.get('remaining_service_s', 0.)
        agents.append({'agent': i, 'tour': tour, 'cursor': int(env.protected_cursor[i]),
            'busy_target': a['task'], 'busy_finish_s': float(env.clock + math.ceil(remaining / 5 - 1e-10) * 5) if a['task'] is not None else None,
            'detours_used': int(env.detours_used[i]), 'baseline_completed': int(env.baseline_completed[i]),
            'first_costs_s': first, 'costs_s': costs, 'period_s': plan.period})
    return {'time_s': float(env.clock), 'tasks': task_rows, 'agents': agents}


def strength(row, delta_s):
    kind = row['kind']
    if kind == 'thermal':
        linear = np.clip(row['last_temperature_c'] + row['velocity_c_s'] * min(120., row['temperature_age_s'] + delta_s), 35, 110)
        temperature = max(row['temperature_c'], float(linear))
        return max(temperature - 65, 0.) / 20
    if kind == 'electrical':
        return max(row['contact_c'] - 60, 0.) / 30
    if kind == 'cooling':
        return max(1 - row['flow_kg_s'] / 33.6, 0.)
    return float(row['sensor_verified'])


def return_slack(clock, costs, last, tour, cursor, target, deadlines, edge):
    """Known deterministic completion + whole round, with no event labels."""
    finish = clock + costs[str(target)]
    virtual = dict(last)
    slack = virtual[target] + deadlines[target] - finish
    if slack < -1e-9:
        return None
    virtual[target] = finish
    cursor = (cursor + 1) % len(tour) if target == tour[cursor] else cursor
    order = tour[cursor:] + tour[:cursor];t = finish;previous = target
    for j in order:
        t += edge[str(previous)][str(j)]
        gap = virtual[j] + deadlines[j] - t
        if gap < -1e-9:
            return None
        slack = min(slack, gap);virtual[j] = t;previous = j
    return float(slack)


def opportunities(packet, horizon_s):
    """Known tour schedule under WAIT, evaluated at all free 5s-grid states.

    Future professional measurements are not supplied or assimilated. Current
    evidence is extrapolated/held, so this is a limited public surrogate model.
    """
    if horizon_s not in (0, 120):
        raise ValueError('Frozen planning horizon')
    root = packet['time_s'];tasks = packet['tasks'];result = []
    for a in packet['agents']:
        tour = a['tour'];cursor = a['cursor'];edge = a['costs_s'];costs = a['first_costs_s']
        last = {j: max(0., tasks[j]['last_check_s']) for j in tour}
        deadlines = {j: tasks[j]['deadline_s'] for j in tour}
        t = a['busy_finish_s'] if a['busy_target'] is not None else root
        busy = a['busy_target'];completed = a['baseline_completed'];detours = a['detours_used']
        if busy is not None:
            # Before optional budget use, every pending assignment is tour work.
            if busy != tour[cursor]:
                raise ValueError('Wait projection requires a tour-continuation state')
            last[busy] = t;costs = edge[str(busy)];cursor = (cursor + 1) % len(tour);completed += 1
            if completed == len(tour):
                completed = detours = 0
        while t <= min(root + horizon_s, 1795.) + 1e-9:
            baseline = tour[cursor];base_finish = t + costs[str(baseline)]
            if base_finish > 1800:
                break
            base_slack = return_slack(t, costs, last, tour, cursor, baseline, deadlines, edge)
            if base_slack is None:
                raise ValueError('Projected protected tour infeasible')
            order = tour[cursor:] + tour[:cursor];arrivals = {};at = t;previous = None
            for j in order:
                at += costs[str(j)] if previous is None else edge[str(previous)][str(j)]
                arrivals[j] = min(1800., at);previous = j
            for j in tour:
                row = tasks[j];eta = costs[str(j)];finish = t + eta
                if j == baseline or detours >= 2 or finish > 1800 or row['already_alarm']:
                    continue
                # Unmeasured initial -1 becomes zero only for deadline arithmetic.
                if (last[j] > 0 or row['last_check_s'] >= 0) and t - last[j] < 60:
                    continue
                slack = return_slack(t, costs, last, tour, cursor, j, deadlines, edge)
                if slack is None:
                    continue
                signal = strength(row, finish - root)
                lead = max(arrivals[j] - finish, 0.)
                added = max(eta - costs[str(baseline)], 0.)
                pressure = max(base_slack - slack, 0.)
                value = row['weight_scaled'] * signal * lead / 300 - .05 * added / 300 - .05 * pressure / 300
                if value > 0:
                    result.append({'agent': a['agent'], 'task': j, 'dispatch_time_s': float(t), 'finish_s': float(finish),
                        'signal': float(signal), 'surrogate_value': float(value), 'baseline_target_finish_s': float(arrivals[j]),
                        'return_slack_s': slack, 'discounted_value': float(value * math.exp(-(t - root) / 300))})
            last[baseline] = base_finish;costs = edge[str(baseline)];cursor = (cursor + 1) % len(tour)
            completed += 1
            if completed == len(tour):
                completed = detours = 0
            t = base_finish
    return result


class TimingPolicy:
    def __init__(self, config, score=None):
        self.config = dict(config);self.score = score
        if self.config['method'] not in ('tour', 'cue', 'net', 'evidence', 'lookahead'):
            raise ValueError('Unknown timing policy')
        if self.config.get('not_before_s', 0) not in (0, 60, 180, 300, 'off'):
            raise ValueError('Fixed waiting threshold domain')

    def choose(self, groups, packet, used):
        choices, logs = tour_choose(groups, 'tour')
        now = packet['time_s'];method = self.config['method'];gate = self.config.get('not_before_s', 0)
        diagnostic = {'time_s': now, 'budget_available': not used, 'decision': 'continue_tour',
            'now_value': None, 'wait_value': None, 'future_best': None}
        if used or method == 'tour' or gate == 'off' or now < gate:
            diagnostic['decision'] = 'budget_used' if used else 'wait_gate' if method != 'tour' else 'continue_tour'
            return choices, logs, False, diagnostic
        if method in ('cue', 'net'):
            choices, logs, consumed = old_choose(groups, 'cue_rule' if method == 'cue' else 'net', used=False,
                threshold=self.config.get('threshold', 0.), score=self.score)
            diagnostic['decision'] = 'check' if consumed else 'wait_no_candidate'
            return choices, logs, consumed, diagnostic
        horizon = 120 if method == 'lookahead' else 0
        candidates = opportunities(packet, horizon)
        current = [r for r in candidates if r['dispatch_time_s'] == now]
        future = [r for r in candidates if r['dispatch_time_s'] > now]
        best = max(current, key=lambda r: (r['surrogate_value'], -r['agent'], -r['task']), default=None)
        later = max(future, key=lambda r: (r['discounted_value'], -r['dispatch_time_s'], -r['agent'], -r['task']), default=None)
        value = best['surrogate_value'] if best else 0.;wait = later['discounted_value'] if later else 0.
        diagnostic.update(now_value=value, wait_value=wait, future_best=later,
            projected_candidates=len(candidates), decision='wait_value' if later and wait >= value - 1e-9 else 'wait_no_candidate')
        if best is None or value <= max(0., wait) + 1e-9:
            return choices, logs, False, diagnostic
        g = next(g for g in groups if g['agent'] == best['agent'])
        r = next((r for r in g['candidates'] if r['task'] == best['task'] and r['is_optional']), None)
        if r is None:
            raise RuntimeError('Projected NOW action disagrees with current common legal mask')
        choices[g['agent']] = r['task']
        log = next(r for r in logs if r['agent'] == g['agent'])
        log.update(task=r['task'], tool='timing_' + method, rule_index=None, cue_strength=r['cue_strength'],
            known_measured_alarm=r['known_measured_alarm'], skill_value=value)
        diagnostic.update(decision='check', selected_now=best)
        return choices, logs, True, diagnostic
