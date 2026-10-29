"""B2 frozen information controls, blind scheduling and conservative waiting."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math

from .public_dispatch import choose as tour_choose
from .inspection_timing import TimingPolicy, opportunities, strength

OBSERVATION_FIELDS = ('temperature_c', 'last_temperature_c', 'temperature_age_s', 'temperature_stamp_s',
    'velocity_c_s', 'nominal_temperature_c', 'contact_c', 'flow_kg_s', 'sensor_verified')


def digest(text):
    return hashlib.sha256(text.encode('ascii')).hexdigest()


def blind_time(seed):
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError('Three predeclared public blind seeds')
    return 5 * (1 + int(digest('b2-blind-time:' + str(seed))[:16], 16) % 300)


def observation_control(packet, mode):
    if mode not in ('original', 'permuted', 'masked'):
        raise ValueError('Unknown information control')
    if mode == 'original':
        return packet
    groups = defaultdict(list)
    for j, row in enumerate(packet['tasks']):
        groups[row['kind']].append(j)
    changed = dict(packet, tasks=[dict(r) for r in packet['tasks']])
    if mode == 'masked':
        for r in changed['tasks']:
            r.update(temperature_c=50., last_temperature_c=50., velocity_c_s=0., nominal_temperature_c=50.,
                contact_c=38. if r['kind'] == 'electrical' else None,
                flow_kg_s=48. if r['kind'] == 'cooling' else None, sensor_verified=False)
    else:
        for kind, ids in groups.items():
            ordered = sorted(ids, key=lambda j: digest('b2-permute:' + kind + ':' + str(j)))
            for target, source in zip(ordered, ordered[1:] + ordered[:1]):
                for field in OBSERVATION_FIELDS:
                    changed['tasks'][target][field] = packet['tasks'][source][field]
    return changed


def projected_values(packet, *, conservative=False):
    rows = opportunities(packet, 120)
    if not conservative:
        return rows
    result = []
    for r in rows:
        row = dict(r)
        if r['dispatch_time_s'] > packet['time_s']:
            task = packet['tasks'][r['task']]
            cap = strength(task, r['finish_s'] - r['dispatch_time_s'])
            reduced = min(r['signal'], cap)
            lead = max(r['baseline_target_finish_s'] - r['finish_s'], 0.)
            row['surrogate_value'] -= task['weight_scaled'] * (r['signal'] - reduced) * lead / 300
            row['signal'] = reduced
            row['discounted_value'] = row['surrogate_value'] * math.exp(-(r['dispatch_time_s'] - packet['time_s']) / 300)
        if row['surrogate_value'] > 0:
            result.append(row)
    return result


class RobustTimingPolicy:
    def __init__(self, config, score=None):
        if set(config) - {'method', 'signal_mode', 'conservative', 'blind_seed'}:
            raise ValueError('Unrecognized policy metadata')
        if config.get('method') not in ('tour', 'cue', 'evidence', 'lookahead', 'blind'):
            raise ValueError('Unknown B2 method')
        self.config = dict(config)
        if config.get('signal_mode', 'original') not in ('original', 'permuted', 'masked'):
            raise ValueError('Unknown signal mode')
        if type(config.get('conservative', False)) is not bool:
            raise ValueError('Conservative flag must be boolean')
        if config['method'] == 'blind':
            blind_time(config.get('blind_seed'))
        elif 'blind_seed' in config:
            raise ValueError('Blind seed only belongs to blind comparator')

    def needs_observations(self, used):
        return not used and self.config['method'] in ('evidence', 'lookahead')

    def choose(self, groups, packet, used):
        method = self.config['method'];now = packet['time_s']
        if method in ('tour', 'cue', 'evidence'):
            return TimingPolicy({'method': method}).choose(groups, packet, used)
        choices, logs = tour_choose(groups, 'tour')
        info = {'time_s': now, 'budget_available': not used, 'decision': 'budget_used' if used else 'wait_no_candidate',
            'now_value': None, 'wait_value': None, 'future_best': None}
        if used:
            return choices, logs, False, info
        if method == 'blind':
            seed = self.config['blind_seed'];gate = blind_time(seed)
            info.update(blind_seed=seed, blind_trigger_s=gate)
            if now < gate:
                info['decision'] = 'wait_gate'
                return choices, logs, False, info
            optional = [(g, r) for g in groups for r in g['candidates'] if r['is_optional']]
            if not optional:
                return choices, logs, False, info
            group, candidate = min(optional, key=lambda pair: digest(
                'b2-blind-action:%d:%g:%d:%d' % (seed, now, pair[0]['agent'], pair[1]['task'])))
            value = None
            info.update(decision='check', selected_blind={'agent': group['agent'], 'task': candidate['task']})
        else:
            controlled = observation_control(packet, self.config.get('signal_mode', 'original'))
            rows = projected_values(controlled, conservative=self.config.get('conservative', False))
            current = [r for r in rows if r['dispatch_time_s'] == now]
            future = [r for r in rows if r['dispatch_time_s'] > now]
            best = max(current, key=lambda r: (r['surrogate_value'], -r['agent'], -r['task']), default=None)
            later = max(future, key=lambda r: (r['discounted_value'], -r['dispatch_time_s'], -r['agent'], -r['task']), default=None)
            value = best['surrogate_value'] if best else 0.;wait = later['discounted_value'] if later else 0.
            info.update(now_value=value, wait_value=wait, future_best=later, projected_candidates=len(rows),
                decision='wait_value' if later and wait >= value - 1e-9 else 'wait_no_candidate')
            if best is None or value <= max(0., wait) + 1e-9:
                return choices, logs, False, info
            group = next(g for g in groups if g['agent'] == best['agent'])
            candidate = next((r for r in group['candidates'] if r['task'] == best['task'] and r['is_optional']), None)
            if candidate is None:
                raise RuntimeError('Signal control cannot change current action legality')
            info.update(decision='check', selected_now=best)
        choices[group['agent']] = candidate['task']
        log = next(r for r in logs if r['agent'] == group['agent'])
        log.update(task=candidate['task'], tool='b2_' + method, rule_index=None, cue_strength=candidate['cue_strength'],
            known_measured_alarm=candidate['known_measured_alarm'], skill_value=value)
        return choices, logs, True, info
