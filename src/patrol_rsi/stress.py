"""M7 public deployment environments and small-team planning controls.

M6 source is unchanged. Its 14-feature dispatcher and low-level controller are
reused; M7 adds a declared layout family, deadline profile and paired noise.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import permutations

import numpy as np

from .dispatch import DispatchEnv, heuristic_choices
from .environment import PatrolEnv, Scenario
from .experiment import distances, room


def stress_room(layout_seed, family, noise=0.0):
    if family == 'rack':
        return room(layout_seed, noise)
    if family != 'baffle':
        raise ValueError('Unknown layout family')
    rng = np.random.default_rng(layout_seed)
    nav = np.ones((24, 24), dtype=bool)
    nav[[0, -1], :] = False
    nav[:, [0, -1]] = False
    for col, base in ((8, 5), (15, 18)):
        nav[1:23, col] = False
        gap = base + int(rng.integers(-1, 2))
        nav[gap:gap + 2, col] = True
    targets = np.zeros_like(nav)
    for row, col in ((5, 4), (17, 4), (4, 11), (11, 11),
                     (19, 11), (5, 19), (12, 19), (19, 19)):
        rr, cc = row + int(rng.integers(-1, 2)), col + int(rng.integers(-1, 2))
        targets[rr, cc] = True
    starts = np.asarray(((2, 2), (2, 21), (21, 2), (21, 21)))
    return Scenario(f'm7-{family}-{layout_seed}', nav, targets, starts, noise=noise)


class PairedNoiseEnv(PatrolEnv):
    """Noise keyed by field seed, absolute step and cell, not visit history."""
    def _observe(self, mask):
        if self.scenario.noise:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.t, 19539]))
            noise = rng.normal(0, self.scenario.noise, self._truth.shape)
            values = self._truth[mask] + noise[mask]
        else:
            values = self._truth[mask]
        observation = np.clip(values, 0, 1)
        self.values[mask] = observation
        self.measured[mask] = True
        self.ever_measured[mask] = True
        self.estimate[mask] = observation


@lru_cache(maxsize=128)
def stress_geometry(layout_seed, family):
    env = PatrolEnv(stress_room(layout_seed, family), seed=0)
    targets = tuple(map(tuple, np.argwhere(env.scenario.targets)))
    costs = tuple(distances(env, target) for target in targets)
    if len(targets) != 8 or any(not np.isfinite(cost[env.scenario.navigation]).all() for cost in costs):
        raise ValueError('M7 requires eight connected targets')
    for cost in costs:
        cost.setflags(write=False)
    return targets, costs


def stress_deadlines(layout_seed, profile):
    values = {'balanced': (16, 24, 32), 'tight': (12, 16, 24), 'pressure': (8, 12, 16)}
    if profile not in values:
        raise ValueError('Unknown deadline profile')
    deadlines = np.resize(np.asarray(values[profile], dtype=int), 8).copy()
    np.random.default_rng(layout_seed + 447).shuffle(deadlines)
    return deadlines


class StressDispatchEnv(DispatchEnv):
    def __init__(self, case):
        self.case = case.copy()
        self.env = PairedNoiseEnv(stress_room(case['layout_seed'], case['family'], case['noise']), case['field_seed'])
        self.targets, self.costs = stress_geometry(case['layout_seed'], case['family'])
        self.deadlines = stress_deadlines(case['layout_seed'], case['profile'])
        self.assigned = np.full(4, -1, dtype=int)
        self.assignments = 0
        self.duplicate_assignments = 0

    def initial_access_bounds(self):
        # Optimistic: ignores executor congestion and shared-target contention.
        earliest = np.asarray([min(max(0, int(cost[tuple(p)]) - self.env.scenario.radius)
                                   for p in self.env.scenario.starts) for cost in self.costs])
        unavoidable = sum(max(0, min(96, int(arrival) - 1) - int(due))
                          for arrival, due in zip(earliest, self.deadlines)) / (96 * 8)
        return {'earliest_possible_first_observation_steps': earliest.tolist(),
                'startup_violation_lower_bound': unavoidable,
                'scope': 'Optimistic startup-only bound; not a recurrent feasibility proof'}


@lru_cache(maxsize=5)
def assignment_tuples(count):
    values = np.asarray(list(permutations(range(8), count)), dtype=int).reshape(-1, count) if count else np.zeros((1, 0), dtype=int)
    values.setflags(write=False)
    return values


def best_joint_assignment(utilities, state):
    """Exact maximum for this additive assignment proxy, not for patrol."""
    actors = np.asarray([i for i in state['order'] if state['free'][i]], dtype=int)
    choices = np.full(4, -1, dtype=int)
    if not len(actors):
        return choices, {'valid_combinations': 1, 'proxy_value': 0.0}
    options = assignment_tuples(len(actors))
    allowed = state['allowed'][actors] & ~state['reserved']
    legal = allowed[np.arange(len(actors))[None, :], options].all(axis=1)
    candidates = options[legal]
    if not len(candidates):
        raise ValueError('No valid joint assignment')
    score = utilities[actors][np.arange(len(actors))[None, :], candidates].sum(axis=1)
    best = int(np.argmax(score))
    choices[actors] = candidates[best]
    return choices, {'valid_combinations': len(candidates), 'proxy_value': float(score[best])}


def arrival_benefit(state, horizon=24):
    features = state['features'].astype(np.float64)
    deadline = features[:, :, 1] * 32
    age = features[:, :, 0] * deadline
    travel = np.rint(features[:, :, 4] * 22)
    importance = features[:, :, 2]
    benefit = np.zeros((4, 8), dtype=np.float64)

    def cost(value):
        return ((value > deadline) + .1 * np.minimum(value / deadline, 4)
                + .25 * (.1 + importance) / .6 * np.minimum(value * .05, 1)) / 8

    for tau in range(1, horizon + 1):
        baseline = age + tau
        refreshed = np.where(tau >= travel, tau - travel, baseline)
        benefit += .99 ** tau * (cost(baseline) - cost(refreshed))
    benefit += features[:, :, 3] * (travel <= horizon) * .99 ** travel / 8
    return benefit


def pairings(points):
    points = tuple(points)
    if not points:
        yield ()
        return
    first = points[0]
    for slot in range(1, len(points)):
        second = points[slot]
        for rest in pairings(points[1:slot] + points[slot + 1:]):
            yield ((first, second),) + rest


class PairCyclePlanner:
    """Exhaustive pair partition with optimistic distance-cycle proxy."""
    def __init__(self, wrapper):
        points = wrapper.targets
        due = wrapper.deadlines
        best = None
        for pairs in pairings(range(8)):
            objective = 0.0
            for left, right in pairs:
                leg = max(1, int(wrapper.costs[right][points[left]]) - 2 * wrapper.env.scenario.radius)
                cycle = 2 * leg
                objective += (cycle / due[left]) ** 2 + (cycle / due[right]) ** 2
            candidate = (objective, pairs)
            if best is None or candidate < best:
                best = candidate
        _, pairs = best
        matches = []
        for perm in permutations(range(4)):
            distance = sum(min(wrapper.costs[goal][tuple(wrapper.env.positions[actor])]
                               for goal in pairs[perm[actor]]) for actor in range(4))
            matches.append((distance, perm))
        _, perm = min(matches)
        self.pairs = tuple(tuple(pairs[perm[actor]]) for actor in range(4))
        self.proxy_objective = float(best[0])
        self.cross_pair_assignments = 0
        self.assignments = 0

    def choices(self, state):
        selected = np.full(4, -1, dtype=int)
        reserved = state['reserved'].copy()
        for actor in state['order']:
            if not state['free'][actor]:
                continue
            legal = state['allowed'][actor] & ~reserved
            own = legal & np.isin(np.arange(8), self.pairs[actor])
            features = state['features'][actor]
            score = ((features[:, 0] + .05) * (.1 + features[:, 2]) + features[:, 3]) / (1 + features[:, 4] * 22)
            target = int(np.argmax(np.where(own if own.any() else legal, score, -1e9)))
            if not legal[target]:
                raise ValueError('No legal cycle target')
            selected[actor] = target
            reserved[target] = True
            self.assignments += 1
            self.cross_pair_assignments += int(target not in self.pairs[actor])
        return selected


def planner_choices(state, method, pair_planner=None):
    if method == 'matching':
        return best_joint_assignment(arrival_benefit(state), state)[0]
    if method == 'pair_cycle':
        if pair_planner is None:
            raise ValueError('Pair planner required')
        return pair_planner.choices(state)
    return heuristic_choices(state, method)
