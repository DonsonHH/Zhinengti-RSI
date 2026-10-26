"""Public-state revisit skill and diagnostics; no simulator truth is read."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json

import numpy as np

from .environment import MOVES, PatrolEnv
from .experiment import distances, room


@dataclass(frozen=True)
class PatrolSkill:
    estimator_samples: int = 1
    estimator_refresh_interval: int = 1
    coordination_order: str = 'q_priority'
    revisit_trigger: int = 16

    def __post_init__(self):
        for key, allowed in [('estimator_samples', (1, 2, 4)),
                             ('estimator_refresh_interval', (1, 2, 4, 8)),
                             ('revisit_trigger', (0, 12, 16, 24, 32))]:
            value = getattr(self, key)
            if type(value) is not int or value not in allowed:
                raise ValueError(f'Invalid {key}')
        if self.coordination_order not in ('q_priority', 'cyclic'):
            raise ValueError('Invalid coordination order')
        if self.estimator_samples * ((96+self.estimator_refresh_interval-1)//self.estimator_refresh_interval) > 192:
            raise ValueError('Prior-sample budget exceeded')

    def proposed(self, patch: dict) -> 'PatrolSkill':
        if not isinstance(patch, dict) or set(patch)-set(asdict(self)):
            raise ValueError('Unknown/protected skill field')
        return PatrolSkill(**(asdict(self) | patch))

    @property
    def version_id(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


@lru_cache(maxsize=128)
def geometry(layout_seed: int):
    env = PatrolEnv(room(layout_seed), seed=0)
    goals = tuple(map(tuple, np.argwhere(env.scenario.targets)))
    costs = tuple(distances(env, g) for g in goals)
    return goals, costs


class RevisitSkill:
    """Latch an overdue target until observed; the DDQN handles other moves.

    Assignment uses raw ages, known geometry, and distance. It is an explicit
    human-authored tool available identically to every M2 optimization method.
    """
    def __init__(self, env: PatrolEnv, layout_seed: int):
        self.goals, self.costs = geometry(layout_seed)
        self.assigned: dict[int, int] = {}
        self.assignments = 0
        self.executor_steps = 0

    def scores(self, env: PatrolEnv, learned: np.ndarray, trigger: int) -> np.ndarray:
        result = learned.copy()
        if trigger == 0:
            return result
        # Any executor may have completed a latched target since the last step.
        for agent, goal in list(self.assigned.items()):
            if env.raw_age[self.goals[goal]] == 0:
                del self.assigned[agent]
        reserved = set(self.assigned.values())
        overdue = [j for j, g in enumerate(self.goals)
                   if env.raw_age[g] >= trigger and j not in reserved]
        overdue.sort(key=lambda j: (-int(env.raw_age[self.goals[j]]), j))
        available = set(range(len(env.positions)))-set(self.assigned)
        for goal in overdue:
            if not available:
                break
            agent = min(available, key=lambda a: (self.costs[goal][tuple(env.positions[a])], a))
            self.assigned[agent] = goal
            available.remove(agent)
            self.assignments += 1
        for agent, goal in self.assigned.items():
            # Higher scores give assigned urgent moves precedence in the same
            # existing conflict resolver. It still enforces all movement rules.
            result[agent] = -1e6
            for action, move in enumerate(MOVES):
                if env.valid_move(env.positions[agent], action):
                    result[agent, action] = 100-float(self.costs[goal][tuple(env.positions[agent]+move)])
        self.executor_steps += len(self.assigned)
        return result


def trajectory_diagnostics(env: PatrolEnv, age_history: list[np.ndarray],
                           position_history: list[np.ndarray], tool: str) -> dict:
    """Different diagnostic tools expose different public trace summaries."""
    goals = list(map(tuple, np.argwhere(env.scenario.targets)))
    history = np.asarray(age_history)
    common = {'tool': tool, 'target_count': len(goals)}
    if tool == 'coverage':
        initial = np.any([env.neighborhood(p) for p in position_history[0]], axis=0)
        common['never_observed_positions'] = [list(map(int, g)) for g in goals if not env.ever_measured[g]]
        common['first_observation_step'] = [
            {'position': list(map(int, g)), 'step': next((t for t, age in enumerate(history[:, j]) if age == 0 and (t > 0 or initial[g])), None)}
            for j, g in enumerate(goals)]
    elif tool == 'revisit':
        worst = sorted(enumerate(goals), key=lambda item: -int(history[:, item[0]].max()))[:4]
        common['longest_unobserved'] = [{'position': list(map(int, g)), 'peak_age': int(history[:, j].max()),
                                        'steps_over_24': int((history[:, j] > 24).sum())} for j, g in worst]
    elif tool == 'oscillation':
        points = np.asarray(position_history)
        common['two_step_returns_per_executor'] = (points[2:] == points[:-2]).all(-1).sum(0).tolist()
        common['distinct_positions_per_executor'] = [len(set(map(tuple, points[:, i]))) for i in range(points.shape[1])]
    else:
        raise ValueError('Unknown diagnostic tool')
    return common
