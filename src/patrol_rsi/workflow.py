"""A bounded executable patrol DSL. It reads public observations only."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .environment import MOVES, PatrolEnv
from .revisit import geometry

CONDITIONS = ('unseen', 'due_soon', 'overdue', 'stalled', 'always')
PRIORITIES = ('least_slack', 'age_ratio', 'risk_age', 'oldest')
ASSIGNMENTS = ('nearest', 'global_min')
COMMITS = ('until_observed', 'one_step')


@dataclass(frozen=True)
class Workflow:
    """Immutable JSON snapshot of a short, ordered branch/dispatch program."""
    serialized: str

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {'steps'}:
            raise ValueError('Workflow must contain steps only')
        steps = value['steps']
        if not isinstance(steps, list) or not 2 <= len(steps) <= 5:
            raise ValueError('Workflow needs 1..3 dispatch rules, optional release, and final fallback')
        dispatches = 0; release = 0
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                raise ValueError('Each step must be an object')
            op = step.get('op')
            if op == 'dispatch':
                if set(step) != {'op', 'when', 'priority', 'assignment', 'commit'}:
                    raise ValueError('dispatch requires when, priority, assignment, commit only')
                for name, domain in [('when', CONDITIONS), ('priority', PRIORITIES),
                                     ('assignment', ASSIGNMENTS), ('commit', COMMITS)]:
                    if step[name] not in domain:
                        raise ValueError(f'Invalid workflow {name}')
                dispatches += 1
            elif op == 'release_stalled':
                if set(step) != {'op'} or i != 0 or release:
                    raise ValueError('Optional release_stalled must occur once at the start')
                release += 1
            elif op == 'fallback':
                if set(step) != {'op', 'policy'} or step['policy'] != 'ddqn' or i != len(steps)-1:
                    raise ValueError('Final step must be fallback with policy ddqn')
            else:
                raise ValueError('Unknown DSL operation; code and other tools are not executable')
        if not 1 <= dispatches <= 3 or steps[-1].get('op') != 'fallback':
            raise ValueError('Need 1..3 dispatch rules and final DDQN fallback')
        return cls(json.dumps(value, sort_keys=True, separators=(',', ':')))

    def to_dict(self):
        return json.loads(self.serialized)

    @property
    def version_id(self):
        return hashlib.sha256(self.serialized.encode()).hexdigest()[:16]


def dispatch(when, priority, assignment='nearest', commit='until_observed'):
    return {'op': 'dispatch', 'when': when, 'priority': priority, 'assignment': assignment, 'commit': commit}


def human_workflow():
    return Workflow.from_dict({'steps': [
        {'op': 'release_stalled'},
        dispatch('due_soon', 'least_slack', 'global_min'),
        dispatch('unseen', 'oldest'),
        {'op': 'fallback', 'policy': 'ddqn'}]})


def random_workflow(rng):
    steps = [{'op': 'release_stalled'}] if rng.random() < .5 else []
    for _ in range(int(rng.integers(1, 4))):
        steps.append(dispatch(str(rng.choice(CONDITIONS)), str(rng.choice(PRIORITIES)),
                              str(rng.choice(ASSIGNMENTS)), str(rng.choice(COMMITS))))
    steps.append({'op': 'fallback', 'policy': 'ddqn'})
    return Workflow.from_dict({'steps': steps})


class WorkflowExecutor:
    """Fixed interpreter: branches, assignment, latched navigation, DDQN return.

    DSL programs can change rule ordering, target eligibility, ranking, joint
    assignment, commitment and stalled recovery. No Python/eval/exec is used.
    """
    def __init__(self, env: PatrolEnv, layout_seed: int, deadlines: np.ndarray, program: Workflow):
        self.goals, self.costs = geometry(layout_seed)
        self.deadlines = np.asarray(deadlines, dtype=float)
        if self.deadlines.shape != (len(self.goals),) or np.any(self.deadlines <= 0):
            raise ValueError('One positive public deadline per inspection point required')
        self.program = program
        self.assigned = {}  # executor -> (goal index, commit type)
        self.history = [env.positions.copy()]
        self.last_step = env.t
        self.interventions = 0; self.assignments = 0; self.releases = 0
        self.executed_rules = [0]*len(program.to_dict()['steps'])

    def stalled(self, env):
        if self.last_step != env.t:
            self.history.append(env.positions.copy())
            self.history = self.history[-5:]
            self.last_step = env.t
        if len(self.history) < 5:
            return set()
        h = np.asarray(self.history)
        return {i for i in range(len(env.positions))
                if np.array_equal(h[-1,i],h[-3,i]) and np.array_equal(h[-3,i],h[-5,i])}

    def scores(self, env: PatrolEnv, learned: np.ndarray):
        scores = learned.copy(); stalled = self.stalled(env)
        for agent, (goal, commit) in list(self.assigned.items()):
            if env.raw_age[self.goals[goal]] == 0 or commit == 'one_step':
                del self.assigned[agent]
        ages = np.array([env.raw_age[g] for g in self.goals])
        seen = np.array([env.ever_measured[g] for g in self.goals])
        risk = np.array([env.estimate[g] for g in self.goals])
        steps = self.program.to_dict()['steps']
        for index, step in enumerate(steps):
            if step['op'] == 'release_stalled':
                for agent in stalled:
                    if agent in self.assigned:
                        del self.assigned[agent]; self.releases += 1; self.executed_rules[index] += 1
                continue
            if step['op'] == 'fallback':
                self.executed_rules[index] += len(env.positions)-len(self.assigned)
                break
            free = set(range(len(env.positions)))-set(self.assigned)
            reserved = {goal for goal, _ in self.assigned.values()}
            candidates = []
            for j, goal in enumerate(self.goals):
                if j in reserved or not free:
                    continue
                nearest = min(max(0, self.costs[j][tuple(env.positions[a])]-1) for a in free)
                when = step['when']
                eligible = (when == 'always' or when == 'unseen' and not seen[j]
                            or when == 'due_soon' and ages[j]+nearest >= .75*self.deadlines[j]
                            or when == 'overdue' and ages[j] >= self.deadlines[j]
                            or when == 'stalled' and bool(free & stalled))
                if eligible:
                    priority = {'least_slack': -(self.deadlines[j]-ages[j]-nearest),
                                'age_ratio': ages[j]/self.deadlines[j],
                                'risk_age': (risk[j]+.1)*ages[j]/self.deadlines[j],
                                'oldest': float(ages[j])}[step['priority']]
                    candidates.append((j, priority))
            # Different assignment tools operate on the same known geometry.
            while candidates and free:
                if step['assignment'] == 'nearest':
                    goal, _ = max(candidates, key=lambda item: (item[1], -item[0]))
                    agent = min(free, key=lambda a: (self.costs[goal][tuple(env.positions[a])], a))
                else:
                    agent, goal = min(((a,j) for a in free for j,_ in candidates),
                        key=lambda pair: (self.costs[pair[1]][tuple(env.positions[pair[0]])]
                                          -2*next(v for j,v in candidates if j==pair[1]), pair))
                self.assigned[agent] = (goal, step['commit']); free.remove(agent)
                candidates = [(j,v) for j,v in candidates if j != goal]
                self.assignments += 1; self.executed_rules[index] += 1
        for agent, (goal, _) in self.assigned.items():
            scores[agent] = -1e6
            for action, move in enumerate(MOVES):
                if env.valid_move(env.positions[agent], action):
                    scores[agent,action] = 100-float(self.costs[goal][tuple(env.positions[agent]+move)])
        self.interventions += len(self.assigned)
        return scores
