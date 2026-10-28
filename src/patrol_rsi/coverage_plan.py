"""Fixed-owner cyclic fallback certificates under deterministic route times."""
from __future__ import annotations

import math
import numpy as np


class InfeasibleCoverage(ValueError):
    pass


class ReturnPlan:
    def __init__(self, graph, tasks, tour, deadlines):
        self.graph, self.tasks = graph, tasks
        self.tour = tuple(int(j) for j in tour)
        if not self.tour or len(set(self.tour)) != len(self.tour):
            raise ValueError('Unique nonempty owned tour required')
        self.deadlines = deadlines
        self.cost = {}
        for before in self.tour:
            for after in self.tour:
                distance = graph.path(tasks[before].position_m, tasks[after].position_m)[0]
                self.cost[(before, after)] = math.ceil((distance + tasks[after].service_s) / 5 - 1e-10) * 5.
        self.period = sum(self.cost[(a, b)] for a, b in zip(self.tour[-1:] + self.tour[:-1], self.tour))
        if any(self.period > deadlines[j] + 1e-9 for j in self.tour):
            raise InfeasibleCoverage('Owned baseline cycle exceeds a protected revisit requirement')

    def travel_service(self, position, target):
        distance = self.graph.path(tuple(position), self.tasks[target].position_m)[0]
        return math.ceil((distance + self.tasks[target].service_s) / 5 - 1e-10) * 5.

    def check(self, clock, position, last, cursor, candidate=None):
        if not 0 <= cursor < len(self.tour):
            raise ValueError('Invalid protected cursor')
        if candidate is not None and candidate not in self.tour:
            raise ValueError('Candidate outside owned task contract')
        virtual = {j: max(0., float(last[j])) for j in self.tour}
        next_cursor, projected = cursor, float(clock)
        candidate_due, candidate_finish = None, None
        if candidate is not None:
            candidate_due = virtual[candidate] + self.deadlines[candidate]
            projected += self.travel_service(position, candidate)
            candidate_finish = projected
            if projected > candidate_due + 1e-9:
                return {'ok': False, 'reason': 'candidate_misses_its_deadline'}
            virtual[candidate] = projected
            position = self.tasks[candidate].position_m
            if candidate == self.tour[cursor]:
                next_cursor = (cursor + 1) % len(self.tour)
        order = self.tour[next_cursor:] + self.tour[:next_cursor]
        completions, due_times, minimum_slack = [], [], float('inf')
        if candidate is not None:
            minimum_slack = candidate_due - candidate_finish
        previous = candidate
        for j in order:
            duration = self.travel_service(position, j) if previous is None else self.cost[(previous, j)]
            projected += duration
            due = virtual[j] + self.deadlines[j]
            slack = due - projected
            if slack < -1e-9:
                return {'ok': False, 'reason': 'fallback_round_misses_deadline', 'failed_task': j}
            completions.append(float(projected))
            due_times.append(float(due))
            minimum_slack = min(minimum_slack, slack)
            virtual[j] = projected
            previous, position = j, self.tasks[j].position_m
        return {'ok': True, 'candidate': candidate, 'clock_s': float(clock),
                'cursor_before': cursor, 'cursor_after': next_cursor,
                'candidate_finish_s': candidate_finish,
                'candidate_due_s': float(candidate_due) if candidate_due is not None else None,
                'fallback_order': list(order), 'fallback_completion_s': completions,
                'fallback_due_s': due_times, 'repeating_cycle_s': self.period,
                'minimum_projected_slack_s': float(minimum_slack)}


def exact_task_gaps(tasks, events, horizon=1800.):
    by_task = {j: [] for j in range(len(tasks))}
    for event in events:
        by_task[event['task']].append(float(event['time_s']))
    maxima, intervals = [], []
    for j, task in enumerate(tasks):
        times = [0.] + sorted(by_task[j]) + [float(horizon)]
        gaps = np.diff(times)
        maxima.append(float(gaps.max()))
        intervals.append({'task': j, 'task_id': task.task_id, 'completion_times_s': times[1:-1],
                          'maximum_gap_including_initial_and_terminal_s': float(gaps.max())})
    return np.asarray(maxima), intervals
