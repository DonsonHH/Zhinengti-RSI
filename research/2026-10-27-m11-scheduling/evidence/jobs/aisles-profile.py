"""Synthetic rack-front aisle graph and explicit work/service certificates."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import heapq
import math

import numpy as np

from .facility_observation import InspectionTask

ROLES = ('thermal', 'electrical', 'cooling', 'control')
DEADLINES = {'thermal': 300., 'electrical': 600., 'cooling': 600., 'control': 600.}


class AisleGraph:
    def __init__(self, config, variant='open'):
        if variant not in ('open', 'restricted'):
            raise ValueError('Unknown aisle variant')
        self.config, self.variant = config, variant
        self.columns = math.ceil(math.sqrt(config.pods))
        self.adjacency = defaultdict(dict)
        self.tasks = []
        point = lambda x, y: (round(float(x), 6), round(float(y), 6))
        self.point = point

        def edge(a, b):
            cost = abs(a[0] - b[0]) + abs(a[1] - b[1])
            if cost <= 0:
                raise ValueError('No zero-length aisle edges')
            self.adjacency[a][b] = cost
            self.adjacency[b][a] = cost
        for pod in range(config.pods):
            x, y = (pod % self.columns) * 12., (pod // self.columns) * 12.
            for yy in (-1., 2.5):
                nodes = [point(x + dx, y + yy) for dx in (-1., 0., 1.2, 2.4, 3.6, 5.)]
                for a, b in zip(nodes, nodes[1:]):
                    edge(a, b)
            edge(point(x - 1, y - 1), point(x - 1, y + 2.5))
            bay = point(x + 5, y + 1)
            edge(point(x + 5, y - 1), bay)
            edge(bay, point(x + 5, y + 2.5))
            for local in range(8):
                target = point(x + (local % 4) * 1.2, y + (-1 if local < 4 else 2.5))
                rack = pod * 8 + local
                self.tasks.append(InspectionTask(f'thermal:{rack}', 'thermal', rack, 'thermal', 10., target))
            for role, duration in [('electrical', 20.), ('cooling', 20.), ('control', 15.)]:
                self.tasks.append(InspectionTask(f'{role}:{pod}', role, pod, role, duration, bay))
            if pod % self.columns:
                for yy in (-1., 2.5):
                    edge(point(x - 7, y + yy), point(x - 1, y + yy))
            if pod >= self.columns:
                for side in (-1., 5.):
                    # Close some internal north/south links while preserving
                    # the entire left boundary and all east/west connections.
                    if variant == 'restricted' and pod % self.columns and side == 5.:
                        continue
                    edge(point(x + side, y - 9.5), point(x + side, y - 1))
        self.tasks.sort(key=lambda t: (ROLES.index(t.required_role), t.index))

    @lru_cache(maxsize=200000)
    def path(self, source, target):
        if source not in self.adjacency or target not in self.adjacency:
            raise ValueError('Point outside passable aisle graph')
        heuristic = lambda a: abs(a[0] - target[0]) + abs(a[1] - target[1])
        queue = [(heuristic(source), 0., source)]
        distance, previous = {source: 0.}, {}
        while queue:
            _, cost, node = heapq.heappop(queue)
            if cost > distance[node] + 1e-10:
                continue
            if node == target:
                nodes = [target]
                while nodes[-1] != source:
                    nodes.append(previous[nodes[-1]])
                return float(cost), tuple(reversed(nodes))
            for neighbor, length in self.adjacency[node].items():
                candidate = cost + length
                if candidate < distance.get(neighbor, float('inf')) - 1e-10:
                    distance[neighbor], previous[neighbor] = candidate, node
                    heapq.heappush(queue, (candidate + heuristic(neighbor), candidate, neighbor))
        raise ValueError('Unreachable inspection approach')

    def description(self):
        edges = []
        for a, items in sorted(self.adjacency.items()):
            for b, length in sorted(items.items()):
                if a < b:
                    edges.append({'from': list(a), 'to': list(b), 'length_m': length})
        return {'variant': self.variant, 'nodes': [list(p) for p in sorted(self.adjacency)],
                'edges': edges, 'task_approaches': [{'id': t.task_id, 'point': list(t.position_m)} for t in self.tasks],
                'scope': 'Synthetic aisle centerlines outside rack interiors; no collision/queue simulation'}

    def spatial_rank(self, pod):
        row, column = divmod(pod, self.columns)
        width = min(self.columns, self.config.pods - row * self.columns)
        return row * self.columns + (column if row % 2 == 0 else width - 1 - column)


def fleet_counts(config, level):
    if level not in ('scarce', 'provisioned'):
        raise ValueError('Unknown resource level')
    factor = max(1, config.pods // 16)
    counts = (2, 1, 1, 1) if level == 'scarce' else (8, 2, 2, 2)
    return {role: count * factor for role, count in zip(ROLES, counts)}


def build_agents(graph, level):
    counts = fleet_counts(graph.config, level)
    result = []
    for role in ROLES:
        number = counts[role]
        zones = math.ceil(number / 2)
        for local_agent in range(number):
            zone, slot = divmod(local_agent, 2)
            peers = min(2, number - zone * 2)
            allowed = [j for j, t in enumerate(graph.tasks) if t.required_role == role and
                       min(zones - 1, graph.spatial_rank(t.index // 8 if role == 'thermal' else t.index) * zones // graph.config.pods) == zone]
            own = allowed[slot::peers]
            if not own:
                raise ValueError('Resource partition produced an empty inspector')
            # Nearest-neighbor closed tour is explicit and independently
            # checkable, not a claim of solving TSP.
            remaining, tour, node = set(own), [], graph.tasks[own[0]].position_m
            while remaining:
                selected = min(remaining, key=lambda j: (graph.path(node, graph.tasks[j].position_m)[0], j))
                tour.append(selected)
                remaining.remove(selected)
                node = graph.tasks[selected].position_m
            result.append({'id': len(result), 'role': role, 'zone': zone, 'allowed': allowed,
                           'tour': tour, 'cursor': 0, 'position': graph.tasks[tour[0]].position_m,
                           'task': None, 'remaining_travel_s': 0., 'remaining_service_s': 0.,
                           'travel_m': 0., 'service_s': 0., 'idle_s': 0., 'completed': 0})
    return result


def feasibility(graph, level):
    agents = build_agents(graph, level)
    counts = fleet_counts(graph.config, level)
    rows = {}
    for role in ROLES:
        tasks = [t for t in graph.tasks if t.required_role == role]
        demand = sum(t.service_s for t in tasks)
        cycles = []
        for agent in [a for a in agents if a['role'] == role]:
            tour = agent['tour']
            seconds = 0.
            distance = 0.
            for before, after in zip(tour[-1:] + tour[:-1], tour):
                leg = graph.path(graph.tasks[before].position_m, graph.tasks[after].position_m)[0]
                distance += leg
                seconds += math.ceil((leg + graph.tasks[after].service_s) / 5 - 1e-10) * 5
            cycles.append({'agent': agent['id'], 'tasks': len(tour), 'closed_distance_m': distance,
                           'closed_quantized_cycle_s': seconds})
        rows[role] = {'tasks': len(tasks), 'agents': counts[role], 'reference_revisit_s': DEADLINES[role],
                      'service_only_minimum_agents': math.ceil(demand / DEADLINES[role]),
                      'service_only_utilization': demand / (counts[role] * DEADLINES[role]),
                      'service_only_necessary_condition': demand <= counts[role] * DEADLINES[role],
                      'constructed_cycles': cycles,
                      'all_constructed_cycles_within_revisit': all(c['closed_quantized_cycle_s'] <= DEADLINES[role] for c in cycles)}
    return {'racks': graph.config.racks, 'variant': graph.variant, 'level': level,
            'total_agents': len(agents), 'roles': rows,
            'claim': 'Service is a necessary lower bound; listed closed tours certify only the constructed repetitive schedule, not all policies or initial backlog'}
