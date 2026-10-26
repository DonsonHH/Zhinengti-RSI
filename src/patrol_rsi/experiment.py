"""Split manifests, known indoor geometry, and fixed-controller evaluation."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .environment import MOVES, PatrolEnv, Scenario
from .versions import AgentVersion


@dataclass(frozen=True)
class Case:
    split: str
    layout_seed: int
    field_seed: int


def split_manifest() -> dict:
    manifest = {"validation": [asdict(Case("validation", 2000+i, 300000+i)) for i in range(8)],
                "test": [asdict(Case("test", 3000+i, 400000+i)) for i in range(12)],
                "development": [asdict(Case("development", 4000+i, 500000+i)) for i in range(6)],
                "train_layout_seeds": list(range(1000, 1064)),
                "train_field_rule": "1000000 + training_seed*10000 + episode",
                "vae_train_field_seeds": list(range(10000, 11024)),
                "vae_validation_field_seeds": list(range(20000, 20064)),
                "test_access_rule": "Only frozen validation-selected checkpoints; never provide test logs to improver"}
    return manifest


def room(layout_seed: int, noise: float = 0) -> Scenario:
    rng = np.random.default_rng(layout_seed)
    size = 24
    nav = np.ones((size, size), dtype=bool)
    nav[[0, -1], :] = False
    nav[:, [0, -1]] = False
    targets = np.zeros_like(nav)
    for base_row in (6, 17):
        for base_col in (5, 9, 14, 18):
            row = base_row + int(rng.integers(-1, 2))
            col = base_col + int(rng.integers(-1, 2))
            nav[row:row+2, col:col+2] = False
            targets[row-1, col] = True
    starts = np.array([(2, 2), (2, 21), (21, 2), (21, 21)])
    return Scenario(f"room-{layout_seed}", nav, targets, starts, noise=noise)


def distances(env: PatrolEnv, goal: tuple[int, int]) -> np.ndarray:
    costs = np.full(env.values.shape, np.inf)
    costs[goal] = 0
    queue = deque([np.array(goal)])
    while queue:
        p = queue.popleft()
        for a, move in enumerate(MOVES):
            nxt = p + move
            if env.valid_move(p, a) and not np.isfinite(costs[tuple(nxt)]):
                costs[tuple(nxt)] = costs[tuple(p)] + 1
                queue.append(nxt)
    return costs


class RouteGreedy:
    """Known-map multistep reference; only observed/estimated importance is used."""
    def __init__(self, env: PatrolEnv):
        self.goals = list(map(tuple, np.argwhere(env.scenario.targets)))
        self.costs = [distances(env, goal) for goal in self.goals]

    def scores(self, env: PatrolEnv) -> np.ndarray:
        result = np.full((len(env.positions), 8), -1e6)
        reserved = set()
        for i in np.roll(np.arange(len(env.positions)), -(env.t % len(env.positions))):
            position = tuple(env.positions[i])
            utilities = [(env.idleness[g] + env.scenario.forgetting) *
                         (env.estimate[g] + env.scenario.background_importance) /
                         (1 + self.costs[j][position]) if j not in reserved else -1
                         for j, g in enumerate(self.goals)]
            selected = int(np.argmax(utilities))
            reserved.add(selected)
            cost = self.costs[selected]
            for a, move in enumerate(MOVES):
                if env.valid_move(env.positions[i], a):
                    result[i, a] = -float(cost[tuple(env.positions[i]+move)])
        return result


def metrics_mean(rows: list[dict]) -> dict:
    keys = ("mean_normalized_weighted_idleness", "target_coverage", "max_unobserved_steps",
            "reconstruction_mse", "total_movement_distance", "estimator_calls", "prior_samples")
    return {key: float(np.mean([r[key] for r in rows])) for key in keys}


def evaluate_case(estimator, policy, case: dict, device, method: str = "learned",
                  version: AgentVersion | None = None, horizon: int = 96,
                  record: bool = False) -> tuple[dict, list[dict]]:
    import torch
    from .environment import greedy_scores
    version = version or AgentVersion()
    env = PatrolEnv(room(case["layout_seed"]), seed=case["field_seed"])
    route = RouteGreedy(env) if method == "route_greedy" else None
    rng = np.random.default_rng(case["field_seed"] + 900000)
    # Same per-case sampling sequence for paired comparisons.
    torch.manual_seed(case["field_seed"])
    estimator.eval()
    policy.eval()
    calls = 0
    scores_history = []
    for t in range(horizon):
        if t % version.estimator_refresh_interval == 0:
            with torch.no_grad():
                x = torch.as_tensor(env.estimator_input()[None], dtype=torch.float32, device=device)
                env.set_estimate(estimator.predict(x, version.estimator_samples)[0, 0].cpu().numpy())
            calls += 1
        if method == "learned":
            with torch.no_grad():
                x = torch.as_tensor(env.policy_observations(), dtype=torch.float32, device=device)
                scores = policy(x).cpu().numpy()
        elif method == "route_greedy":
            scores = route.scores(env)
        elif method == "local_greedy":
            scores = greedy_scores(env)
        elif method == "random":
            scores = rng.random((4, 8))
        else:
            raise ValueError("Unknown fixed baseline")
        env.step(env.safe_actions(scores, version.coordination_order))
        scores_history.append(env.evaluation()["normalized_weighted_idleness"])
    result = env.evaluation()
    result.update(case)
    result.update({"method": method, "mean_normalized_weighted_idleness": float(np.mean(scores_history)),
                   "estimator_calls": calls, "prior_samples": calls*version.estimator_samples})
    # Improvers may read this trace summary; it contains measurements and known
    # geometry only, never simulator truth or target-wise evaluator weights.
    result['public_diagnostics']={
        'final_positions':env.positions.tolist(),
        'held_executor_steps':sum(e['held_executors'] for e in env.events),
        'targets':[{'position':list(map(int,p)),'observed':bool(env.measured[tuple(p)]),
                    'last_measured_value':float(env.values[tuple(p)]) if env.measured[tuple(p)] else None,
                    'steps_since_inspection':int(env.raw_age[tuple(p)])} for p in np.argwhere(env.scenario.targets)],
        'trajectory_excerpt':env.events[::max(1,horizon//8)]}
    return result, env.events if record else []


def manifest_hash(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
