"""Persistent monitoring corresponding to paper Eqs. (1), (2), and (8).

The observation API contains measured values and estimates only. Ground truth
is accessed by the simulator and the evaluator, never by policy_observations.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numpy as np

# Same eight headings as the candidate source, with a deterministic hold fallback.
MOVES = np.array([(1, 0), (1, 1), (0, 1), (-1, 1),
                  (-1, 0), (-1, -1), (0, -1), (1, -1)], dtype=int)


@dataclass(frozen=True)
class Scenario:
    name: str
    navigation: np.ndarray
    targets: np.ndarray
    starts: np.ndarray
    radius: int = 1
    forgetting: float = 0.05
    background_importance: float = 0.1
    noise: float = 0.0

    def __post_init__(self) -> None:
        nav = np.asarray(self.navigation, dtype=bool)
        targets = np.asarray(self.targets, dtype=bool)
        starts = np.asarray(self.starts, dtype=int)
        if nav.ndim != 2 or targets.shape != nav.shape or not targets.any():
            raise ValueError("A nonempty target mask must match the navigation map")
        if np.any(targets & ~nav):
            raise ValueError("Targets must be reachable inspection positions")
        if starts.ndim != 2 or starts.shape[1] != 2 or len(set(map(tuple, starts))) != len(starts):
            raise ValueError("Executors need distinct start positions")
        for row, col in starts:
            if not (0 <= row < nav.shape[0] and 0 <= col < nav.shape[1] and nav[row, col]):
                raise ValueError("Start outside the free space")
        if self.radius < 0 or self.forgetting <= 0 or self.noise < 0 or self.background_importance <= 0:
            raise ValueError("Invalid observation or importance parameters")
        object.__setattr__(self, "navigation", nav.copy())
        object.__setattr__(self, "targets", targets.copy())
        object.__setattr__(self, "starts", starts.copy())


def make_scenario(name: str = "supercomputer", size: int = 24) -> Scenario:
    if size < 16:
        raise ValueError("Reference neural inputs require at least a 16x16 map")
    nav = np.ones((size, size), dtype=bool)
    nav[[0, -1], :] = False
    nav[:, [0, -1]] = False
    if name == "paper_grid":
        nav[7:12, 8:11] = False
        targets = nav.copy()
    elif name == "supercomputer":
        # Rack bodies form blocked rows. Inspections occur from aisle positions.
        targets = np.zeros_like(nav)
        for row in (6, size - 7):
            for col in (5, 9, size - 10, size - 6):
                nav[row:row + 2, col:col + 2] = False
                targets[row - 1, col] = True
    else:
        raise ValueError("Unknown scenario")
    starts = np.array([(2, 2), (2, size - 3), (size - 3, 2), (size - 3, size - 3)])
    return Scenario(name, nav, targets, starts)


def synthetic_field(shape: tuple[int, int], seed: int) -> np.ndarray:
    """Synthetic importance, not measured temperatures or calibrated fault risk."""
    rng = np.random.default_rng(seed)
    rr, cc = np.indices(shape)
    field = np.zeros(shape, dtype=np.float32)
    for _ in range(3):
        r, c = rng.uniform(2, shape[0] - 2), rng.uniform(2, shape[1] - 2)
        scale = rng.uniform(1.5, 4.0)
        field += rng.uniform(0.4, 1.0) * np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2 * scale ** 2))
    return np.clip(field / max(float(field.max()), 1e-8), 0, 1).astype(np.float32)


class PatrolEnv:
    def __init__(self, scenario: Scenario, seed: int = 0, truth: np.ndarray | None = None):
        self.scenario = scenario
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._truth = synthetic_field(scenario.navigation.shape, seed) if truth is None else np.asarray(truth, np.float32).copy()
        if self._truth.shape != scenario.navigation.shape or not np.isfinite(self._truth).all():
            raise ValueError("Bad simulator truth")
        if np.any(self._truth < 0) or np.any(self._truth > 1):
            raise ValueError("Simulator field must be normalized to [0,1]")
        self._neighborhoods: dict[tuple[int, int], np.ndarray] = {}
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.positions = self.scenario.starts.copy()
        self.t = 0
        self.values = np.zeros_like(self._truth)
        self.measured = np.zeros_like(self._truth, dtype=bool)
        self.idleness = np.zeros_like(self._truth)
        self.raw_age = np.zeros_like(self._truth, dtype=int)
        self.estimate = np.full_like(self._truth, 0.5)
        self.ever_measured = np.zeros_like(self.measured)
        self.events: list[dict] = []
        self._observe(np.any([self.neighborhood(p) for p in self.positions], axis=0))

    def valid_move(self, position: np.ndarray, action: int) -> bool:
        if action == -1:
            return True
        if action < 0 or action >= 8:
            return False
        nxt = position + MOVES[action]
        r, c = map(int, nxt)
        nav = self.scenario.navigation
        if not (0 <= r < nav.shape[0] and 0 <= c < nav.shape[1] and nav[r, c]):
            return False
        # Avoid cutting through rack corners, a declared indoor adaptation.
        if np.all(MOVES[action] != 0):
            return bool(nav[int(position[0]), c] and nav[r, int(position[1])])
        return True

    def neighborhood(self, position: np.ndarray) -> np.ndarray:
        key = tuple(map(int, position))
        if key in self._neighborhoods:
            return self._neighborhoods[key]
        found = {key}
        queue = deque([(key, 0)])
        while queue:
            point, distance = queue.popleft()
            if distance == self.scenario.radius:
                continue
            for a, move in enumerate(MOVES):
                nxt = tuple(np.array(point) + move)
                if nxt not in found and self.valid_move(np.array(point), a):
                    found.add(nxt)
                    queue.append((nxt, distance + 1))
        mask = np.zeros_like(self.scenario.navigation)
        for point in found:
            mask[point] = self.scenario.targets[point]
        self._neighborhoods[key] = mask
        return mask

    def _observe(self, mask: np.ndarray) -> None:
        observation = np.clip(self._truth[mask] + self.rng.normal(0, self.scenario.noise, int(mask.sum())), 0, 1)
        self.values[mask] = observation
        self.measured[mask] = True
        self.ever_measured[mask] = True
        self.estimate[mask] = observation

    def set_estimate(self, estimate: np.ndarray) -> None:
        value = np.asarray(estimate, np.float32)
        if value.shape != self.values.shape or not np.isfinite(value).all():
            raise ValueError("Invalid estimated map")
        self.estimate = np.clip(value, 0, 1).copy()
        self.estimate[self.measured] = self.values[self.measured]

    def estimator_input(self) -> np.ndarray:
        return np.stack([self.values, self.measured.astype(np.float32)])

    def legal_masks(self) -> np.ndarray:
        return np.array([[self.valid_move(p, a) for a in range(8)] for p in self.positions])

    def policy_observations(self) -> np.ndarray:
        observations = []
        for i, p in enumerate(self.positions):
            own = np.zeros_like(self.values)
            others = np.zeros_like(self.values)
            own[tuple(p)] = 1
            for j, q in enumerate(self.positions):
                if i != j:
                    others[tuple(q)] = 1
            observations.append(np.stack([self.scenario.navigation, self.estimate,
                                           self.idleness, own, others]).astype(np.float32))
        return np.stack(observations)

    def safe_actions(self, scores: np.ndarray, order: str = "q_priority") -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (len(self.positions), 8) or not np.isfinite(scores).all():
            raise ValueError("Need finite scores for every executor and action")
        if order not in ("q_priority", "cyclic"):
            raise ValueError("Unknown coordination order")
        indices = sorted(range(len(scores)), key=lambda i: (-float(scores[i].max()), i))
        if order == "cyclic":
            indices = list(np.roll(np.arange(len(scores)), -(self.t % len(scores))))
        occupied = {tuple(p) for p in self.positions}
        reserved: set[tuple[int, int]] = set()
        segments: list[tuple[np.ndarray, np.ndarray]] = []
        result = np.full(len(scores), -1, dtype=int)
        for i in indices:
            p = self.positions[i]
            for a in np.argsort(-scores[i], kind="stable"):
                nxt = p + MOVES[a]
                if not self.valid_move(p, int(a)) or tuple(nxt) in occupied or tuple(nxt) in reserved:
                    continue
                if any(np.array_equal(p + nxt, old + end) for old, end in segments):
                    continue  # opposing diagonals crossing at a midpoint
                result[i] = int(a)
                reserved.add(tuple(nxt))
                segments.append((p.copy(), nxt.copy()))
                break
            if result[i] == -1:
                reserved.add(tuple(p))
        return result

    def step(self, actions: np.ndarray) -> dict:
        raw = np.asarray(actions)
        if raw.shape != (len(self.positions),):
            raise ValueError("One action per executor is required")
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("Actions must be integers; fractional commands cannot be truncated")
        actions = raw.astype(int)
        nxt = self.positions.copy()
        for i, a in enumerate(actions):
            if not self.valid_move(self.positions[i], int(a)):
                raise ValueError("Illegal movement")
            if a >= 0:
                nxt[i] += MOVES[a]
        if len(set(map(tuple, nxt))) != len(nxt):
            raise ValueError("Two executors occupy the same destination")
        for i in range(len(nxt)):
            for j in range(i):
                if not np.array_equal(nxt[i], self.positions[i]) and not np.array_equal(nxt[j], self.positions[j]):
                    if np.array_equal(nxt[i] + self.positions[i], nxt[j] + self.positions[j]):
                        raise ValueError("Crossing or swapping movement")
        masks = np.stack([self.neighborhood(p) for p in nxt])
        count = masks.sum(axis=0)
        weighted = self.idleness * (self.scenario.background_importance + self.estimate)
        rewards = np.array([(weighted * mask / np.maximum(count, 1)).sum() for mask in masks], np.float32)
        self.idleness = np.where(count > 0, 0, np.minimum(1, self.idleness + self.scenario.forgetting))
        self.idleness *= self.scenario.targets
        self.raw_age = np.where(count > 0, 0, self.raw_age + 1) * self.scenario.targets
        movement = float(np.linalg.norm(nxt - self.positions, axis=1).sum())
        self.positions = nxt
        self.t += 1
        self._observe(count > 0)
        event = {"step": self.t, "actions": actions.tolist(), "positions": nxt.tolist(),
                 "rewards": rewards.tolist(), "movement_distance": movement,
                 "observed_target_cells": int((count > 0).sum()), "held_executors": int((actions == -1).sum())}
        self.events.append(event)
        return event

    def evaluation(self) -> dict:
        targets = self.scenario.targets
        weights = self.scenario.background_importance + self._truth
        return {"terminal_weighted_idleness_sum": float((self.idleness * weights * targets).sum()),
                "target_coverage": float(self.ever_measured[targets].mean()),
                "max_unobserved_steps": int(self.raw_age[targets].max()),
                "reconstruction_mse": float(((self.estimate[targets] - self._truth[targets]) ** 2).mean()),
                "total_movement_distance": sum(x["movement_distance"] for x in self.events),
                "steps": self.t}


def greedy_scores(env: PatrolEnv) -> np.ndarray:
    """Observed-estimate/idleness baseline; no access to hidden state."""
    value = (env.idleness + env.scenario.forgetting) * (env.scenario.background_importance + env.estimate)
    return np.array([[float((value * env.neighborhood(p + move)).sum()) if env.valid_move(p, a) else -1e6
                      for a, move in enumerate(MOVES)] for p in env.positions])
