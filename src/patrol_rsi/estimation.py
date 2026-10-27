"""M10 observation-only features and bounded state-estimation controls.

No simulator reference, target temperature or hidden scenario is accepted here.
Physical scaling is fixed before seeing validation/test data.
"""
from __future__ import annotations

import numpy as np

from .facility_observation import FacilityObserver, rack_positions

FEATURES = ('last_temperature', 'measurement_age', 'ever_measured', 'pod_measured_mean',
            'secondary_supply', 'delivered_power', 'requested_power', 'pump_command',
            'last_flow', 'flow_age', 'last_contact', 'sensor_verified', 'local_x', 'local_y')
PACKET_KEYS = frozenset(('time_s', 'temperature_indices', 'temperature_c', 'public_requested_it_kw',
                        'public_delivered_it_kw', 'pod_pump_command', 'primary_supply_c',
                        'secondary_supply_c', 'flow_pod_indices', 'flow_pod_kg_s'))


class ObservationState:
    def __init__(self, config):
        self.config = config
        self.observer = FacilityObserver(config)
        self.previous_temperature = np.full(config.racks, 50.)
        self.previous_stamp = np.full(config.racks, -1.)
        self.velocity = np.zeros(config.racks)

    def update(self, packet, completed):
        if set(packet) != PACKET_KEYS:
            raise ValueError('Only the frozen public telemetry schema is accepted')
        t = float(packet['time_s'])
        if any(event['time_s'] > t for event in completed):
            raise ValueError('Future inspection')
        old_temp = self.observer.last_temperature.copy()
        old_stamp = self.observer.temperature_stamp.copy()
        self.observer.update(packet)
        for task, role, measurement, when in (event['completion'] for event in completed):
            self.observer.complete_inspection(task, role, measurement, when)
        new_stamp = self.observer.temperature_stamp
        newer = new_stamp > old_stamp
        dt = new_stamp - old_stamp
        valid = newer & (old_stamp >= 0) & (dt > 0)
        self.velocity[valid] = np.clip(
            (self.observer.last_temperature[valid] - old_temp[valid]) / dt[valid], -.2, .2)
        self.previous_temperature, self.previous_stamp = old_temp, old_stamp
        return self.observer.readout(packet)

    def features(self, packet):
        c, obs = self.config, self.observer
        t = float(packet['time_s'])
        known = obs.temperature_stamp >= 0
        values = obs.last_temperature.reshape(c.pods, 8)
        masks = known.reshape(c.pods, 8)
        pod_mean = np.divide((values * masks).sum(1), masks.sum(1),
                             out=np.full(c.pods, 50.), where=masks.sum(1) > 0)
        age = np.where(known, t - obs.temperature_stamp, t)
        flow_age = np.where(obs.flow_stamp >= 0, t - obs.flow_stamp, t)
        repeat = lambda x: np.repeat(x, 8)
        local = np.arange(c.racks) % 8
        return np.stack(((obs.last_temperature - 50) / 20, np.minimum(age / 300, 6),
                         known.astype(float), repeat((pod_mean - 50) / 20),
                         repeat((packet['secondary_supply_c'] - 35) / 20),
                         packet['public_delivered_it_kw'] / 250,
                         packet['public_requested_it_kw'] / 250,
                         repeat(packet['pod_pump_command']),
                         repeat(obs.last_flow / 48), repeat(np.minimum(flow_age / 300, 6)),
                         repeat((obs.last_contact - 38) / 20),
                         repeat(obs.sensor_bias_verified.astype(float)),
                         (local % 4) / 3, (local // 4).astype(float)), axis=-1).astype(np.float32)


def history_context(frames, history=4, current_only=False):
    """Only indices <= the requested frame; first-frame padding is explicit."""
    a = np.asarray(frames)
    if a.ndim != 3 or history < 1:
        raise ValueError('Expected time x rack x feature')
    times = np.arange(len(a))
    offsets = [0] * history if current_only else list(range(history - 1, -1, -1))
    return np.concatenate([a[np.maximum(0, times - offset)] for offset in offsets], axis=-1)


def radial_knn(last, stamp, positions, time_s, k=4):
    known = np.flatnonzero(stamp >= 0)
    if not len(known):
        return np.full(len(last), 50.)
    d2 = ((positions[:, None] - positions[known][None]) ** 2).sum(-1) / 8 ** 2
    d2 += ((time_s - stamp[known])[None] / 90) ** 2
    nearest = np.argsort(d2, axis=1)[:, :min(k, len(known))]
    distance = np.take_along_axis(d2, nearest, axis=1)
    weights = 1 / np.maximum(distance, 1e-6)
    return (weights * last[known][nearest]).sum(1) / weights.sum(1)


def bounded_gp(last, stamp, positions, time_s, capacity=64):
    if capacity < 1:
        raise ValueError('Positive GP budget required')
    known = np.flatnonzero(stamp >= 0)
    if not len(known):
        return np.full(len(last), 50.)
    ordered = known[np.argsort(-stamp[known], kind='stable')]
    recent = ordered[stamp[ordered] >= stamp[ordered[0]] - 90]
    # With hundreds of simultaneous sensors, take distributed rack-index
    # quantiles rather than retaining only the first facility region.
    if len(recent) >= capacity:
        known = recent[np.linspace(0, len(recent) - 1, capacity, dtype=int)]
    else:
        known = ordered[:capacity]
    mean = float(last[known].mean())
    x = np.column_stack((positions[known] / 8, (stamp[known] - time_s) / 90))
    q = np.column_stack((positions / 8, np.zeros(len(last))))
    kernel = lambda a, b: np.exp(-.5 * ((a[:, None] - b[None]) ** 2).sum(-1))
    chol = np.linalg.cholesky(kernel(x, x) + .01 * np.eye(len(x)))
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, last[known] - mean))
    return mean + kernel(q, x) @ alpha


def nonlearned(state, packet):
    obs, c = state.observer, state.config
    t = packet['time_s']
    age = np.where(obs.temperature_stamp >= 0, t - obs.temperature_stamp, 0)
    return {'hold': obs.last_temperature.copy(),
            'linear': obs.last_temperature + state.velocity * np.minimum(age, 120),
            'nominal_model': obs.predicted_temperature.copy(),
            'rknn': radial_knn(obs.last_temperature, obs.temperature_stamp, rack_positions(c), t),
            'gp64': bounded_gp(obs.last_temperature, obs.temperature_stamp, rack_positions(c), t)}


def temperature_metrics(truth, estimate, weights, fresh=None):
    y, pred = np.asarray(truth), np.asarray(estimate)
    if y.shape != pred.shape or not np.isfinite(pred).all():
        raise ValueError('Invalid prediction')
    error = pred - y
    w = np.broadcast_to(weights, y.shape)
    denominator = np.sum((y - y.mean()) ** 2)
    hot, found = y > 65, pred > 65
    tp, fp, fn = int((hot & found).sum()), int((~hot & found).sum()), int((hot & ~found).sum())
    unmeasured = ~np.asarray(fresh) if fresh is not None else np.ones_like(y, dtype=bool)
    return {'rmse_c': float(np.sqrt(np.mean(error ** 2))),
            'weighted_rmse_c': float(np.sqrt(np.sum(w * error ** 2) / w.sum())),
            'mae_c': float(np.mean(abs(error))), 'max_absolute_error_c': float(abs(error).max()),
            'unmeasured_rmse_c': float(np.sqrt(np.mean(error[unmeasured] ** 2))) if unmeasured.any() else None,
            'r2': float(1 - np.sum(error ** 2) / denominator) if denominator > 1e-12 else None,
            'tp': tp, 'fp': fp, 'fn': fn, 'hot_count': int(hot.sum()),
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None}
