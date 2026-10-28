"""Small supervised value models; frozen JSON weights are portable evidence."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .inspection_value import FEATURE_NAMES


def identity(pack):
    return hashlib.sha256(json.dumps(pack, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def network():
    import torch
    return torch.nn.Sequential(torch.nn.Linear(len(FEATURE_NAMES), 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))


def freeze(model, seed, objective, updates):
    return {'kind': 'mlp34_32_32_1_tanh', 'seed': seed, 'objective': objective, 'updates': updates,
        'feature_names': list(FEATURE_NAMES), 'state': {k: v.tolist() for k, v in model.state_dict().items()}}


def fit(rows, seed, *, updates=400):
    import torch
    torch.manual_seed(seed)
    initial = network()
    frozen = {'untrained': freeze(initial, seed, 'none', 0)}
    x = torch.tensor([r['x'] for r in rows], dtype=torch.float32)
    logs = {}
    for objective in ('net', 'target'):
        model = network()
        model.load_state_dict(initial.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        y = torch.tensor([r['labels'][objective] for r in rows], dtype=torch.float32)
        history = []
        for step in range(updates):
            loss = ((model(x).squeeze(-1) - y) ** 2).mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
            if step == 0 or (step + 1) % 100 == 0 or step == updates - 1:
                history.append({'update': step + 1, 'loss_before_update': float(loss.detach())})
        model.eval()
        frozen[objective] = freeze(model, seed, objective, updates)
        logs[objective] = {'history': history, 'final_training_mse': float(((model(x).squeeze(-1) - y) ** 2).mean().detach()),
            'parameters': sum(p.numel() for p in model.parameters())}
    return frozen, logs


def ridge(rows):
    x = np.asarray([r['x'] + [1.] for r in rows], dtype=np.float64)
    y = np.asarray([r['labels']['net'] for r in rows], dtype=np.float64)
    penalty = np.eye(x.shape[1]);penalty[-1, -1] = 0.
    w = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    return {'kind': 'ridge', 'objective': 'net', 'l2': 1., 'feature_names': list(FEATURE_NAMES), 'weights': w.tolist()}


def scorer(pack):
    if pack['feature_names'] != list(FEATURE_NAMES):
        raise ValueError('Frozen feature schema mismatch')
    if pack['kind'] == 'ridge':
        w = np.asarray(pack['weights'])
        return lambda x: np.asarray(x) @ w[:-1] + w[-1]
    import torch
    model = network().eval()
    model.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in pack['state'].items()})
    def predict(x):
        with torch.inference_mode():
            return model(torch.as_tensor(x, dtype=torch.float32)).squeeze(-1).numpy()
    return predict
