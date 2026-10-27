"""Train frozen CVAE budgets, then evaluate on separately generated test cases."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .estimation import history_context, temperature_metrics
from .estimation_data import contract
from .estimation_model import TemperatureCVAE


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case(data, row):
    path = data / row['path']
    if sha(path) != row['sha256']:
        raise RuntimeError('Dataset changed: ' + row['case'])
    with np.load(path, allow_pickle=False) as pack:
        return {key: pack[key] for key in pack.files}


def matrix(data, index, split, current_only):
    xs, ys = [], []
    for row in index:
        if row['split'] != split:
            continue
        case = load_case(data, row)
        x = history_context(case['features'], current_only=current_only)
        xs.append(x.reshape(-1, x.shape[-1]))
        ys.append(((case['temperature_c'] - 50) / 20).reshape(-1, 1).astype(np.float32))
    if not xs:
        raise ValueError('Missing split ' + split)
    return np.concatenate(xs), np.concatenate(ys)


def parameter_digest(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def predict_chunks(model, x, device):
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(x), 8192):
            outputs.append(model.predict(torch.as_tensor(x[start:start + 8192], device=device)).cpu().numpy())
    return np.concatenate(outputs)


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU training requested; CPU fallback is forbidden')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    index = json.loads((args.data / 'index.json').read_text())
    if not args.smoke and {row['split'] for row in index} != {'train', 'validation'}:
        raise RuntimeError('Training process may open train/validation only')
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = (0,) if args.smoke else contract()['training_seeds']
    steps, interval = (20, 10) if args.smoke else (2000, 200)
    records = []
    for mode in ('current', 'history'):
        train_split = 'development' if args.smoke else 'train'
        val_split = 'development' if args.smoke else 'validation'
        x, y = matrix(args.data, index, train_split, mode == 'current')
        vx, vy = matrix(args.data, index, val_split, mode == 'current')
        xgpu, ygpu = torch.as_tensor(x, device='cuda'), torch.as_tensor(y, device='cuda')
        for seed in seeds:
            torch.manual_seed(seed)
            model = TemperatureCVAE(x.shape[1]).cuda()
            initial = parameter_digest(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            rng = np.random.default_rng(seed + 7000)
            learning, best, selected = [], float('inf'), None
            checkpoint = args.output / f'cvae_{mode}--seed-{seed}.pt'
            start = time.perf_counter()
            for step in range(1, steps + 1):
                indices = rng.integers(0, len(x), size=1024)
                loss, kl = model(xgpu[indices], ygpu[indices])
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % interval == 0:
                    prediction = predict_chunks(model, vx, 'cuda')
                    rmse = float(np.sqrt(np.mean((prediction - vy) ** 2)) * 20)
                    learning.append({'update': step, 'validation_rmse_c': rmse,
                                     'loss': float(loss.detach()), 'KL': float(kl.detach())})
                    if rmse < best:
                        best, selected = rmse, step
                        torch.save({'state_dict': model.state_dict(), 'inputs': x.shape[1],
                                    'mode': mode, 'seed': seed, 'selected_update': step,
                                    'normalization': {'center_c': 50, 'scale_c': 20}}, checkpoint)
            torch.cuda.synchronize()
            record = {'method': 'cvae_' + mode, 'seed': seed, 'updates': steps,
                      'batch_size': 1024, 'parameters': sum(p.numel() for p in model.parameters()),
                      'initial_parameters_sha256': initial, 'selected_update': selected,
                      'best_validation_rmse_c': best, 'learning': learning,
                      'training_nodes': len(x), 'validation_nodes': len(vx),
                      'seconds': time.perf_counter() - start, 'checkpoint': checkpoint.name,
                      'checkpoint_sha256': sha(checkpoint)}
            records.append(record)
            print(json.dumps(record), flush=True)
        del xgpu, ygpu
    result = {'smoke': args.smoke, 'device': torch.cuda.get_device_name(0),
              'torch': torch.__version__, 'numpy': np.__version__,
              'job_id': os.environ.get('SLURM_JOB_ID'), 'training_records': records,
              'data_index_sha256': sha(args.data / 'index.json'),
              'test_data_opened_by_training': False}
    (args.output / 'training.json').write_text(json.dumps(result, indent=2))


def evaluate(args):
    torch.set_num_threads(1)
    training = json.loads((args.models / 'training.json').read_text())
    if training['smoke']:
        raise RuntimeError('Development models cannot enter formal evaluation')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'private-predictions').mkdir()
    models = {}
    for record in training['training_records']:
        path = args.models / record['checkpoint']
        if sha(path) != record['checkpoint_sha256']:
            raise RuntimeError('Checkpoint changed')
        pack = torch.load(path, map_location='cpu', weights_only=True)
        model = TemperatureCVAE(pack['inputs'])
        model.load_state_dict(pack['state_dict'])
        model.eval()
        models[(record['method'], record['seed'])] = model
    index = json.loads((args.data / 'index.json').read_text())
    if len(index) != 318 or {r['split'] for r in index} != {'test', 'gw'}:
        raise RuntimeError('Frozen formal evaluation scope differs')
    rows, prediction_index = [], []
    for item in index:
        case = load_case(args.data, item)
        y, weights, fresh = case['temperature_c'], case['criticality'], case['fresh']
        predictions = {}
        for method in contract()['methods'][:5]:
            prediction = case['pred_' + method]
            rows.append(dict(case=item['case'], split=item['split'], condition=item['condition'],
                             scenario=item['scenario'], simulation_seed=item['seed'], control=item['control'],
                             method=method, training_seed=None,
                             metrics=temperature_metrics(y, prediction, weights, fresh),
                             mean_incremental_CPU_seconds=float(case['seconds_' + method].mean())))
            predictions[method] = prediction
        for method in ('cvae_current', 'cvae_history'):
            start = time.perf_counter()
            context = history_context(case['features'], current_only=method == 'cvae_current')
            context_seconds = (time.perf_counter() - start) / len(context)
            for seed in (0, 1, 2):
                model = models[(method, seed)]
                # Warm-up has no state or learning, and sees no evaluation label.
                with torch.inference_mode():
                    model.predict(torch.as_tensor(context[0]))
                values, elapsed = [], []
                for frame in context:
                    start = time.perf_counter()
                    with torch.inference_mode():
                        value = model.predict(torch.as_tensor(frame)).numpy().squeeze(-1) * 20 + 50
                    elapsed.append(time.perf_counter() - start)
                    values.append(value)
                prediction = np.stack(values)
                rows.append(dict(case=item['case'], split=item['split'], condition=item['condition'],
                                 scenario=item['scenario'], simulation_seed=item['seed'], control=item['control'],
                                 method=method, training_seed=seed,
                                 metrics=temperature_metrics(y, prediction, weights, fresh),
                                 mean_incremental_CPU_seconds=float(np.mean(elapsed)),
                                 context_preparation_seconds_per_frame=context_seconds))
                predictions[method + f'--seed-{seed}'] = prediction
        path = args.output / 'private-predictions' / (item['case'] + '.npz')
        np.savez_compressed(path, **predictions)
        prediction_index.append({'case': item['case'], 'path': str(path.relative_to(args.output)),
                                 'sha256': sha(path), 'bytes': path.stat().st_size})
        print('EVALUATED', item['case'], flush=True)
    (args.output / 'metrics.json').write_text(json.dumps(rows, indent=2))
    (args.output / 'prediction-index.json').write_text(json.dumps(prediction_index, indent=2))
    result = {'cases': len(index), 'method_condition_records': len(rows),
              'checkpoints_read': len(models), 'cpu_threads': 1, 'hostname': platform.node(),
              'torch': torch.__version__, 'numpy': np.__version__,
              'data_index_sha256': sha(args.data / 'index.json'),
              'training_manifest_sha256': sha(args.models / 'training.json'),
              'job_id': os.environ.get('SLURM_JOB_ID'),
              'latency_scope': 'Incremental inference on prepared public state; common updates and context preparation reported separately; simulator excluded'}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('train', 'evaluate'))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
