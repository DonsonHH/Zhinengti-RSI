"""Independent whole-dataset and actual-selected-weight replay for M10."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from patrol_rsi.estimation import history_context
from patrol_rsi.estimation_data import case_specs, case_name, generate_case, CompositeFacility
from patrol_rsi.estimation_model import TemperatureCVAE
from patrol_rsi.m10 import load_case, matrix, predict_chunks, parameter_digest


def require(value, message):
    if not value:
        raise RuntimeError(message)


def close_metadata(a, b):
    if isinstance(a, dict):
        require(set(a) == set(b), 'Metadata keys')
        for key in a:
            close_metadata(a[key], b[key])
    elif isinstance(a, list):
        require(len(a) == len(b), 'Metadata length')
        for left, right in zip(a, b):
            close_metadata(left, right)
    elif isinstance(a, float):
        require(np.isclose(a, b, rtol=1e-11, atol=1e-10), 'Metadata float')
    else:
        require(a == b, 'Metadata value')


def independent_metrics(y, p, weights, fresh):
    e = p - y
    w = np.broadcast_to(weights, y.shape)
    tp = int(((y > 65) & (p > 65)).sum())
    fp = int(((y <= 65) & (p > 65)).sum())
    fn = int(((y > 65) & (p <= 65)).sum())
    variance_sum = ((y - y.mean()) ** 2).sum()
    return dict(rmse_c=float(np.sqrt((e * e).mean())),
                weighted_rmse_c=float(np.sqrt((w * e * e).sum() / w.sum())),
                mae_c=float(abs(e).mean()), max_absolute_error_c=float(abs(e).max()),
                unmeasured_rmse_c=float(np.sqrt((e[~fresh] ** 2).mean())) if (~fresh).any() else None,
                r2=float(1 - (e * e).sum() / variance_sum) if variance_sum > 1e-12 else None,
                tp=tp, fp=fp, fn=fn, hot_count=int((y > 65).sum()),
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-data', type=Path, required=True)
    p.add_argument('--test-data', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    train = json.loads((args.train_data / 'index.json').read_text())
    test = json.loads((args.test_data / 'index.json').read_text())
    expected = {case_name(r) for r in case_specs('train')}
    require({r['case'] for r in train} == expected and len(train) == 280, 'Train/validation coverage')
    require({r['case'] for r in test} == {case_name(r) for r in case_specs('test')} and len(test) == 318, 'Test coverage')
    independent_balance = {'steps': 0, 'max_electric_kw': 0., 'max_energy_kj': 0.}
    original_step = CompositeFacility.step

    def checked_step(plant):
        c = plant.config
        def energy():
            return (c.chip_capacity_kj_k * plant._rack_c.sum() +
                    c.secondary_capacity_kj_k * plant._secondary_c.sum() +
                    c.primary_capacity_per_pod_kj_k * c.pods * plant._primary_c +
                    c.contact_capacity_kj_k * plant._contact_c.sum())
        before = energy()
        value = original_step(plant)
        residual = energy() - before - c.dt_s * (value['source_power_kw'] - value['rejected_heat_kw'])
        electric = value['source_power_kw'] - value['it_power_kw'] - value['aux_power_kw'] - value['conversion_loss_kw'] - value['contact_loss_kw']
        require(abs(residual) < 1e-4 and abs(electric) < 1e-6, 'Independent whole-data control-volume balance')
        independent_balance['steps'] += 1
        independent_balance['max_electric_kw'] = max(independent_balance['max_electric_kw'], abs(electric))
        independent_balance['max_energy_kj'] = max(independent_balance['max_energy_kj'], abs(residual))
        return value

    CompositeFacility.step = checked_step
    regenerated, frames, maximum_energy, maximum_electric, data_cache = 0, 0, 0., 0., {}
    for data, index in ((args.train_data, train), (args.test_data, test)):
        for row in index:
            stored = load_case(data, row)
            spec = {k: row[k] for k in ('split', 'condition', 'scenario', 'seed', 'control')}
            repeated, metadata = generate_case(spec, baselines=row['split'] in ('test', 'gw'))
            require(set(stored) == set(repeated), 'Stored keys')
            for key in repeated:
                if 'seconds' not in key:
                    np.testing.assert_allclose(stored[key], repeated[key], rtol=1e-11, atol=1e-10,
                                               err_msg=row['case'] + '/' + key)
            close_metadata(metadata, {key: row[key] for key in metadata})
            require(row['maximum_energy_residual_kj'] < 1e-4, 'Thermal balance')
            require(row['maximum_electric_residual_kw'] < 1e-6, 'Electrical balance')
            maximum_energy = max(maximum_energy, row['maximum_energy_residual_kj'])
            maximum_electric = max(maximum_electric, row['maximum_electric_residual_kw'])
            if row['split'] == 'gw':
                require(row['IT_nameplate_MW'] == 800 and row['global_supply_budget_MW'] == 1000, 'GW supply/IT units')
                require(row['peak_source_input_MW'] <= 1000, 'Observed GW design budget exceeded')
                require(not row['budget_limiter_implemented'], 'Do not claim global power control')
            if row['split'] in ('test', 'gw'):
                data_cache[row['case']] = stored
            regenerated += 1
            frames += len(stored['time_s'])
    training = json.loads((args.models / 'training.json').read_text())
    records = training['training_records']
    require(len(records) == 6 and not training['smoke'], 'Six formal models')
    models, initial_by_seed, selected_checks = {}, {}, []
    for row in records:
        require(row['updates'] == 2000 and row['batch_size'] == 1024, 'Training budget')
        require(row['training_nodes'] == 224 * 60 * 128 and row['validation_nodes'] == 56 * 60 * 128, 'Training split sizes')
        require([v['update'] for v in row['learning']] == list(range(200, 2001, 200)), 'Validation schedule')
        best = min(row['learning'], key=lambda v: v['validation_rmse_c'])
        require(row['selected_update'] == best['update'] and row['best_validation_rmse_c'] == best['validation_rmse_c'], 'Validation-only selection')
        torch.manual_seed(row['seed'])
        model = TemperatureCVAE()
        require(parameter_digest(model) == row['initial_parameters_sha256'], 'Actual paired initialization')
        initial_by_seed.setdefault(row['seed'], set()).add(row['initial_parameters_sha256'])
        path = args.models / row['checkpoint']
        require(hashlib.sha256(path.read_bytes()).hexdigest() == row['checkpoint_sha256'], 'Selected weight bytes')
        pack = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(pack['state_dict'])
        model.eval()
        vx, vy = matrix(args.train_data, train, 'validation', row['method'] == 'cvae_current')
        prediction = predict_chunks(model, vx, 'cpu')
        rmse = float(np.sqrt(np.mean((prediction - vy) ** 2)) * 20)
        require(abs(rmse - row['best_validation_rmse_c']) < 1e-4, 'Actual CPU prior validation replay')
        models[(row['method'], row['seed'])] = model
        selected_checks.append({'method': row['method'], 'seed': row['seed'], 'CPU_validation_rmse_c': rmse})
    require(all(len(v) == 1 for v in initial_by_seed.values()), 'Paired history/current weights')
    require(independent_balance['steps'] == 598 * 360, 'Full independent physical balance scope')
    metrics = json.loads((args.results / 'metrics.json').read_text())
    require(len(metrics) == 318 * 11, 'All baseline and model condition records')
    metric_map = {(r['case'], r['method'], r['training_seed']): r for r in metrics}
    require(len(metric_map) == len(metrics), 'Unique metrics')
    indices = json.loads((args.results / 'prediction-index.json').read_text())
    require({r['case'] for r in indices} == set(data_cache), 'All prediction files')
    metric_count, prior_prediction_values = 0, 0
    for row in indices:
        path = args.results / row['path']
        require(hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], 'Prediction file fingerprint')
        data = data_cache[row['case']]
        with np.load(path, allow_pickle=False) as pack:
            require(len(pack.files) == 11, 'All estimator arrays')
            for key in pack.files:
                if '--seed-' in key:
                    method, number = key.split('--seed-')
                    seed = int(number)
                    context = history_context(data['features'], current_only=method == 'cvae_current')
                    # Re-run the actual selected prior on all frames and racks.
                    values = []
                    with torch.inference_mode():
                        for frame in context:
                            values.append(models[(method, seed)].predict(torch.as_tensor(frame)).numpy().squeeze(-1) * 20 + 50)
                    prediction = np.stack(values)
                    np.testing.assert_allclose(prediction, pack[key], rtol=0, atol=1e-5)
                    prior_prediction_values += prediction.size
                else:
                    method, seed = key, None
                    np.testing.assert_equal(pack[key], data['pred_' + method])
                calculated = independent_metrics(data['temperature_c'], pack[key], data['criticality'], data['fresh'])
                stored = metric_map[(row['case'], method, seed)]['metrics']
                for metric, value in calculated.items():
                    if value is None:
                        require(stored[metric] is None, 'Undefined metric must stay null')
                    else:
                        require(np.isclose(value, stored[metric], rtol=1e-10, atol=1e-10), 'Independent metric: ' + metric)
                metric_count += 1
    result = {'all_dataset_cases_regenerated': regenerated, 'physical_steps_replayed': regenerated * 360,
              'evaluation_frames': frames, 'selected_checkpoints_read': len(models),
              'all_actual_prior_prediction_values_replayed': prior_prediction_values,
              'independent_method_condition_metrics': metric_count,
              'paired_initializations_verified': True, 'validation_only_selection_verified': True,
              'selected_validation_rechecks': selected_checks,
              'maximum_energy_residual_kj': maximum_energy, 'maximum_electric_residual_kw': maximum_electric,
              'independent_physical_balances': independent_balance,
              'gw_supply_budget_and_IT_distinction_verified': True,
              'scope': 'Whole synthetic data, selected weights, causal feature and metric implementation audit; not field calibration or model superiority'}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print('M10_FULL_AUDIT_PASSED', regenerated, 'cases;', metric_count, 'metric records')


if __name__ == '__main__':
    main()
