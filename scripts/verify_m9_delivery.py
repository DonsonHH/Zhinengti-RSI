"""Local M9 transport, scalar-balance, grouping and reference verification."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha(raw): return hashlib.sha256(raw).hexdigest()
def near(a, b): return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
def require(condition, message):
    if not condition: raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--research', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); evidence = args.research / 'evidence'
    sources = {}
    for name in ('source_manifest_v1.json','report_source_manifest_v1.json','package_source_manifest_v1.json'):
        sources.update(json.loads((args.research/name).read_text())['files'])
    for name, fingerprint in sources.items():
        require(sha((args.workspace/name).read_bytes().replace(b'\r\n',b'\n')) == fingerprint, f'Source changed: {name}')
    transfer = json.loads((evidence/'transfer-files.json').read_text())
    path = evidence/'text-normalization.json'
    normalized = json.loads(path.read_text())['files'] if path.exists() else {}
    for name, fingerprint in transfer.items():
        raw=(evidence/name).read_bytes()
        if name in normalized:
            row=normalized[name]
            require(row['original_sha256']==fingerprint['sha256'] and row['original_bytes']==fingerprint['bytes'],'Original SVG transfer')
            require(name.endswith('.svg') and row['xml_token_content_identical'],'Allowed SVG processing')
            require(sha(raw)==row['normalized_sha256'] and len(raw)==row['normalized_bytes'],'Normalized SVG fingerprint')
        else:
            require(len(raw)==fingerprint['bytes'] and sha(raw)==fingerprint['sha256'], f'Transfer changed: {name}')
    ids=json.loads((evidence/'job-identities.json').read_text())
    with (evidence/'slurm-outcomes-final.psv').open() as handle:jobs={row['JobID']:row for row in csv.DictReader(handle,delimiter='|')}
    for job in ids.values():
        require(jobs[str(job)]['State']=='COMPLETED' and jobs[str(job)]['ExitCode']=='0:0',f'Job incomplete: {job}')
    log=(evidence/f"jobs/smoke-{ids['smoke']}.err").read_text()
    require('Ran 94 tests' in log and log.rstrip().endswith('OK') and 'skipped=' not in log,'Full 94-test log')
    audit=json.loads((evidence/'full-chain-audit.json').read_text())
    require(audit['cases_replayed']==150 and audit['numeric_microsteps_replayed']==54000 and audit['full_private_numeric_traces_read']==150,'Raw replay scope')
    require(audit['independent_electric_and_control_volume_balance_passed'],'Independent balance')
    records=[]
    for item in json.loads((evidence/'case-index.json').read_text()):
        compressed=(evidence/item['record']).read_bytes();raw=gzip.decompress(compressed)
        require(sha(compressed)==item['stored_sha256'] and sha(raw)==item['uncompressed_sha256'],'Stored/original case bytes')
        record=json.loads(raw);require(len(record['scalar_trace'])==360,'Complete scalar trace')
        rows=record['scalar_trace']; dt=record['configuration']['dt_s']; before=record['initial_energy_kj']
        for row in rows:
            residual=row['stored_energy_kj']-before-dt*(row['source_power_kw']-row['rejected_heat_kw'])
            require(abs(residual)<1e-4,'Local independent heat balance');before=row['stored_energy_kj']
            electric=row['source_power_kw']-row['it_power_kw']-row['aux_power_kw']-row['conversion_loss_kw']-row['contact_loss_kw']
            require(abs(electric)<1e-6,'Local independent electricity balance')
        m=record['metrics'];require(near(statistics.mean(row['impact_weighted_severity'] for row in rows),m['mean_impact_weighted_severity']),'Severity average')
        require(near(sum(row['unserved_it_kw'] for row in rows)*dt/3600,m['unserved_IT_kwh']),'Service energy units')
        for method in ('hold','nominal_model'):
            require(near(math.sqrt(statistics.mean(row[f'{method}_mse_c2'] for row in rows)),m[f'{method}_temperature_rmse_c']),'Temperature RMSE')
        records.append(record)
    require(len(records)==150,'All final conditions')
    aggregate=json.loads((evidence/'report/aggregate.json').read_text())
    for scenario, modes in aggregate['groups'].items():
        for mode, metrics in modes.items():
            cases=sorted([r for r in records if r['scenario']==scenario and r['control']==mode],key=lambda r:r['seed'])
            require([r['seed'] for r in cases]==list(range(5)),'Five paired seeds')
            for metric, entry in metrics.items():
                values=[r['metrics'][metric] for r in cases]
                require(all(near(a,b) for a,b in zip(values,entry['per_simulation_seed'])),'Group seed values')
                require(near(statistics.mean(values),entry['mean']) and near(statistics.stdev(values),entry['sample_sd_across_simulation_seeds']),'Group mean/SD')
    for scenario, metrics in aggregate['paired_feedback_minus_fixed'].items():
        for metric, entry in metrics.items():
            left=aggregate['groups'][scenario]['fixed_commands'][metric]['per_simulation_seed'];right=aggregate['groups'][scenario]['feedback'][metric]['per_simulation_seed']
            values=[b-a for a,b in zip(left,right)]
            require(all(near(a,b) for a,b in zip(values,entry['per_simulation_seed'])),'Control paired effects')
            require(near(statistics.mean(values),entry['mean']) and near(statistics.stdev(values),entry['sample_sd_across_simulation_seeds']),'Effect mean/SD')
    storage=json.loads((evidence/'raw-numeric-storage.json').read_text())
    require(len(storage['files'])==150 and storage['all_numeric_arrays_replayed_by_independent_job'],'Remote full-array retention')
    result={'transport_and_statistics_verified':True,'source_files':len(sources),'transferred_files':len(transfer),
            'successful_terminal_jobs':len(ids),'full_tests':94,'final_cases':150,'numeric_microsteps':54000,
            'remote_full_numeric_traces_audited':150,'all_scalar_heat_and_power_balances_recomputed_locally':True,
            'all_scenario_means_sample_sds_and_control_pairings_recomputed':True,
            'scope':'Does not calibrate plant parameters, prove standard compliance or validate VAE/B-PID/RSI; full arrays retained remotely, scalar/demo arrays retrieved'}
    with args.output.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
