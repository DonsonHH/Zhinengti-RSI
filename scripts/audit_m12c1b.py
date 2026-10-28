"""Independent public-mask, skill, sample, label and whole-loop replay audit."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import patrol_rsi.value_run as runner
from patrol_rsi.inspection_value import feature_rows
from patrol_rsi.m12 import actor_scorer, load_actors
from patrol_rsi.m11 import frozen_predictor, seed_digest
from patrol_rsi.m12c1b import specs, THRESHOLDS
from patrol_rsi.value_learning import identity
from audit_m11 import independent_schedule, independent_events
from audit_m111 import equivalent, independent_certificates, independent_power_events
from audit_m12 import independent_public_and_quota
import audit_m12c1 as masks


def need(value, message):
    if not value:
        raise RuntimeError(message)


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def independent_features(groups):
    result = []
    signals = ('thermal_excess', 'thermal_disagreement', 'contact_heat', 'flow_deficit', 'verified_sensor_bias', 'raw_remote_thermal_excess')
    for g in groups:
        available = g['candidates']
        if not available:
            continue
        baseline = [r for r in available if r['task'] == g['baseline_next']][0]
        for r in available:
            if r['task'] == g['baseline_next']:
                continue
            age_sum = sum(float(a['features'][0]) for a in available)
            extra = [r['minimum_return_slack_s']/300, (r['candidate_finish_s']-g['time_s'])/300,
                *baseline['features'][:3], baseline['minimum_return_slack_s']/300,
                (baseline['minimum_return_slack_s']-r['minimum_return_slack_s'])/300,
                age_sum/len(available), max(a['features'][0] for a in available),
                g['baseline_completed']/32, len(available)/32, len(groups)/18]
            x = np.array(r['features'] + [r['signals'][s] for s in signals] + extra, dtype=np.float32)
            result.append((g, r, x))
    return result


def independent_score(pack, x):
    if pack['kind'] == 'ridge':
        w = np.asarray(pack['weights']);return x @ w[:-1] + w[-1]
    state = {k: torch.tensor(v, dtype=torch.float32) for k,v in pack['state'].items()}
    value = torch.tensor(x, dtype=torch.float32)
    for i in (0, 2, 4):
        value = torch.nn.functional.linear(value, state[str(i)+'.weight'], state[str(i)+'.bias'])
        if i != 4:
            value = torch.tanh(value)
    return value.detach().squeeze(-1).numpy()


ORIGINAL_SNAPSHOT, ORIGINAL_CHOOSE = runner.snapshot, runner.choose
ACTIVE_PACK = None
FRAMES, OPTIONAL = 0, 0


def checked_snapshot(env, score):
    actual = ORIGINAL_SNAPSHOT(env, score)
    expected = masks.independent_snapshot(env, score)
    equivalent(actual, expected, 'C1B independent public masks and inputs')
    old, new = feature_rows(actual), independent_features(expected)
    need(len(old) == len(new), 'Optional feature row count')
    for a,b in zip(old,new):
        need((a[0]['agent'],a[1]['task']) == (b[0]['agent'],b[1]['task']), 'Optional feature identity')
        np.testing.assert_array_equal(a[2], b[2])
    return actual


def checked_choose(groups, method, *, used=False, threshold=0., score=None):
    global FRAMES, OPTIONAL
    actual, logs, consumed = ORIGINAL_CHOOSE(groups, method, used=used, threshold=threshold, score=score)
    expected = {g['agent']: g['baseline_next'] if g['candidates'] else None for g in groups}
    rows = independent_features(groups)
    eligible = []
    if not used and method != 'tour' and threshold != 'off':
        if method in ('net','target','untrained','ridge'):
            need(ACTIVE_PACK is not None, 'Frozen selected scoring model bound')
            x = np.stack([r[2] for r in rows]) if rows else np.zeros((0,34), dtype=np.float32)
            values = independent_score(ACTIVE_PACK, x)
            if rows:
                np.testing.assert_allclose(values, score(x), rtol=1e-6, atol=2e-7)
            eligible = [(float(v),g,r) for (g,r,_),v in zip(rows,values) if v > float(threshold)]
        elif method == 'cue_rule':
            eligible = [((.01+r['cue_strength'])*r['asset_weight_scaled']/(.03+r['eta_over_deadline']),g,r)
                for g,r,_ in rows if r['cue_strength'] >= .05 and r['known_measured_alarm'] is False]
        elif method == 'actor':
            for g,r,_ in rows:
                base = [a for a in g['candidates'] if a['task'] == g['baseline_next']][0]
                margin = r['actor_score'] - base['actor_score']
                if margin > 0:
                    eligible.append((margin,g,r))
    if eligible:
        value,g,r = eligible[int(np.argmax([a[0] for a in eligible]))]
        expected[g['agent']] = r['task']
        log = [a for a in logs if a['agent'] == g['agent']][0]
        need(log['tool'] == 'one_'+method and abs(log['skill_value']-value)<1e-6, 'Independent skill attribution')
    need(actual == expected and consumed == bool(eligible), 'Independent one-global-action controller')
    FRAMES += 1;OPTIONAL += int(consumed)
    return actual,logs,consumed


def positive_at_completion(env, j, finish):
    task, a, t = env.tasks[j], env.trace.arrays, int(finish/5)-1
    if task.kind == 'thermal':
        noise = np.random.default_rng(3000000+env.spec['seed']*10000+task.index*361+t).normal(0,.1)
        return a['rack_temperature_c'][t,task.index]+noise > 65
    if task.kind == 'electrical':
        return a['contact_temperature_c'][t,task.index] > 60
    if task.kind == 'cooling':
        return a['rack_flow_kg_s'][t].reshape(16,8)[task.index].sum() < 33.6
    delta = a['public_sensor_temperature_c'][t]-a['rack_temperature_c'][t]
    return abs(delta.reshape(16,8)[task.index].mean()) > 1


class IndependentCollector:
    def __init__(self, intervals):
        self.public, self.quality = {}, []
        self.confusion = dict(tp=0, fp=0, fn=0, tn=0)
        self.windows = {(r['task'],(t+1)*5.) for r in intervals
            for t in range(int(r['onset_s']/5), int(r['end_s']/5))}
        self.probe = masks.IndependentProbe(intervals)

    def observe(self, env, groups):
        self.probe.observe(env, groups)
        for g,r,x in independent_features(groups):
            key = (float(g['time_s']),int(g['agent']),int(r['task']))
            # Hash input uses only the three public action coordinates.
            digest = hashlib.sha256(('%g:%d:%d' % key).encode()).hexdigest()
            cue = r['cue_strength'] >= .05 and not r['known_measured_alarm']
            truth = (r['task'],r['candidate_finish_s']) in self.windows and positive_at_completion(env,r['task'],r['candidate_finish_s'])
            self.confusion['tp' if truth and cue else 'fp' if cue else 'fn' if truth else 'tn'] += 1
            self.public[key] = {'intervention': dict(zip(('dispatch_time_s','agent','task'),key)), 'x':x.tolist(),
                'cue': bool(cue), 'age':r['features'][0], 'hash':digest}
        if env.clock % 30 == 0:
            obs = env.state.observer
            last = np.asarray(obs.last_temperature)
            dt = np.array([env.clock-t if t >= 0 else env.clock for t in obs.temperature_stamp])
            self.quality.append({'time_s':env.clock, 'truth_c':env.trace.arrays['rack_temperature_c'][int(env.clock/5)-1].tolist(),
                'fresh': [bool(t == env.clock) for t in obs.temperature_stamp], 'hold':last.tolist(),
                'linear':np.maximum(35.,np.minimum(110.,last+env.state.velocity*dt)).tolist(),
                'nominal':obs.predicted_temperature.tolist(), 'history':env.current_history.tolist()})

    def sample(self, test):
        result = {};ordered = sorted(self.public.values(),key=lambda r:r['hash'])
        if test:
            pools = [('cue',[r for r in ordered if r['cue']],2), ('other',[r for r in ordered if not r['cue']],2)]
        else:
            pools = [('cue',[r for r in ordered if r['cue']],6),
                ('high_age',[r for r in ordered if not r['cue'] and r['age'] >= .5],3),
                ('other',[r for r in ordered if not r['cue'] and r['age'] < .5],3)]
        for label,pool,cap in pools:
            for row in pool[:cap]:
                result[tuple(row['intervention'].values())] = dict(row,provenance=['natural_'+label])
        if not test:
            segments = [r for r in self.probe.finish()['segments'] if not r['baseline_detected'] and r['first_witness'] is not None]
            for segment in segments[:4]:
                w = segment['first_witness'];key=(w['dispatch_time_s'],w['agent'],segment['task'])
                if key not in result:
                    result[key] = dict(self.public[key],provenance=[])
                result[key]['provenance'].append('teacher_segment_'+str(segment['segment_index']))
        return list(result.values())


def independent_labels(base, branch, target):
    old,new = base['anomaly_intervals'],branch['anomaly_intervals']
    need([(r['task'],r['onset_s'],r['end_s']) for r in old] == [(r['task'],r['onset_s'],r['end_s']) for r in new], 'Unchanged event identity')
    missed = sum(not r['detected'] for r in new)-sum(not r['detected'] for r in old)
    gains = [a['weight']*(a['delay_or_censor_s']-b['delay_or_censor_s']) for a,b in zip(old,new)]
    pooled = sum(gains)/sum(r['weight'] for r in old) if old else 0.
    target_weight = sum(r['weight'] for r in old if r['task']==target)
    target_gain = sum(v for r,v in zip(old,gains) if r['task']==target)/target_weight if target_weight else 0.
    recovered = [a['task'] for a,b in zip(old,new) if not a['detected'] and b['detected']]
    lost = [a['task'] for a,b in zip(old,new) if a['detected'] and not b['detected']]
    travel = sum(r['distance_m'] for r in branch['assignments'])-sum(r['distance_m'] for r in base['assignments'])
    service = sum(r['service_s'] for r in branch['assignments'])-sum(r['service_s'] for r in base['assignments'])
    return dict(net=-missed+pooled/3000-travel/100000, target=recovered.count(target)+max(target_gain,0)/3000,
        recovered_segments=len(recovered),collateral_lost_segments=len(lost),target_recovered_segments=recovered.count(target),
        missed_delta=missed,weighted_wait_gain_s=pooled,travel_delta_m=travel,service_delta_s=service)


def aggregate(rows):
    denom = sum(sum(r['weight'] for r in v['anomaly_intervals']) for v in rows)
    delay = sum(sum(r['weight']*r['delay_or_censor_s'] for r in v['anomaly_intervals']) for v in rows)
    return dict(cases=len(rows),segments=sum(len(r['anomaly_intervals']) for r in rows),
        missed=sum(sum(not x['detected'] for x in r['anomaly_intervals']) for r in rows),weighted_wait_s=delay/denom if denom else None,
        travel_m=sum(r['metrics']['travel_m'] for r in rows),service_s=sum(r['metrics']['service_s'] for r in rows),
        extra_checks=sum(r['metrics']['selected_detours'] for r in rows),
        false_measured_confirmations=sum(r['metrics']['false_measured_confirmations'] for r in rows),
        exact_violations=sum(r['metrics']['tasks_violating_exact_gap'] for r in rows))


def verify_training(folder,data):
    from patrol_rsi.value_learning import fit, ridge
    training=read(folder/'training.json');rows=read(data/'dataset.json')
    need(training['dataset_sha256']==sha(data/'dataset.json'),'Training data bytes')
    # Refit deterministic fixed updates and reconstruct frozen tensors; no test feedback.
    packs,logs=fit(rows,training['trial'],updates=training['updates'])
    if training['trial']==0:
        packs['ridge']=ridge(rows)
    equivalent(logs,training['training'],'All gradient update logs')
    for name,pack in packs.items():
        original=read(folder/'models'/(name+'.json'))
        need(identity(pack)==identity(original),'Deterministic final weights and paired initialization')
        x=np.asarray([r['x'] for r in rows],dtype=np.float32)
        state=original.get('state')
        if state:
            h=np.tanh(x@np.asarray(state['0.weight']).T+np.asarray(state['0.bias']))
            h=np.tanh(h@np.asarray(state['2.weight']).T+np.asarray(state['2.bias']))
            pred=(h@np.asarray(state['4.weight']).T+np.asarray(state['4.bias'])).ravel()
            np.testing.assert_allclose(pred,independent_score(original,x),rtol=3e-5,atol=2e-6)
        else:
            design=np.column_stack((x.astype(np.float64),np.ones(len(x))))
            w=np.asarray(original['weights']);y=np.asarray([r['labels']['net'] for r in rows])
            residual=design.T@(design@w-y)+np.r_[w[:-1],0.]
            need(abs(residual).max()<1e-6,'Independent ridge normal equation')
    return len(packs)


def audit(args):
    global ACTIVE_PACK
    folder=args.results;summary=read(folder/'summary.json');index=read(folder/'case-index.json')
    need(summary['actual_cases']==len(index),'Actual case count')
    stage=summary['stage'];smoke=summary.get('smoke',False)
    actors,fingerprints=load_actors(args.actors);actor=actor_scorer(actors[0])
    predictor=frozen_predictor(args.estimator_weights,args.estimator_manifest)
    inputs=read(folder/'immutable-inputs.json')
    need(inputs['estimator_sha256']==sha(args.estimator_weights),'Frozen estimator bytes')
    lookup={r['case']:r for r in index};records={};collectors={};processes=read(folder/'process-index.json');seen=set()
    samples={};dataset=read(folder/'dataset.json') if stage=='data' else (read(folder/'heldout-labels.json') if stage=='heldout' else [])
    for row in dataset:
        samples[row['branch_case']]=row
    quality={r['case']:r for r in read(folder/'quality.json')} if stage=='data' else (
        {r['case']:r for r in read(folder/'test-tour-samples-and-quality.json')} if stage=='test-common' else {})
    expected_samples={};dispatches=measurements=certs=0;max_energy=max_electric=0.
    runner.snapshot,runner.choose=checked_snapshot,checked_choose
    try:
        for number,row in enumerate(index):
            path=folder/row['path'];need(sha(path)==row['sha256'] and path.stat().st_size==row['bytes'],'Raw case bytes')
            original=json.loads(gzip.decompress(path.read_bytes()));dep=row['deployment']
            method=dep['public_controller'];kwargs={};ACTIVE_PACK=None
            if dep['model_id'] is not None:
                fit_folder=folder if stage=='fit' else args.run_root/('fit-'+str(0 if stage=='test-common' else summary['trial']))
                ACTIVE_PACK=read(fit_folder/'models'/(method+'.json'))
                need(identity(ACTIVE_PACK)==dep['model_id'],'Frozen model bound to actual deployment')
                kwargs['score']=lambda x:independent_score(ACTIVE_PACK,x)
            if row['case'] in quality:
                collector=IndependentCollector(original['anomaly_intervals']);collectors[row['case']]=collector;kwargs['observer']=collector
            replay,env=runner.run(row['specification'],predictor,actor,method,threshold=dep['threshold'],
                model_id=dep['model_id'],probe_enabled=dep['probe_enabled'],intervention=dep['evaluator_intervention'],**kwargs)
            equivalent(replay,original,'Whole-loop actual replay')
            independent_schedule(original,env);independent_events(original,env)
            passed,count=independent_certificates(original,env);independent_power_events(env,original);independent_public_and_quota(original,env)
            need(original['metrics']['selected_detours']<=1,'One optional global action')
            dispatches+=count;certs+=passed;measurements+=len(original['inspection_events'])
            if row['case'] in quality:
                saved=quality[row['case']];collector=collectors[row['case']]
                equivalent(saved['confusion'],collector.confusion,'All legal optional cue positive/negative samples')
                equivalent(saved['temperature'],collector.quality,'Same-history current-temperature quality arrays')
                equivalent(collector.probe.finish(),original['evaluator_probe'],'Independent teacher finite witnesses')
                if stage=='data' and row['specification']['phase'] in ('train','smoke'):
                    selected=collector.sample(False)[:3] if smoke else collector.sample(False)
                    for i,item in enumerate(selected):
                        key=row['case'].rsplit('--',1)[0]+'--label-'+str(i)
                        expected_samples[key]=dict(item,base_case=row['case'])
                if stage=='test-common':
                    equivalent(saved['samples'],collector.sample(True),'No-teacher heldout sample from public inputs')
            if row['case'] in samples:
                sample=samples[row['case']]
                if stage=='data':
                    need(row['case'] in expected_samples,'Predeclared natural or teacher training sample')
                    expected=expected_samples[row['case']]
                    equivalent({k:sample[k] for k in expected},expected,'Inputs and provenance before branch label')
                    base=records[sample['base_case']]
                else:
                    base=json.loads(gzip.decompress((args.run_root/'test-common/cases'/(sample['baseline']+'.json.gz')).read_bytes()))
                equivalent(dep['evaluator_intervention'],sample['intervention'],'One declared branch action')
                equivalent(sample['labels'],independent_labels(base,original,sample['intervention']['task']),'Independent target/net/collateral labels')
                for old,new in zip(base['action_log'],original['action_log']):
                    if old['time_s']<sample['intervention']['dispatch_time_s']:
                        need(old==new,'No pre-branch action changes')
                    elif old['time_s']==sample['intervention']['dispatch_time_s']:
                        for agent,target in old['choices'].items():
                            need(new['choices'][agent]==(sample['intervention']['task'] if int(agent)==sample['intervention']['agent'] else target),'Single override boundary')
                        break
            if row['process_key'] not in seen:
                info=processes[row['process_key']];p=folder/info['path'];need(sha(p)==info['sha256'],'Physical array bytes')
                with np.load(p,allow_pickle=False) as pack:
                    for k in pack.files:
                        np.testing.assert_allclose(pack[k],env.trace.arrays[k],rtol=1e-11,atol=1e-9)
                    c=env.config;scalar=pack['process_scalars']
                    energy=c.chip_capacity_kj_k*pack['rack_temperature_c'].sum(1)+c.secondary_capacity_kj_k*pack['secondary_supply_c'].sum(1)+c.primary_capacity_per_pod_kj_k*c.pods*pack['primary_supply_c']+c.contact_capacity_kj_k*pack['contact_temperature_c'].sum(1)
                    residual=energy-np.r_[env.trace.metadata['initial_energy_kj'],energy[:-1]]-5*(scalar[:,2]-scalar[:,4])
                    electric=scalar[:,2]-scalar[:,5]-scalar[:,6]-scalar[:,7]-scalar[:,8]
                    need(abs(residual).max()<1e-4 and abs(electric).max()<1e-6,'Independent physical balances')
                    max_energy=max(max_energy,float(abs(residual).max()));max_electric=max(max_electric,float(abs(electric).max()))
                seen.add(row['process_key'])
            records[row['case']]=original
            if number%32==0:
                print('C1B_AUDIT',stage,number+1,'/',len(index),flush=True)
    finally:
        runner.snapshot,runner.choose=ORIGINAL_SNAPSHOT,ORIGINAL_CHOOSE
    if stage=='data':
        need(set(samples)==set(expected_samples),'Complete train branch sample budget')
        baseline_specs=[r['specification'] for r in index if r['tag']=='tour-data']
        expected=specs('train',smoke)+(specs('validation') if not smoke else [])
        equivalent(baseline_specs,expected,'All declared data splits')
    trained=verify_training(folder,args.data) if stage=='fit' else 0
    if stage=='fit':
        selected=read(folder/'selections-frozen.json')['models']
        for name,entry in selected.items():
            options=[]
            for value in entry['all_candidates']:
                rows=[lookup[k] for k in value['case_ids']]
                equivalent([r['specification'] for r in rows],specs('validation',smoke),'All validation cases')
                equivalent(value['validation'],aggregate(rows),'Threshold validation aggregation')
                need(all(r['deployment']['threshold']==value['threshold'] for r in rows),'Actual validation threshold')
                options.append(value)
            need([v['threshold'] for v in options]==list(THRESHOLDS),'Same threshold budget')
            chosen=min(options,key=lambda r:(r['validation']['missed'],r['validation']['weighted_wait_s'] or 0.,r['validation']['travel_m']))
            need(entry['threshold']==chosen['threshold'],'Validation selection and fixed tie order')
    if stage.startswith('test'):
        gate=read(folder/'test-start-gate.json')
        need(len(gate['all_three_trials_frozen'])==3,'Every model frozen before test')
        for entry in gate['all_three_trials_frozen']:
            p=args.run_root/('fit-'+str(entry['trial']))/'selections-frozen.json'
            need(sha(p)==entry['selection_sha256'] and entry['frozen_at_unix']<gate['test_start_unix'],'Frozen selection hash and chronology')
        for name,entry in read(folder/'test-groups.json').items():
            rows=[lookup[k] for k in entry['case_ids']]
            expected=specs('test')
            if name=='net-nominal':
                expected=[dict(v,estimator='nominal') for v in expected]
            equivalent([r['specification'] for r in rows],expected,'Every predeclared test case')
            equivalent(entry['aggregate'],aggregate(rows),'Independent test aggregation')
            for row in rows:
                if row['deployment']['model_id'] is not None:
                    fit_folder=args.run_root/('fit-'+str(0 if stage=='test-common' else summary['trial']))
                    selection=read(fit_folder/'selections-frozen.json')['models'][row['deployment']['public_controller']]
                    need(row['deployment']['threshold']==selection['threshold'] and row['deployment']['model_id']==selection['model_id'],'Selected threshold/weights unchanged at test')
    if stage=='heldout':
        frozen=read(folder/'predictions-frozen-before-labels.json')
        need(len(frozen['samples'])==len(dataset),'All heldout predictions have labels')
        for predicted,labeled in zip(frozen['samples'],dataset):
            equivalent({k:labeled[k] for k in predicted},predicted,'Predictions saved before outcomes')
            for name,value in predicted['predictions'].items():
                trial,method=name.split('/')
                pack=read(args.run_root/('fit-'+trial)/'models'/(method+'.json'))
                need(abs(value-float(independent_score(pack,np.asarray([predicted['x']],dtype=np.float32))[0]))<1e-6,'Frozen heldout predictions')
    for seed,model in actors.items():
        need(seed_digest(model)==fingerprints[str(seed)]['parameters_sha256'],'Old actor parameters unchanged')
    return dict(stage=stage,all_actual_cases_replayed=len(index),physical_arrays_verified=len(seen),
        actual_dispatches=dispatches,completed_measurements=measurements,complete_return_plans_checked=certs,
        independent_public_candidates=masks.CANDIDATES_CHECKED,independent_public_choice_frames=FRAMES,
        optional_skill_actions=OPTIONAL,labels_independently_recomputed=len(dataset),quality_baselines=len(quality),
        frozen_models_refit=trained,exact_violations=0,global_optional_budget=1,
        maximum_energy_residual_kj=max_energy,maximum_electric_residual_kw=max_electric)


def main():
    p=argparse.ArgumentParser()
    for key in ('results','actors','estimator-weights','estimator-manifest','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--data',type=Path);p.add_argument('--run-root',type=Path)
    args=p.parse_args();torch.set_num_threads(1)
    result=audit(args)
    with args.output.open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print('C1B_AUDIT_PASSED',result['all_actual_cases_replayed'],flush=True)


if __name__=='__main__':
    main()
