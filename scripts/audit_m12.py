"""Replay every unique M12 case and independently verify decisions and selection."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import torch

import patrol_rsi.facility_workflow as fw
from patrol_rsi.m11 import frozen_predictor, seed_digest
from patrol_rsi.m12 import (METHODS, GENERATION_SEEDS, FacilityWorkflow, actor_scorer,
    case_specs, load_actors, prompt, random_workflow, summarize_public)
from audit_m111 import equivalent, independent_certificates, independent_power_events


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(value):
    return tuple(round(float(value[k]) * sign, 8) for k, sign in (
        ('violating_tasks', 1), ('coverage', -1), ('first_alert_task_weight', -1),
        ('weighted_age_ratio', 1), ('travel_m', 1)))


ORIGINAL_CHOICES = fw.workflow_choices
checked_actor_choices = 0


def checked_choices(env, scorer, **deployment):
    """Recalculate rule predicates and ranking without production rule helpers."""
    global checked_actor_choices
    choices, logs = ORIGINAL_CHOICES(env, scorer, **deployment)
    log_by_agent = {r['agent']: r for r in logs}
    reserved = env.reserved()
    for raw_i in np.roll(np.arange(len(env.agents)), -(env.tick % len(env.agents))):
        i = int(raw_i)
        if env.agents[i]['task'] is not None:
            continue
        ids, features, _ = env.options(i, reserved)
        if not len(ids):
            require(choices[i] is None, 'Empty candidate set requires idle')
            continue
        actor = int(ids[int(np.argmax(scorer(features)))])
        signals = features.copy()
        if deployment.get('signals_permuted', False):
            order = np.random.default_rng(8800000 + env.spec['seed'] * 10000 + env.tick * 32 + i).permutation(len(ids))
            signals[:, 2], signals[:, 5] = features[order, 2], features[order, 5]
        rule_index = None
        if deployment.get('control', 'workflow') == 'cycle':
            selected, tool = env.plans[i].tour[env.protected_cursor[i]], 'cycle'
        else:
            selected = None
            for number, rule in enumerate(env.workflow.rules):
                if rule.role not in ('all', env.agents[i]['role']):
                    continue
                col = {'risk': 2, 'uncertainty': 5, 'age': 0}[rule.signal]
                candidates = np.flatnonzero(signals[:, col] >= rule.threshold)
                if not len(candidates):
                    continue
                f = signals[candidates]
                if rule.ranker == 'urgency':
                    values = f[:, 0] + f[:, 1]
                else:
                    values = (.01 + f[:, 2 if rule.ranker == 'risk' else 5]) * f[:, 6] / (.03 + f[:, 1])
                selected, tool = int(ids[candidates[int(np.argmax(values))]]), rule.ranker
                rule_index = number
                break
            if selected is None:
                if deployment.get('actor_fallback', 'ppo') == 'ppo':
                    selected, tool = actor, 'ppo'
                else:
                    selected, tool = int(ids[int(np.argmax(features[:, 0] + features[:, 1]))]), 'urgency_fallback'
        require(choices[i] == selected, 'Independent workflow action mismatch')
        log = log_by_agent[i]
        require(log['task'] == selected and log['actor_task'] == actor and log['tool'] == tool and log['rule_index'] == rule_index, 'Independent attribution mismatch')
        require(log['legal_candidates'] == len(ids), 'Candidate count')
        reserved.add(selected)
        checked_actor_choices += 1
    return choices, logs


def independent_public_and_quota(record, env):
    events = record['inspection_events']
    by_time = {}
    for event in events:
        by_time.setdefault(event['time_s'], []).append(event)
    by_dispatch = {}
    for row in record['workflow_decisions']:
        by_dispatch.setdefault(row['time_s'], []).append(row)
    last, extras, baseline_count, pending = np.zeros(len(env.tasks)), Counter(), Counter(), {}
    for t in range(5, 1801, 5):
        for event in by_time.get(t, []):
            i, j = event['agent'], event['task']
            previous = pending.pop(i)
            require(previous['task'] == j, 'Matching completion')
            if j == previous['baseline_next']:
                baseline_count[i] += 1
                if baseline_count[i] == len(env.agents[i]['tour']):
                    baseline_count[i] = extras[i] = 0
            last[j] = t
        for row in by_dispatch.get(t, []):
            i, j = row['agent'], row['task']
            require(row['detours_used_before'] == extras[i] and row['baseline_completed_before'] == baseline_count[i], 'Independent quota bookkeeping')
            if j != row['baseline_next']:
                require(extras[i] < 2, 'Extra-check quota')
                require(last[j] == 0 or t - last[j] >= record['workflow']['cooldown_s'], 'Extra-check cooldown')
                extras[i] += 1
            pending[i] = row
    require(not pending, 'All dispatched work completed')
    weights = np.asarray(record['workflow_contract']['task_weights'])
    np.testing.assert_array_equal(weights, env.asset_weights)
    np.testing.assert_array_equal(record['workflow_contract']['deadlines_s'], env.deadlines)
    grid = np.arange(5, 1801, 5)
    weighted_age = np.zeros(360)
    maximum, ever = [], 0
    for j in range(len(env.tasks)):
        times = np.asarray([0.] + sorted(e['time_s'] for e in events if e['task'] == j))
        maximum.append(float(np.diff(np.r_[times, 1800.]).max()))
        observed = times[np.searchsorted(times, grid, side='right') - 1]
        weighted_age += (grid - observed) / env.deadlines[j] * weights[j] / weights.sum()
        ever += len(times) > 1
    alerted = sorted({e['task'] for e in events if e['alarm']})
    transitions = {kind: sum(e['measurement_transition'] == kind for e in events) for kind in
        ('new_measured_alert', 'persistent_recheck', 'recovery_confirmation', 'normal_check')}
    logs = record['workflow_decisions']
    recomputed = {'violating_tasks': int((np.asarray(maximum) > env.deadlines + 1e-9).sum()),
        'coverage': ever / len(env.tasks), 'first_alert_tasks': len(alerted),
        'first_alert_task_weight': float(weights[alerted].sum()), 'weighted_age_ratio': float(weighted_age.mean()),
        'travel_m': sum(a['distance_m'] for a in record['assignments']),
        'service_s': sum(env.tasks[e['task']].service_s for e in events),
        'measurement_transitions': transitions,
        'estimator_calls': 1800 // record['workflow']['refresh_s'] if record['specification']['estimator'] == 'history' else 0,
        'dispatches': len(logs), 'extra_checks': sum(r['task'] != r['baseline_next'] for r in logs),
        'rule_dispatches': sum(r['rule_index'] is not None for r in logs),
        'ppo_dispatches': sum(r['tool'] == 'ppo' for r in logs),
        'forced_single_candidate': sum(r['legal_candidates'] == 1 for r in logs),
        'different_from_actor': sum(r['task'] != r['actor_task'] for r in logs)}
    equivalent(recomputed, record['public_selection'], 'independent_public_selection')
    require(recomputed['violating_tasks'] == 0, 'Exact initial/interior/terminal revisit')
    return recomputed


def verify_adaptation(results, index, public_by_case):
    lookup = {r['case']: r for r in index}
    selections = json.loads((results / 'selections-frozen.json').read_text())
    require(len(selections) == 9, 'Nine frozen selections')
    calls, slots, accepted, duplicate, invalid, cost = 0, 0, Counter(), 0, 0, Counter()
    for method in METHODS:
        for seed in GENERATION_SEEDS:
            parent = FacilityWorkflow()
            dev = [r for r in index if r['specification']['phase'] == 'development' and
                r['version_id'] == parent.version_id and r['specification']['actor_seed'] == 0]
            score = summarize_public([public_by_case[r['case']] for r in dev])
            require(len(dev) == 8, 'Common initial development cases')
            history, seen = [], {parent.version_id}
            for slot in range(3):
                entry = json.loads((results / 'proposals' / f'{method}-{seed}-{slot}.json').read_text())
                require(entry['parent_id'] == parent.version_id, 'Parent version chain')
                equivalent(entry['parent_public_summary'], score)
                slots += 1
                if method != 'random_search':
                    generation = entry['generation']
                    expected = json.loads(json.dumps(prompt(parent, score, history, memory=method == 'llm_memory')))
                    equivalent(generation['prompt'], expected, 'Explicit public-only prompt reconstruction')
                    # Numeric reconstruction can differ in the final float bit.
                    # Above checks the independent values; here bind the exact
                    # recorded input object to the string actually rendered.
                    require(json.dumps(generation['prompt']) in generation['rendered_prompt'], 'Rendered prompt includes actual supplied public user object')
                    require(generation['input_tokens'] <= 6144 and generation['output_tokens'] <= 768, 'Token envelopes')
                    calls += 1
                    for key in ('input_tokens', 'output_tokens', 'generation_seconds'):
                        cost[key] += generation[key]
                    text = generation['raw_response']
                    try:
                        value = json.loads(text[text.index('{'):text.rindex('}') + 1])
                        if set(value) != {'diagnosis', 'workflow', 'expected_effect'}:
                            raise ValueError('Model proposal grammar')
                        if any(not isinstance(value[k], str) or not 1 <= len(value[k]) <= 600 for k in ('diagnosis', 'expected_effect')):
                            raise ValueError('Proposal text type or size')
                        candidate = FacilityWorkflow.from_dict(value['workflow'])
                        require(entry['valid'], 'Valid raw response marked invalid')
                    except (ValueError, TypeError, KeyError):
                        require(not entry['valid'] and not entry['accepted'], 'Invalid proposal must consume slot')
                        invalid += 1
                        history.append({'valid': False, 'accepted': False, 'reason': entry['reason']})
                        continue
                    equivalent(value, entry['proposal'], 'Actual raw model response adopted without repair')
                else:
                    candidate = random_workflow(seed * 100 + slot)
                    equivalent(candidate.to_dict(), entry['proposal']['workflow'])
                require(candidate.version_id == entry['candidate_id'], 'Candidate identity')
                if candidate.version_id in seen:
                    require(not entry['accepted'] and entry['reason'] == 'duplicate_consumes_slot', 'Duplicate consumes slot')
                    duplicate += 1
                    history.append({'workflow': candidate.to_dict(), 'accepted': False, 'reason': entry['reason']})
                    continue
                seen.add(candidate.version_id)
                rows = [lookup[k] for k in entry['case_ids']]
                require(len(rows) == 8 and all(r['specification']['phase'] == 'development' and
                    r['specification']['actor_seed'] == 0 and r['version_id'] == candidate.version_id for r in rows), 'Candidate development scope')
                actual_specs = [{k: v for k, v in r['specification'].items() if k != 'actor_seed'} for r in rows]
                equivalent(actual_specs, case_specs('development'))
                child_score = summarize_public([public_by_case[r['case']] for r in rows])
                equivalent(child_score, entry['candidate_public_summary'])
                improve = rank(child_score) < rank(score)
                require(entry['accepted'] == improve, 'Independent accept or rollback')
                history.append({'workflow': candidate.to_dict(), 'public_summary': child_score, 'accepted': improve})
                if improve:
                    parent, score = candidate, child_score
                    accepted[method] += 1
            selected = next(s for s in selections if s['method'] == method and s['generation_seed'] == seed)
            equivalent(parent.to_dict(), selected['workflow'])
            equivalent(score, selected['public_summary'])
            require(selected['version_id'] == parent.version_id, 'Frozen selected identity')
    journal = [json.loads(s) for s in (results / 'journal.jsonl').read_text().splitlines()]
    require([r['sequence'] for r in journal] == list(range(len(journal))), 'Journal order')
    freeze = next(r for r in journal if r['event'] == 'all_selections_frozen')
    require(freeze['sha256'] == sha(results / 'selections-frozen.json'), 'Selections unchanged after freeze')
    test = next(r for r in journal if r['event'] == 'test_begin')
    require(test['sequence'] > freeze['sequence'], 'Test begins only after all selections')
    require(all(r['sequence'] > test['sequence'] for r in journal if r['event'] == 'case_completed' and r['phase'] == 'test'), 'No test cases before freeze')
    require(calls == 18 and slots == 27, 'Predeclared model and candidate budgets')
    return {'candidate_slots_checked': slots, 'actual_pilot_model_calls': calls, 'accepted_by_method': dict(accepted),
        'duplicate_slots': duplicate, 'invalid_slots': invalid, 'actual_generation_cost': dict(cost),
        'public_only_prompt_chains_reconstructed': True, 'selection_before_all_test_execution_verified': True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--actors', type=Path, required=True)
    p.add_argument('--estimator-weights', type=Path, required=True)
    p.add_argument('--estimator-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--metadata-only', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.metadata_only:
        index = json.loads((args.results / 'case-index.json').read_text())
        result = verify_adaptation(args.results, index, {r['case']: r['public_selection'] for r in index})
        result['scope'] = 'Metadata preflight only; complete independent case replay still required'
        with args.output.open('x') as f:
            json.dump(result, f, indent=2)
        print('M12_METADATA_PREFLIGHT_PASSED', flush=True)
        return
    actors, fingerprints = load_actors(args.actors)
    predictor = frozen_predictor(args.estimator_weights, args.estimator_manifest)
    helpers = runpy.run_path(str(Path(__file__).with_name('audit_m11.py')))
    index = json.loads((args.results / 'case-index.json').read_text())
    processes = json.loads((args.results / 'process-index.json').read_text())
    public, checked_processes, count_dispatch, count_events, certificates = {}, set(), 0, 0, 0
    maximum_ratio, maximum_energy, maximum_electric = 0., 0., 0.
    fw.workflow_choices = checked_choices
    try:
        for number, row in enumerate(index):
            path = args.results / row['path']
            require(sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Case bytes')
            record = json.loads(gzip.decompress(path.read_bytes()))
            repeated, env = fw.run_workflow(row['specification'], FacilityWorkflow.from_dict(row['workflow']),
                predictor, actor_scorer(actors[row['specification']['actor_seed']]), **row['deployment'])
            equivalent(repeated, record)
            helpers['independent_schedule'](record, env)
            helpers['independent_events'](record, env)
            accepted_plans, dispatches = independent_certificates(record, env)
            independent_power_events(env, record)
            public[row['case']] = independent_public_and_quota(record, env)
            certificates += accepted_plans
            count_dispatch += dispatches
            count_events += len(record['inspection_events'])
            maximum_ratio = max(maximum_ratio, record['metrics']['maximum_exact_gap_ratio'])
            if row['process_key'] not in checked_processes:
                process = processes[row['process_key']]
                source = args.results / process['path']
                require(sha(source) == process['sha256'], 'Process bytes')
                with np.load(source, allow_pickle=False) as pack:
                    for key in pack.files:
                        np.testing.assert_allclose(pack[key], env.trace.arrays[key], rtol=1e-11, atol=1e-9)
                    c, scalar = env.config, pack['process_scalars']
                    energy = (c.chip_capacity_kj_k * pack['rack_temperature_c'].sum(1) +
                        c.secondary_capacity_kj_k * pack['secondary_supply_c'].sum(1) +
                        c.primary_capacity_per_pod_kj_k * c.pods * pack['primary_supply_c'] +
                        c.contact_capacity_kj_k * pack['contact_temperature_c'].sum(1))
                    residual = energy - np.r_[env.trace.metadata['initial_energy_kj'], energy[:-1]] - 5 * (scalar[:, 2] - scalar[:, 4])
                    electric = scalar[:, 2] - scalar[:, 5] - scalar[:, 6] - scalar[:, 7] - scalar[:, 8]
                    require(abs(residual).max() < 1e-4 and abs(electric).max() < 1e-6, 'Independent physical balance')
                    maximum_energy = max(maximum_energy, float(abs(residual).max()))
                    maximum_electric = max(maximum_electric, float(abs(electric).max()))
                checked_processes.add(row['process_key'])
            if number % 32 == 0:
                print('M12_AUDIT', number + 1, '/', len(index), flush=True)
    finally:
        fw.workflow_choices = ORIGINAL_CHOICES
    with args.output.with_suffix('.cases.json').open('x') as f:
        json.dump({'all_unique_cases_replayed': len(index), 'case_index_sha256': sha(args.results / 'case-index.json'),
            'physical_arrays_independently_checked': len(checked_processes),
            'dispatches_checked': count_dispatch, 'completed_measurements_checked': count_events,
            'scope': 'Completed whole-case phase; selection metadata phase reported separately'}, f, indent=2)
    adaptation = {'interface_smoke_only': True} if args.smoke else verify_adaptation(args.results, index, public)
    for seed, model in actors.items():
        require(seed_digest(model) == fingerprints[str(seed)]['parameters_sha256'], 'Frozen actor unchanged')
    result = {'all_unique_cases_replayed': len(index), 'physical_arrays_independently_checked': len(checked_processes),
        'actual_dispatches_and_completed_measurements': [count_dispatch, count_events],
        'independently_checked_complete_return_plans': certificates,
        'independently_recalculated_actor_and_rule_decisions': checked_actor_choices,
        'maximum_exact_gap_ratio': maximum_ratio, 'all_quota_cooldown_and_public_outcomes_recomputed': True,
        'immutable_actor_bytes_and_parameters_verified': True,
        'maximum_independent_energy_residual_kj': maximum_energy,
        'maximum_independent_electric_residual_kw': maximum_electric, **adaptation,
        'scope': 'Frozen-improver integration pilot, not updated-improverRSI or field evidence'}
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print('M12_AUDIT_PASSED', len(index), flush=True)


if __name__ == '__main__':
    main()
