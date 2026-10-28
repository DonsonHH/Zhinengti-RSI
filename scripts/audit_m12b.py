"""Whole-case replay and independent M12-B action, selection and reset checks."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import patrol_rsi.improver_task as task
from patrol_rsi.improver_protocol import Improver, program_prompt, protocol_prompt
from patrol_rsi.m12b import specs, random_program, random_protocol
from patrol_rsi.m12 import actor_scorer, load_actors
from patrol_rsi.m11 import frozen_predictor, seed_digest
from audit_m11 import independent_schedule, independent_events
from audit_m111 import equivalent, independent_certificates, independent_power_events
from audit_m12 import independent_public_and_quota


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(value):
    return tuple(round(float(v), 8) for v in (value['violating_tasks'], 1 - value['coverage'],
        value['missed_segments'], value['weighted_censored_delay_s'] if value['weighted_censored_delay_s'] is not None else 0., value['mean_travel_m']))


ORIGINAL = task.choose_program
DECISIONS = 0


def independent_signal(env, ids, features, signal):
    role = {'age': 'all', 'thermal_excess': 'thermal', 'thermal_disagreement': 'thermal',
        'contact_heat': 'electrical', 'flow_deficit': 'cooling', 'verified_sensor_bias': 'control'}[signal]
    compatible = np.asarray([role == 'all' or env.tasks[int(j)].kind == role for j in ids])
    result = []
    for k, j in enumerate(ids):
        item, observer = env.tasks[int(j)], env.state.observer
        if signal == 'age':
            value = features[k, 0]
        elif not compatible[k]:
            value = 0.
        elif signal in ('thermal_excess', 'thermal_disagreement'):
            temperature = observer.predicted_temperature[item.index] if env.mode == 'nominal' else env.current_history[item.index]
            value = max(temperature - 65, 0) / 20 if signal == 'thermal_excess' else abs(temperature - observer.predicted_temperature[item.index]) / 20
        elif signal == 'contact_heat':
            value = max(observer.last_contact[item.index] - 60, 0) / 30
        elif signal == 'flow_deficit':
            value = max(1 - observer.last_flow[item.index] / 33.6, 0)
        else:
            value = float(observer.sensor_bias_verified[item.index])
        result.append(float(value))
    return np.asarray(result), compatible


def checked_choices(env, scorer, actor_tool='ppo'):
    global DECISIONS
    choices, logs = ORIGINAL(env, scorer, actor_tool)
    rows = {r['agent']: r for r in logs}
    reserved = env.reserved()
    for raw_i in np.roll(np.arange(len(env.agents)), -(env.tick % len(env.agents))):
        i = int(raw_i)
        if env.agents[i]['task'] is not None:
            continue
        ids, f, _ = env.options(i, reserved)
        if not len(ids):
            need(choices[i] is None, 'No candidates implies idle')
            continue
        actor_scores = scorer(f)
        actor = int(ids[int(np.argmax(actor_scores))])
        baseline = env.plans[i].tour[env.protected_cursor[i]]
        selected, tool, rule_index = None, None, None
        for k, rule in enumerate(env.workflow.rules):
            values, compatible = independent_signal(env, ids, f, rule.signal)
            candidates = np.flatnonzero(compatible & (values >= rule.threshold))
            if not len(candidates):
                continue
            if rule.action == 'return_tour':
                selected = int(baseline)
            else:
                if rule.action == 'ppo' and actor_tool == 'ppo':
                    priority = actor_scores
                elif rule.action in ('ppo', 'urgency'):
                    priority = f[:, 0] + f[:, 1]
                else:
                    priority = (.01 + values) * f[:, 6] / (.03 + f[:, 1])
                selected = int(ids[candidates[int(np.argmax(priority[candidates]))]])
            tool = 'urgency' if rule.action == 'ppo' and actor_tool == 'urgency' else rule.action
            rule_index = k
            break
        if selected is None:
            if env.workflow.fallback == 'tour':
                selected, tool = int(baseline), 'return_tour'
            elif actor_tool == 'ppo':
                selected, tool = actor, 'ppo'
            else:
                selected, tool = int(ids[int(np.argmax(f[:, 0] + f[:, 1]))]), 'urgency'
        need(choices[i] == selected, 'Independent typed choice mismatch')
        row = rows[i]
        need((row['task'], row['actor_task'], row['tool'], row['rule_index']) == (selected, actor, tool, rule_index), 'Decision attribution mismatch')
        need(row['legal_candidates'] == len(ids), 'Candidate count mismatch')
        reserved.add(selected)
        DECISIONS += 1
    return choices, logs


def independent_diagnostics(record):
    public = record['public_selection']
    result = {'coverage': {k: public[k] for k in ('coverage', 'violating_tasks', 'weighted_age_ratio', 'first_alert_tasks')},
        'efficiency': {k: public[k] for k in ('travel_m', 'service_s', 'extra_checks', 'estimator_calls', 'dispatches', 'different_from_actor')}}
    result['professional_evidence'] = {}
    fields = {'thermal': ('temperature_c', 'degC'), 'electrical': ('contact_temperature_c', 'degC'),
        'cooling': ('pod_flow_kg_s', 'kg/s'), 'control': ('controller_sensor_error_c', 'degC')}
    for role, (field, unit) in fields.items():
        events = [e for e in record['inspection_events'] if e['kind'] == role]
        result['professional_evidence'][role] = {'checks': len(events),
            'distinct_checked_tasks': len({e['task'] for e in events}),
            'distinct_measured_alarm_tasks': len({e['task'] for e in events if e['alarm']}),
            'new_measured_alerts': sum(e['measurement_transition'] == 'new_measured_alert' for e in events),
            'persistent_rechecks': sum(e['measurement_transition'] == 'persistent_recheck' for e in events),
            'recovery_confirmations': sum(e['measurement_transition'] == 'recovery_confirmation' for e in events),
            'measurement_field': field, 'unit': unit,
            'mean_measured_value': float(np.mean([e['measurement'][field] for e in events])) if events else None,
            'largest_public_completion_gap_s': max(g['maximum_gap_including_initial_and_terminal_s']
                for g in record['exact_task_gaps'] if g['task_id'].split(':')[0] == role)}
    observations = record['thermal_tool_observations']
    need([r['time_s'] for r in observations] == list(range(30, 1800, 30)), 'Public thermal tool clock')
    result['thermal_disagreement'] = {'meaning': 'TemperatureCVAE versus nominal temperature; thermal units only, not other professions',
        'mean_abs_difference_degC': float(np.mean([r['mean_abs_difference_degC'] for r in observations])),
        'peak_abs_difference_degC': max(r['peak_abs_difference_degC'] for r in observations)}
    return result


def combine(values):
    a = values[0]
    if isinstance(a, dict):
        return {k: combine([v[k] for v in values]) for k in a}
    if a is None or isinstance(a, str):
        need(all(v == a for v in values), 'Diagnostic schema mismatch')
        return a
    return float(np.mean(values))


def score(rows):
    weight = sum(r['event_weight_sum'] for r in rows)
    return {'cases': len(rows), 'anomaly_segments': sum(r['metrics']['anomaly_segments'] for r in rows),
        'missed_segments': sum(r['metrics']['missed_segments'] for r in rows),
        'weighted_censored_delay_s': sum(r['weighted_delay_sum'] for r in rows) / weight if weight else None,
        'violating_tasks': sum(r['metrics']['tasks_violating_exact_gap'] for r in rows),
        'coverage': float(np.mean([r['metrics']['final_professional_coverage'] for r in rows])),
        'mean_travel_m': float(np.mean([r['metrics']['travel_m'] for r in rows])),
        'scope': 'Development-only annotated evaluator; not raw labels supplied to the language model'}


def parse(raw, key):
    value = json.loads(raw[raw.index('{'):raw.rindex('}') + 1])
    if not isinstance(value, dict) or set(value) != {'diagnosis', key, 'expected_effect'}:
        raise ValueError('Response envelope')
    if any(not isinstance(value[k], str) or not 1 <= len(value[k]) <= 500 for k in ('diagnosis', 'expected_effect')):
        raise ValueError('Response text')
    return (task.PatrolProgram.from_dict(value[key]) if key == 'workflow' else Improver.from_dict(value[key])), value


def generation_check(generation, expected):
    equivalent(generation['prompt'], json.loads(json.dumps(expected)), 'public/limited-feedback prompt')
    need(json.dumps(generation['prompt']) in generation['rendered_prompt'], 'Actual rendered user object')
    need(generation['input_tokens'] <= 6144 and generation['output_tokens'] <= 768, 'Token caps')


class MetadataAudit:
    def __init__(self, results, index, views):
        self.results, self.index, self.views = results, index, views
        self.calls, self.proposal_count = [], 0
        self.by_signature = {}
        for row in index:
            signature = {k: row[k] for k in ('specification', 'workflow', 'deployment')}
            raw = json.dumps(signature, sort_keys=True)
            need(hashlib.sha256(raw.encode()).hexdigest()[:24] == row['case'], 'Case signature binding')
            self.by_signature[raw] = row

    def evaluate(self, cases, program, estimator='history', actor_tool='ppo'):
        rows = []
        for case in cases:
            signature = {'specification': dict(case, estimator=estimator, actor_seed=0),
                'workflow': program.to_dict(), 'deployment': {'actor_tool': actor_tool}}
            key = json.dumps(signature, sort_keys=True)
            need(key in self.by_signature, 'Missing required program/case/condition')
            rows.append(self.by_signature[key])
        return {'validator': score(rows), 'public': combine([self.views[r['case']] for r in rows]),
            'case_ids': [r['case'] for r in rows]}

    def adapt(self, tag, cases, protocol, seed, slots, *, random=False, checkpoints=()):
        parent = task.PatrolProgram()
        current = self.evaluate(cases, parent)
        starting = {'version_id': parent.version_id, 'workflow': parent.to_dict(), **current}
        seen, history, saved = {parent.version_id}, [], {}
        for slot in range(slots):
            entry = json.loads((self.results / 'proposals' / f'{tag}-{slot}.json').read_text())
            need(entry['parent_id'] == parent.version_id and entry['protocol_id'] == protocol.version_id, 'Parent/protocol chain')
            equivalent(entry['protocol'], protocol.to_dict())
            equivalent(entry['parent_validation'], current['validator'])
            need(entry['generation_seed'] == seed + slot, 'Paired candidate random seed')
            candidate, proposal = None, None
            if random:
                need('generation' not in entry, 'Random search cannot contain model replies')
                candidate = random_program(seed + slot)
                proposal = entry['proposal']
                equivalent(candidate.to_dict(), proposal['workflow'])
            else:
                generation = entry['generation']
                generation_check(generation, program_prompt(parent, current['public'], history, protocol))
                self.calls.append(generation)
                try:
                    candidate, proposal = parse(generation['raw_response'], 'workflow')
                except (ValueError, TypeError, KeyError):
                    need(not entry['valid'] and not entry['accepted'] and entry['reason'] == 'invalid', 'Invalid proposal consumes slot')
            if candidate is not None:
                need(entry['valid'] and entry['candidate_id'] == candidate.version_id, 'Actual raw candidate')
                equivalent(entry['proposal'], proposal)
                if candidate.version_id in seen:
                    need(not entry['accepted'] and entry['reason'] == 'duplicate', 'Duplicate consumes slot')
                else:
                    seen.add(candidate.version_id)
                    child = self.evaluate(cases, candidate)
                    equivalent(entry['candidate_validation'], child['validator'])
                    equivalent(entry['case_ids'], child['case_ids'])
                    accepted = rank(child['validator']) < rank(current['validator'])
                    need(entry['accepted'] == accepted, 'Independent development validator decision')
                    need(entry['reason'] == ('validated_improvement' if accepted else 'no_improvement'), 'Acceptance reason')
                    if accepted:
                        parent, current = candidate, child
            history.append({'candidate_id': entry.get('candidate_id'), 'workflow': proposal['workflow'] if proposal else None,
                'accepted': entry['accepted'], 'reason': entry['reason']})
            self.proposal_count += 1
            if slot + 1 in checkpoints or slot + 1 == slots:
                expected = {'tag': tag, 'candidate_slots': slot + 1, 'protocol': protocol.to_dict(),
                    'protocol_id': protocol.version_id, 'workflow': parent.to_dict(), 'version_id': parent.version_id,
                    **current, 'task_start': starting, 'history_start_length': 0, 'source': 'random' if random else 'open_model'}
                actual = json.loads((self.results / 'selections' / f'{tag}-{slot + 1}.json').read_text())
                equivalent(actual, expected, 'Reset initial program, empty history and selected snapshot')
                saved[slot + 1] = actual
        return {'selected': saved[slots], 'checkpoints': saved,
            'public_report': {'slots': slots, 'accepted': sum(h['accepted'] for h in history),
                'duplicates': sum(h['reason'] == 'duplicate' for h in history), 'invalid': sum(h['reason'] == 'invalid' for h in history)}}

    def meta(self, method, seed, smoke=False):
        incumbent, reports, retained = Improver(), [], []
        for number in range(1 if smoke else 2):
            entry = json.loads((self.results / 'meta' / f'{method}-round{number}.json').read_text())
            equivalent(entry['incumbent'], incumbent.to_dict())
            need(entry['incumbent_id'] == incumbent.version_id, 'Meta parent chain')
            if method == 'random_meta':
                candidate = random_protocol(seed * 10000 + 3000 + number)
                need(entry['valid'] and 'generation' not in entry, 'Random meta protocol')
            else:
                generation = entry['generation']
                generation_check(generation, protocol_prompt(incumbent, reports))
                self.calls.append(generation)
                try:
                    candidate, proposal = parse(generation['raw_response'], 'protocol')
                    need(entry['valid'], 'Valid protocol response')
                    equivalent(entry['proposal'], proposal)
                except (ValueError, TypeError, KeyError):
                    need(not entry['valid'], 'Invalid meta proposal')
                    candidate = incumbent
            equivalent(candidate.to_dict(), entry['candidate'])
            need(candidate.version_id == entry['candidate_id'], 'Protocol identity')
            cases = specs('smoke' if smoke else f'meta{number}')
            pair_seed = seed * 10000 + number * 10
            arms = {label: self.adapt(f'{method}-round{number}-{label}', cases, protocol, pair_seed, 1 if smoke else 2)
                for label, protocol in [('incumbent', incumbent), ('candidate', candidate)]}
            equivalent(entry['arms'], json.loads(json.dumps(arms)), 'Paired arm records')
            old, new = arms['incumbent']['selected']['validator'], arms['candidate']['selected']['validator']
            accepted = entry['valid'] and candidate.version_id != incumbent.version_id and rank(new) < rank(old)
            need(entry['accepted'] == accepted, 'Independent protocol update acceptance')
            if accepted:
                incumbent = candidate
            need(entry['retained_id'] == incumbent.version_id, 'Meta rollback or retention')
            need(entry['matched_inner_generation_seed'] == pair_seed, 'Same generation seed in both arms')
            reports.append({'round': number, 'protocol_update_accepted': accepted,
                'incumbent_search': arms['incumbent']['public_report'], 'candidate_search': arms['candidate']['public_report']})
            retained.append({'round': number, 'accepted': accepted, 'protocol_id': incumbent.version_id})
        return incumbent, retained

    def run(self, seed, smoke=False):
        if smoke:
            self.adapt('smoke-direct', specs('smoke'), Improver(), seed * 10000 + 8000, 1)
            protocol, updates = self.meta('model_meta', seed, smoke=True)
            self.adapt('smoke-transfer', specs('smoke'), protocol, seed * 10000 + 9000, 1)
            need(len(self.calls) == 5 and self.proposal_count == 4, 'Interface smoke budgets')
            return {'interface_model_calls': 5, 'meta_update_attempts': updates}
        direct = self.adapt('direct', specs('adapt'), Improver(), seed * 10000 + 8000, 13, checkpoints=(3,))
        random_search = self.adapt('random_search', specs('adapt'), Improver(), seed * 10000 + 9500, 13, random=True)
        random_improver, random_updates = self.meta('random_meta', seed)
        random_final = self.adapt('random_meta-transfer', specs('adapt'), random_improver, seed * 10000 + 8000, 5, checkpoints=(3,))
        model_improver, model_updates = self.meta('model_meta', seed)
        model_final = self.adapt('model_meta-transfer', specs('adapt'), model_improver, seed * 10000 + 8000, 3)
        expected = {'direct3': direct['checkpoints'][3], 'direct13': direct['selected'],
            'random_search13': random_search['selected'], 'random_meta3': random_final['checkpoints'][3],
            'random_meta5': random_final['selected'], 'model_meta3': model_final['selected']}
        frozen = json.loads((self.results / 'selections-frozen.json').read_text())
        equivalent(frozen, expected)
        journal = [json.loads(s) for s in (self.results / 'journal.jsonl').read_text().splitlines()]
        need([r['sequence'] for r in journal] == list(range(len(journal))), 'Journal sequence')
        freeze = next(r for r in journal if r['event'] == 'all_selections_frozen')
        need(freeze['sha256'] == sha(self.results / 'selections-frozen.json'), 'Frozen selections bytes')
        test_start = next(r for r in journal if r['event'] == 'test_begin')
        need(test_start['sequence'] > freeze['sequence'] and all(r['sequence'] > test_start['sequence']
            for r in journal if r['event'] == 'case_completed' and r['phase'] == 'test'), 'Test execution after all selections')
        groups = json.loads((self.results / 'test-groups.json').read_text())
        need(len(groups) == 11 and sum(map(len, groups.values())) == 352, 'Predeclared group scope')
        human = task.PatrolProgram(rules=(task.TypedRule('thermal_excess', .05, 'signal'), task.TypedRule('age', .75, 'return_tour')))
        programs = {'tour': task.PatrolProgram(), 'actor': task.PatrolProgram(fallback='ppo'), 'human': human}
        programs.update({name: task.PatrolProgram.from_dict(value['workflow']) for name, value in expected.items()})
        programs['model_meta3/no_actor'] = programs['model_meta3']
        programs['model_meta3/nominal'] = programs['model_meta3']
        for name, program in programs.items():
            result = self.evaluate(specs('test'), program,
                estimator='nominal' if name.endswith('/nominal') else 'history',
                actor_tool='urgency' if name.endswith('/no_actor') else 'ppo')
            equivalent(result['case_ids'], groups[name], 'Exact test process/noise/actor pairing')
        need(len(self.calls) == 39 and self.proposal_count == 50, 'All predeclared calls and candidate slots')
        return {'actual_model_calls': len(self.calls), 'task_candidate_slots': self.proposal_count,
            'random_protocol_updates': random_updates, 'model_protocol_updates': model_updates,
            'all_public_prompts_and_limited_feedback_chains_reconstructed': True,
            'all_meta_arms_reset_and_paired': True, 'only_protocol_transferred_to_fresh_tasks': True,
            'all_test_selections_frozen_before_test_execution': True,
            'generation_cost': {key: sum(r[key] for r in self.calls) for key in ('input_tokens', 'output_tokens', 'generation_seconds')}}


def main():
    p = argparse.ArgumentParser()
    for key in ('results', 'actors', 'estimator-weights', 'estimator-manifest', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--metadata-only', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    index = json.loads((args.results / 'case-index.json').read_text())
    summary = json.loads((args.results / 'summary.json').read_text())
    if args.metadata_only:
        result = MetadataAudit(args.results, index, {r['case']: r['public_diagnostics'] for r in index}).run(
            summary['generation_seed'], smoke=summary['smoke'])
        result['scope'] = 'Metadata preflight only; actual full replay still required'
        with args.output.open('x') as f:
            json.dump(result, f, indent=2)
        print('M12B_METADATA_PREFLIGHT_PASSED', flush=True)
        return
    actors, actor_fingerprints = load_actors(args.actors)
    predictor = frozen_predictor(args.estimator_weights, args.estimator_manifest)
    processes = json.loads((args.results / 'process-index.json').read_text())
    views, checked_processes = {}, set()
    dispatched, measured, accepted_plans = 0, 0, 0
    maximum_ratio, max_energy, max_electric = 0., 0., 0.
    task.choose_program = checked_choices
    try:
        for number, row in enumerate(index):
            path = args.results / row['path']
            need(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Raw case bytes')
            original = json.loads(gzip.decompress(path.read_bytes()))
            program = task.PatrolProgram.from_dict(row['workflow'])
            replay, env = task.run_program(row['specification'], program, predictor, actor_scorer(actors[0]), **row['deployment'])
            equivalent(replay, original, 'Actual actor/tool/observation replay')
            independent_schedule(original, env)
            independent_events(original, env)
            plans, count = independent_certificates(original, env)
            independent_power_events(env, original)
            independent_public_and_quota(original, env)
            need(original['metrics']['maximum_exact_gap_ratio'] <= 1., 'Protected revisit contract')
            views[row['case']] = independent_diagnostics(original)
            equivalent(views[row['case']], row['public_diagnostics'], 'Public professional diagnostics')
            need(all(e['time_s'] <= 1800 for e in original['inspection_events']), 'No fabricated tail completion')
            accepted_plans += plans
            dispatched += count
            measured += len(original['inspection_events'])
            maximum_ratio = max(maximum_ratio, original['metrics']['maximum_exact_gap_ratio'])
            if row['process_key'] not in checked_processes:
                info = processes[row['process_key']]
                npz = args.results / info['path']
                need(sha(npz) == info['sha256'] and npz.stat().st_size == info['bytes'], 'Physical array bytes')
                with np.load(npz, allow_pickle=False) as pack:
                    for key in pack.files:
                        np.testing.assert_allclose(pack[key], env.trace.arrays[key], rtol=1e-11, atol=1e-9)
                    c, scalar = env.config, pack['process_scalars']
                    energy = (c.chip_capacity_kj_k * pack['rack_temperature_c'].sum(1) +
                        c.secondary_capacity_kj_k * pack['secondary_supply_c'].sum(1) +
                        c.primary_capacity_per_pod_kj_k * c.pods * pack['primary_supply_c'] +
                        c.contact_capacity_kj_k * pack['contact_temperature_c'].sum(1))
                    residual = energy - np.r_[env.trace.metadata['initial_energy_kj'], energy[:-1]] - 5 * (scalar[:, 2] - scalar[:, 4])
                    electric = scalar[:, 2] - scalar[:, 5] - scalar[:, 6] - scalar[:, 7] - scalar[:, 8]
                    need(abs(residual).max() < 1e-4 and abs(electric).max() < 1e-6, 'Independent physical balances')
                    max_energy = max(max_energy, float(abs(residual).max()))
                    max_electric = max(max_electric, float(abs(electric).max()))
                checked_processes.add(row['process_key'])
            if number % 32 == 0:
                print('M12B_AUDIT', number + 1, '/', len(index), flush=True)
    finally:
        task.choose_program = ORIGINAL
    with args.output.with_suffix('.cases.json').open('x') as f:
        json.dump({'actual_cases_replayed': len(index), 'actual_dispatches': dispatched,
            'completed_measurements': measured, 'case_index_sha256': sha(args.results / 'case-index.json')}, f, indent=2)
    metadata = MetadataAudit(args.results, index, views).run(summary['generation_seed'], smoke=summary['smoke'])
    for seed, model in actors.items():
        need(seed_digest(model) == actor_fingerprints[str(seed)]['parameters_sha256'], 'Frozen actor unchanged')
    result = {'trial': summary['trial'], 'generation_seed': summary['generation_seed'], 'smoke': summary['smoke'],
        'all_actual_cases_replayed': len(index), 'physical_arrays_verified': len(checked_processes),
        'actual_dispatches': dispatched, 'completed_measurements': measured,
        'full_return_plans_independently_checked': accepted_plans, 'typed_choices_independently_checked': DECISIONS,
        'maximum_exact_gap_ratio': maximum_ratio, 'professional_diagnostics_independently_rebuilt': True,
        'quota_cooldown_and_exact_initial_terminal_gaps_verified': True,
        'maximum_independent_energy_residual_kj': max_energy, 'maximum_independent_electric_residual_kw': max_electric,
        **metadata, 'scope': 'Paired bounded procedural improver study; no weight updates or real-facility guarantee'}
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print('M12B_FULL_AUDIT_PASSED', len(index), flush=True)


if __name__ == '__main__':
    main()
