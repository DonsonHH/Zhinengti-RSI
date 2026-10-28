"""Actual C0 replay with independent public-event and information-arm checks."""
import argparse
from bisect import bisect_left
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import patrol_rsi.improver_task as task
from patrol_rsi.improver_protocol import Improver
from patrol_rsi.m12c import specs, diagnostic_prompt
from patrol_rsi.m12b import random_program
from patrol_rsi.m12 import actor_scorer, load_actors
from patrol_rsi.m11 import frozen_predictor, seed_digest
import audit_m12b as previous
from audit_m111 import equivalent

need, sha = previous.need, previous.sha


def public_events(record):
    """Reconstruct sequences by task and link to last matching public dispatch."""
    events = record['inspection_events']
    assignments = record['assignments']
    decisions = record['workflow_decisions']
    need(len(assignments) == len(decisions), 'Dispatch/decision count')
    field_map = {'thermal': ('temperature_c', 'degC'), 'electrical': ('contact_temperature_c', 'degC'),
        'cooling': ('pod_flow_kg_s', 'kg/s'), 'control': ('controller_sensor_error_c', 'degC')}
    candidates, prior_by_task = [], {}
    assignment_pairs = {}
    for k, assignment in enumerate(assignments):
        pair = (assignment['agent'], assignment['task'])
        assignment_pairs.setdefault(pair, []).append((assignment['time_s'], k, assignment))
    categories = ('new_measured_alert', 'persistent_recheck', 'recovery_confirmation', 'long_completion_gap')
    for event_index, event in enumerate(events):
        j, i, t = event['task'], event['agent'], event['time_s']
        pairs = assignment_pairs[(i, j)]
        position = bisect_left([r[0] for r in pairs], t) - 1
        need(position >= 0, 'No measurement before a matching dispatch')
        _, k, assignment = pairs[position]
        decision = decisions[k]
        need((i, j, assignment['time_s']) == (decision['agent'], decision['task'], decision['time_s']), 'Public provenance')
        need(event['calibrated'] and event['role'] == event['kind'], 'Only valid professional evidence')
        field, unit = field_map[event['kind']]
        value = float(event['measurement'][field])
        old = prior_by_task.get(j)
        gap = t - (old['time'] if old else 0.)
        card = {'task_id': event['task_id'], 'profession': event['kind'], 'measurement_field': field, 'unit': unit,
            'completed_at_s': float(t), 'value': value,
            'previous_completed_at_s': old['time'] if old else None, 'previous_value': old['value'] if old else None,
            'observed_value_delta': value - old['value'] if old else None,
            'interval_basis': 'previous_measurement' if old else 'mission_start_without_prior_measurement',
            'completion_gap_s': float(gap), 'gap_over_deadline': gap / record['workflow_contract']['deadlines_s'][j],
            'observed_transition': event['measurement_transition'], 'measured_alarm': event['alarm'],
            'dispatched_at_s': assignment['time_s'], 'travel_s_at1m_per_s': assignment['distance_m'],
            'service_s': assignment['service_s'], 'optional_check': j != decision['baseline_next'],
            'selected_tool': decision['tool'], 'source': {'event_index': event_index, 'agent': i, 'task': j}}
        prior_by_task[j] = {'time': float(t), 'value': value}
        if event['measurement_transition'] in categories[:3]:
            candidates.append({'category': event['measurement_transition'], 'card': card})
        candidates.append({'category': 'long_completion_gap', 'card': card})
    bounded = []
    for category in categories:
        rows = sorted([r for r in candidates if r['category'] == category],
            key=lambda r: (-r['card']['gap_over_deadline'], r['card']['completed_at_s'], r['card']['task_id']))
        bounded.extend(rows[:2])
    gaps = {}
    for role in field_map:
        values = [g['maximum_gap_including_initial_and_terminal_s'] for g in record['exact_task_gaps'] if g['task_id'].split(':')[0] == role]
        gaps[role] = {'mean_task_maximum_gap_s': sum(values) / len(values), 'worst_task_gap_s': max(values)}
    return bounded, gaps


def batch_event_evidence(values):
    categories = ('new_measured_alert', 'persistent_recheck', 'recovery_confirmation', 'long_completion_gap')
    pool = [dict(r, episode_alias=f'public_run_{episode}') for episode, rows in enumerate(values) for r in rows]
    selected, seen = [], set()
    for kind in categories:
        rows = sorted([r for r in pool if r['category'] == kind], key=lambda r:
            (-r['card']['gap_over_deadline'], r['card']['completed_at_s'], r['episode_alias'], r['card']['task_id']))
        count = 0
        for row in rows:
            source = (row['episode_alias'], row['card']['source']['event_index'])
            if source in seen:
                continue
            selected.append(row)
            seen.add(source)
            count += 1
            if count == 2:
                break
    return {'maximum_source_events': 8, 'selected_source_events': len(selected),
        'category_counts': {kind: sum(r['category'] == kind for r in selected) for kind in categories},
        'details_redacted': False, 'events': selected,
        'meaning': 'Only executed valid observations; no hidden anomaly onset or correctness label; gaps may begin at mission start.'}


class InformationAudit(previous.MetadataAudit):
    def __init__(self, results, index, views, cards, gaps):
        super().__init__(results, index, views)
        self.cards, self.gaps = cards, gaps

    def evaluate(self, cases, program, estimator='history', actor_tool='ppo'):
        result = super().evaluate(cases, program, estimator, actor_tool)
        ids = result['case_ids']
        result['event_evidence'] = batch_event_evidence([self.cards[k] for k in ids])
        values = [self.gaps[k] for k in ids]
        result['gap_extremes'] = {role: {'mean_episode_largest_gap_s': sum(v[role]['worst_task_gap_s'] for v in values) / len(values),
            'worst_observed_gap_s': max(v[role]['worst_task_gap_s'] for v in values)} for role in values[0]}
        return result

    def adapt_information(self, method, cases, seed, slots, checkpoints=()):
        program = task.PatrolProgram()
        current = self.evaluate(cases, program)
        starting = {'version_id': program.version_id, 'workflow': program.to_dict(), **current}
        seen, history, snapshots = {program.version_id}, [], {}
        for slot in range(slots):
            entry = json.loads((self.results / 'proposals' / f'{method}-{slot}.json').read_text())
            need(entry['generation_seed'] == seed + slot and entry['parent_id'] == program.version_id, 'Shared generation prefix and parent chain')
            equivalent(entry['parent_validation'], current['validator'])
            candidate, proposal = None, None
            if method == 'random':
                need('generation' not in entry, 'Random workflow has no model reply')
                candidate = random_program(seed + slot)
                proposal = entry['proposal']
                equivalent(candidate.to_dict(), proposal['workflow'])
            else:
                expected = diagnostic_prompt(program, current, history, method)
                previous.generation_check(entry['generation'], expected)
                prompt = entry['generation']['prompt']
                need('validator' not in prompt and 'missed_segments' not in json.dumps(prompt), 'No hidden numeric score in model input')
                if method == 'summary':
                    need('event_evidence' not in prompt, 'Summary arm cannot see selected raw events')
                else:
                    evidence = prompt['event_evidence']
                    need(evidence['selected_source_events'] <= 8 and max(evidence['category_counts'].values()) <= 2, 'Bounded event budget')
                    if method == 'redacted':
                        need(all(set(r) == {'category', 'details'} for r in evidence['events']), 'Redacted data must remove all event detail fields')
                    else:
                        need(not evidence['details_redacted'], 'Full event evidence must remain intact')
                self.calls.append(entry['generation'])
                try:
                    candidate, proposal = previous.parse(entry['generation']['raw_response'], 'workflow')
                except (ValueError, TypeError, KeyError):
                    need(not entry['valid'] and not entry['accepted'] and entry['reason'] == 'invalid', 'Invalid proposals consume slots')
            if candidate is not None:
                need(entry['valid'] and entry['candidate_id'] == candidate.version_id, 'Actual candidate bytes and grammar')
                equivalent(entry['proposal'], proposal)
                if candidate.version_id in seen:
                    need(not entry['accepted'] and entry['reason'] == 'duplicate', 'Duplicate consumes slot')
                else:
                    seen.add(candidate.version_id)
                    child = self.evaluate(cases, candidate)
                    equivalent(entry['candidate_validation'], child['validator'])
                    equivalent(entry['case_ids'], child['case_ids'])
                    accepted = previous.rank(child['validator']) < previous.rank(current['validator'])
                    need(entry['accepted'] == accepted and entry['reason'] == ('validated_improvement' if accepted else 'no_improvement'), 'Independent acceptance or rollback')
                    if accepted:
                        program, current = candidate, child
            history.append({'candidate_id': entry.get('candidate_id'), 'workflow': proposal['workflow'] if proposal else None,
                'accepted': entry['accepted'], 'reason': entry['reason']})
            self.proposal_count += 1
            if slot + 1 in checkpoints or slot + 1 == slots:
                expected = {'method': method, 'candidate_slots': slot + 1, 'workflow': program.to_dict(),
                    'version_id': program.version_id, **current, 'task_start': starting, 'history_start_length': 0}
                actual = json.loads((self.results / 'selections' / f'{method}-{slot + 1}.json').read_text())
                equivalent(actual, expected, 'Shared start and frozen information-arm selection')
                snapshots[slot + 1] = actual
        return {'selected': snapshots[slots], 'checkpoints': snapshots}

    def run_information(self, seed, smoke=False):
        phase = 'smoke' if smoke else 'adapt'
        adaptations = {name: self.adapt_information(name, specs(phase),
            seed * 10000 + (9500 if name == 'random' else 8000), 1 if smoke else slots,
            checkpoints=(3,) if not smoke and name == 'summary' else ())
            for name, slots in {'summary': 9, 'events': 3, 'redacted': 3, 'random': 9}.items()}
        need(len(self.calls) == (3 if smoke else 15) and self.proposal_count == (4 if smoke else 24), 'Predeclared information-study budgets')
        if not smoke:
            expected = {'summary3': adaptations['summary']['checkpoints'][3], 'summary9': adaptations['summary']['selected'],
                'events3': adaptations['events']['selected'], 'redacted3': adaptations['redacted']['selected'],
                'random9': adaptations['random']['selected']}
            frozen = json.loads((self.results / 'selections-frozen.json').read_text())
            equivalent(frozen, expected)
            journal = [json.loads(s) for s in (self.results / 'journal.jsonl').read_text().splitlines()]
            need([r['sequence'] for r in journal] == list(range(len(journal))), 'Journal order')
            freeze = next(r for r in journal if r['event'] == 'all_selections_frozen')
            test_start = next(r for r in journal if r['event'] == 'test_begin')
            need(freeze['sha256'] == sha(self.results / 'selections-frozen.json') and test_start['sequence'] > freeze['sequence'], 'Freeze before test')
            need(all(r['sequence'] > test_start['sequence'] for r in journal if r['event'] == 'case_completed' and r['phase'] == 'test'), 'No test outcomes used during adaptation')
            groups = json.loads((self.results / 'test-groups.json').read_text())
            need(len(groups) == 11 and sum(map(len, groups.values())) == 352, 'Predeclared test reference scope')
            programs = {'tour': task.PatrolProgram(), 'actor': task.PatrolProgram(fallback='ppo'),
                'human': task.PatrolProgram(rules=(task.TypedRule('thermal_excess', .05, 'signal'), task.TypedRule('age', .75, 'return_tour')))}
            programs.update({name: task.PatrolProgram.from_dict(value['workflow']) for name, value in frozen.items()})
            event_program = programs['events3']
            programs['events3/no_actor'] = event_program
            programs['events3/nominal'] = event_program
            inactive = replace(event_program, rules=tuple(r for r in event_program.rules if r.signal == 'age'))
            programs['events3/rule_signals_inactive'] = inactive
            for name, program in programs.items():
                result = self.evaluate(specs('test'), program,
                    estimator='nominal' if name.endswith('/nominal') else 'history',
                    actor_tool='urgency' if name.endswith('/no_actor') else 'ppo')
                equivalent(groups[name], result['case_ids'])
            deployment = json.loads((self.results / 'deployment-interventions.json').read_text())
            equivalent(deployment['rule_signals_inactive']['original_program'], event_program.to_dict())
            equivalent(deployment['rule_signals_inactive']['effective_program'], inactive.to_dict())
        return {'actual_model_calls': len(self.calls), 'task_candidate_slots': self.proposal_count,
            'all_event_provenance_and_redaction_checks_passed': True,
            'all_information_prompts_and_acceptance_chains_reconstructed': True,
            'all_starts_reset_and_test_selected_before_execution': not smoke,
            'generation_cost': {k: sum(r[k] for r in self.calls) for k in ('input_tokens', 'output_tokens', 'generation_seconds')}}


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
        audit = InformationAudit(args.results, index, {r['case']: r['public_diagnostics'] for r in index},
            {r['case']: r['public_event_candidates'] for r in index}, {r['case']: r['public_gap_cards'] for r in index})
        result = audit.run_information(summary['generation_seed'], summary['smoke'])
        result['scope'] = 'Metadata preflight only; full raw replay still required'
    else:
        actors, fingerprints = load_actors(args.actors)
        predictor = frozen_predictor(args.estimator_weights, args.estimator_manifest)
        processes = json.loads((args.results / 'process-index.json').read_text())
        views, cards, gaps, checked_processes = {}, {}, {}, set()
        count, plans, measurements = 0, 0, 0
        maximum_ratio, max_energy, max_electric = 0., 0., 0.
        task.choose_program = previous.checked_choices
        try:
            for number, row in enumerate(index):
                path = args.results / row['path']
                need(sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Raw case bytes')
                original = json.loads(gzip.decompress(path.read_bytes()))
                replay, env = task.run_program(row['specification'], task.PatrolProgram.from_dict(row['workflow']),
                    predictor, actor_scorer(actors[0]), **row['deployment'])
                equivalent(replay, original)
                previous.independent_schedule(original, env)
                previous.independent_events(original, env)
                passed, dispatched = previous.independent_certificates(original, env)
                previous.independent_power_events(env, original)
                previous.independent_public_and_quota(original, env)
                count += dispatched;plans += passed;measurements += len(original['inspection_events'])
                maximum_ratio = max(maximum_ratio, original['metrics']['maximum_exact_gap_ratio'])
                views[row['case']] = previous.independent_diagnostics(original)
                cards[row['case']], gaps[row['case']] = public_events(original)
                equivalent(views[row['case']], row['public_diagnostics'])
                equivalent(cards[row['case']], row['public_event_candidates'])
                equivalent(gaps[row['case']], row['public_gap_cards'])
                if row['process_key'] not in checked_processes:
                    info = processes[row['process_key']];source = args.results / info['path']
                    need(sha(source) == info['sha256'] and source.stat().st_size == info['bytes'], 'Physical array bytes')
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
                        need(abs(residual).max() < 1e-4 and abs(electric).max() < 1e-6, 'Independent physical balances')
                        max_energy = max(max_energy, float(abs(residual).max()));max_electric = max(max_electric, float(abs(electric).max()))
                    checked_processes.add(row['process_key'])
                if number % 32 == 0:
                    print('M12C_AUDIT', number + 1, '/', len(index), flush=True)
        finally:
            task.choose_program = previous.ORIGINAL
        with args.output.with_suffix('.cases.json').open('x') as f:
            json.dump({'actual_cases_replayed': len(index), 'dispatches': count, 'case_index_sha256': sha(args.results / 'case-index.json')}, f, indent=2)
        result = InformationAudit(args.results, index, views, cards, gaps).run_information(summary['generation_seed'], summary['smoke'])
        for seed, model in actors.items():
            need(seed_digest(model) == fingerprints[str(seed)]['parameters_sha256'], 'Frozen actor parameters')
        result.update(actual_closed_loops_replayed=len(index), physical_arrays_verified=len(checked_processes),
            actual_dispatches=count, completed_measurements=measurements, complete_return_plans_checked=plans,
            typed_choices_checked=previous.DECISIONS, maximum_exact_gap_ratio=maximum_ratio,
            maximum_independent_energy_residual_kj=max_energy, maximum_independent_electric_residual_kw=max_electric,
            scope='Information representation study; no new neural training or recursive protocol update')
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print('M12C_AUDIT_PASSED', len(index), flush=True)


if __name__ == '__main__':
    main()
