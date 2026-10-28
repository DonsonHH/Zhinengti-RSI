"""C0: summary versus causal event evidence under the same program grammar."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import gzip
import json
import os
from pathlib import Path
import time

import torch

from .event_evidence import bounded_case_candidates, batch_events, scalar_cards, TOOL_CARDS
from .improver_task import PatrolProgram, TypedRule
from .improver_protocol import Improver, program_prompt, validation_rank
from .m12b import Caller, TrialStore, parse, specification, random_program, DEVELOPMENT, TEST
from .m12 import load_actors, verify_open_model, sha, write
from .m11 import frozen_predictor, seed_digest

SEEDS = (7501, 7502, 7503)
SLOTS = {'summary': 9, 'events': 3, 'redacted': 3, 'random': 9}


def contract():
    return {'stage': 'M12-C0 event-diagnostic information study', 'generation_seeds': list(SEEDS),
        'candidate_slots_per_trial': SLOTS, 'formal_model_calls_per_trial': 15,
        'summary3_is_prefix_of_summary9': True, 'task_candidate_slots_per_trial': 24,
        'initial_program': PatrolProgram().to_dict(), 'development_seeds': [7300, 7301],
        'test_seeds': [7400, 7401], 'test_physical_processes': 16, 'test_noise_conditions': 32,
        'test_noise_c': [.2, 1.], 'test_groups_per_trial': 11, 'test_group_references_per_trial': 352,
        'model': 'Qwen/Qwen3-8B', 'revision': 'b968826d9c46dd6066d109eabc6255188de91218',
        'input_token_cap': 6144, 'output_token_cap': 768, 'maximum_event_cards': 8,
        'protocol_updated': False, 'neural_training': False,
        'shared_grammar_and_annotated_validator': 'M12-B unchanged; public measurements plus limited development acceptance feedback in prompts',
        'scope': 'Compare public evidence representations on known-layout new instances; completeC1/C2/C3research remains'}


def specs(phase):
    if phase == 'smoke':
        return [specification(phase, a, b, 7299) for a, b in (('healthy', 900), ('coldplate_fouling', 180))]
    if phase == 'adapt':
        return [specification(phase, a, b, seed) for a, b in DEVELOPMENT for seed in (7300, 7301)]
    if phase == 'test':
        return [specification(phase, a, b, seed, noise) for a, b in TEST for seed in (7400, 7401) for noise in (.2, 1.)]
    raise ValueError('Unknown C0 split')


def gap_extremes(values):
    return {role: {'mean_episode_largest_gap_s': sum(v[role]['worst_task_gap_s'] for v in values) / len(values),
        'worst_observed_gap_s': max(v[role]['worst_task_gap_s'] for v in values)} for role in values[0]}


class EventStore(TrialStore):
    def __init__(self, output, predictor, actors):
        super().__init__(output, predictor, actors)
        self.cards, self.gaps = {}, {}

    def evaluate(self, cases, program, *, estimator='history', actor_tool='ppo'):
        result = super().evaluate(cases, program, estimator=estimator, actor_tool=actor_tool)
        for key in result['case_ids']:
            if key not in self.cards:
                row = self.lookup[key]
                record = json.loads(gzip.decompress((self.output / row['path']).read_bytes()))
                self.cards[key] = bounded_case_candidates(record)
                self.gaps[key] = scalar_cards(record)
                row['public_event_candidates'] = self.cards[key]
                row['public_gap_cards'] = self.gaps[key]
        result['event_evidence'] = batch_events([self.cards[k] for k in result['case_ids']])
        result['gap_extremes'] = gap_extremes([self.gaps[k] for k in result['case_ids']])
        return result


def diagnostic_prompt(program, evaluation, history, mode):
    if mode not in ('summary', 'events', 'redacted'):
        raise ValueError('Unknown information arm')
    prompt = program_prompt(program, evaluation['public'], history, Improver())
    prompt['information_arm'] = mode
    # All arms receive the same explicit tool meanings and gap definitions.
    prompt['tool_cards'] = TOOL_CARDS
    prompt['public_gap_extremes'] = evaluation['gap_extremes']
    prompt['aggregation_note'] = 'Legacy summary values are means over development episodes. public_gap_extremes separately names means and worst recorded gaps.'
    if mode != 'summary':
        events = copy.deepcopy(evaluation['event_evidence'])
        if mode == 'redacted':
            events['events'] = [{'category': r['category'],
                'details': 'Task, time, measurement and cost fields withheld by the diagnostic control'} for r in events['events']]
            events['details_redacted'] = True
        prompt['event_evidence'] = events
    prompt['constraints'].append('Event-card categories are public observations, not hidden fault labels. Never infer a tool capability from its name alone.')
    return prompt


def adapt(store, caller, method, cases, seed, slots, checkpoints=(), initial=None):
    program = PatrolProgram()
    current = initial if initial is not None else store.evaluate(cases, program)
    starting = {'version_id': program.version_id, 'workflow': program.to_dict(), **current}
    history, seen, saved = [], {program.version_id}, {}
    for slot in range(slots):
        entry = {'method': method, 'slot': slot, 'generation_seed': seed + slot,
            'parent_id': program.version_id, 'parent_validation': current['validator']}
        candidate, proposal = None, None
        if method == 'random':
            candidate = random_program(seed + slot)
            proposal = {'diagnosis': 'Random program control', 'workflow': candidate.to_dict(), 'expected_effect': 'No model hypothesis'}
        else:
            generation = caller.generate(diagnostic_prompt(program, current, history, method), seed + slot)
            entry['generation'] = generation
            try:
                candidate, proposal = parse(generation['raw_response'], 'workflow')
            except (ValueError, TypeError, KeyError) as error:
                entry.update(valid=False, accepted=False, reason='invalid', error=str(error))
        if candidate is not None:
            entry.update(valid=True, candidate_id=candidate.version_id, proposal=proposal)
            if candidate.version_id in seen:
                entry.update(accepted=False, reason='duplicate')
            else:
                seen.add(candidate.version_id)
                child = store.evaluate(cases, candidate)
                accepted = validation_rank(child['validator']) < validation_rank(current['validator'])
                entry.update(accepted=accepted, reason='validated_improvement' if accepted else 'no_improvement',
                    candidate_validation=child['validator'], case_ids=child['case_ids'])
                if accepted:
                    program, current = candidate, child
        history.append({'candidate_id': entry.get('candidate_id'), 'workflow': proposal['workflow'] if proposal else None,
            'accepted': entry['accepted'], 'reason': entry['reason']})
        write(store.output / 'proposals' / f'{method}-{slot}.json', entry)
        store.event('proposal', method=method, slot=slot, valid=entry['valid'], accepted=entry['accepted'])
        if slot + 1 in checkpoints or slot + 1 == slots:
            snapshot = {'method': method, 'candidate_slots': slot + 1,
                'workflow': program.to_dict(), 'version_id': program.version_id, **current,
                'task_start': starting, 'history_start_length': 0}
            saved[slot + 1] = snapshot
            write(store.output / 'selections' / f'{method}-{slot + 1}.json', snapshot)
    return {'selected': saved[slots], 'checkpoints': saved}


def signal_inactive(program):
    """Declared deployment intervention: remove non-age rules, keep fallback."""
    return replace(program, rules=tuple(r for r in program.rules if r.signal == 'age'))


def main():
    p = argparse.ArgumentParser()
    for key in ('output', 'model', 'actors', 'estimator-weights', 'estimator-manifest'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--trial', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('Real offline model generation requires allocated CUDA')
    started = time.perf_counter()
    predictor = frozen_predictor(args.estimator_weights, args.estimator_manifest)
    actors, actor_info = load_actors(args.actors)
    model_info = verify_open_model(args.model)
    caller = Caller(args.model, 'cuda')
    parameter_versions = tuple(p._version for p in caller.model.parameters())
    store = EventStore(args.output, predictor, actors)
    write(args.output / 'contract.json', contract())
    write(args.output / 'immutable-inputs.json', {'model': model_info, 'actors': actor_info,
        'estimator_sha256': sha(args.estimator_weights), 'job_id': os.environ.get('SLURM_JOB_ID'),
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__})
    seed = 7599 if args.smoke else SEEDS[args.trial]
    cases = specs('smoke' if args.smoke else 'adapt')
    initial = store.evaluate(cases, PatrolProgram())
    adaptations = {name: adapt(store, caller, name, cases,
        seed * 10000 + (9500 if name == 'random' else 8000), 1 if args.smoke else slots,
        checkpoints=(3,) if name == 'summary' and not args.smoke else (), initial=initial)
        for name, slots in SLOTS.items()}
    if args.smoke:
        selections, groups = {name: result['selected'] for name, result in adaptations.items()}, {}
    else:
        selections = {'summary3': adaptations['summary']['checkpoints'][3], 'summary9': adaptations['summary']['selected'],
            'events3': adaptations['events']['selected'], 'redacted3': adaptations['redacted']['selected'],
            'random9': adaptations['random']['selected']}
        write(args.output / 'selections-frozen.json', selections)
        store.event('all_selections_frozen', sha256=sha(args.output / 'selections-frozen.json'))
        store.event('test_begin')
        test, groups = specs('test'), {}
        human = PatrolProgram(rules=(TypedRule('thermal_excess', .05, 'signal'), TypedRule('age', .75, 'return_tour')))
        programs = {'tour': PatrolProgram(), 'actor': PatrolProgram(fallback='ppo'), 'human': human}
        programs.update({name: PatrolProgram.from_dict(value['workflow']) for name, value in selections.items()})
        for name, program in programs.items():
            groups[name] = store.evaluate(test, program)['case_ids']
        event_program = programs['events3']
        groups['events3/no_actor'] = store.evaluate(test, event_program, actor_tool='urgency')['case_ids']
        groups['events3/nominal'] = store.evaluate(test, event_program, estimator='nominal')['case_ids']
        inactive = signal_inactive(event_program)
        groups['events3/rule_signals_inactive'] = store.evaluate(test, inactive)['case_ids']
        write(args.output / 'deployment-interventions.json', {'rule_signals_inactive': {
            'original_program': event_program.to_dict(), 'effective_program': inactive.to_dict(),
            'meaning': 'Only non-age workflow rules removed; actor/estimate/observation permissions unchanged'}})
    store.save_indexes()
    write(args.output / 'test-groups.json', groups)
    if caller.calls != (3 if args.smoke else 15):
        raise RuntimeError('Predeclared call count mismatch')
    for s, model in actors.items():
        if seed_digest(model) != actor_info[str(s)]['parameters_sha256']:
            raise RuntimeError('Frozen actor changed')
    if tuple(p._version for p in caller.model.parameters()) != parameter_versions:
        raise RuntimeError('Language model weights changed')
    write(args.output / 'summary.json', {'trial': args.trial, 'generation_seed': seed, 'smoke': args.smoke,
        'job_id': os.environ.get('SLURM_JOB_ID'), 'unique_cases': len(store.rows), 'unique_processes': len(store.processes),
        'test_group_case_references': sum(map(len, groups.values())), 'actual_model_calls': caller.calls,
        'neural_training': False, 'improver_protocol_updated': False,
        'all_weights_unchanged': True, 'case_evaluation_seconds': store.evaluation_seconds,
        'seconds_including_load_and_verification': time.perf_counter() - started,
        'peak_GPU_bytes': torch.cuda.max_memory_allocated()})
    print('M12C_COMPLETED', args.trial, len(store.rows), 'cases', caller.calls, 'calls', flush=True)


if __name__ == '__main__':
    main()
