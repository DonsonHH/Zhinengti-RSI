"""M12-B: paired, validated improver protocol updates and fresh-task transfer."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .improver_task import PatrolProgram, TypedRule, DOMAINS, ACTIONS, run_program
from .improver_protocol import (Improver, TOOLS, MEMORY, combine_diagnostics,
    program_prompt, protocol_prompt, validation_summary, validation_rank)
from .m12 import (Store, ModelCaller, actor_scorer, load_actors, verify_open_model, sha, write)
from .m11 import frozen_predictor, seed_digest

GENERATION_SEEDS = (7101, 7102, 7103)
DEVELOPMENT = (('healthy', 900), ('coldplate_fouling', 180),
    ('dc_contact_hotspot', 900), ('aux_feeder_loss+cdu_fouling', 180))
TEST = (('healthy', 900), ('coldplate_fouling', 60), ('coldplate_fouling', 180),
    ('coldplate_fouling', 900), ('dc_contact_hotspot', 900), ('it_feeder_loss', 180),
    ('sensor_bias', 900), ('aux_feeder_loss+cdu_fouling', 180))
RANDOM_INSTRUCTIONS = (
    'Use measured professional evidence to choose when an extra check is justified; otherwise return to the protected tour.',
    'Prioritize distinct proposals, inspect previous failures and change one evidence-driven part of the patrol program.',
    'Reduce repeated checking and use history-model disagreement only for thermal tasks.',
    'Compare measurement age and travel cost before selecting a typed observation rule.',
    'Preserve the strong tour baseline unless public measurements justify a targeted detour. Avoid repeating an evaluated program.')


def contract():
    return {'stage': 'M12-B paired improver-update study', 'generation_seeds': list(GENERATION_SEEDS),
        'meta_rounds': 2, 'inner_candidates_each_arm': 2, 'direct_candidates': 13,
        'random_meta_task_candidates': 5, 'model_meta_task_candidates': 3, 'random_search_candidates': 13,
        'fixed3_is_direct13_prefix': True, 'random_meta3_is_random_meta5_prefix': True,
        'maximum_model_calls_per_generation_seed': 39, 'maximum_input_tokens_per_call': 6144,
        'maximum_output_tokens_per_call': 768, 'meta_seeds': [6400, 6401, 6402, 6403],
        'adaptation_seeds': [6500, 6501], 'test_seeds': [6600, 6601],
        'test_physical_conditions': 16, 'test_observation_conditions': 32, 'noise_c': [.2, 1.],
        'fault_start_domain_s': [150, 335, 520, 705], 'new_neural_training': False,
        'model': 'Qwen/Qwen3-8B', 'revision': 'b968826d9c46dd6066d109eabc6255188de91218',
        'actor': 'M11 history-seed0, frozen', 'estimator': 'M10 frozen history-seed1',
        'initial_program': PatrolProgram().to_dict(),
        'validator_change': 'Annotated development evaluation selects programs; prompt only sees public diagnostics plus limited accept/rollback feedback.',
        'transfer': 'Only validated protocol transfers; task program and experience reset.',
        'shared_tail_rule': 'No optional detours once next baseline item cannot finish before1800.',
        'all_test_groups_per_trial': 11, 'all_test_group_case_references_per_trial': 352,
        'scope': 'Synthetic fixed-layout procedural improver updates; not unrestricted self-modifying code or model weight improvement'}


def specification(phase, scenario, duration, seed, noise=.2):
    scenario_code = sum((i + 1) * ord(c) for i, c in enumerate(scenario))
    rng = np.random.default_rng(seed * 7919 + scenario_code + duration)
    start = int(rng.choice((150, 335, 520, 705)))
    return dict(phase=phase, seed=seed, scenario=scenario, duration_s=duration, fault_start_s=start,
        noise_c=noise, estimator='history', control='feedback', resources='provisioned', variant='open')


def specs(phase):
    if phase == 'smoke':
        return [specification('smoke', scenario, duration, 6990) for scenario, duration in
            (('healthy', 900), ('coldplate_fouling', 180))]
    if phase.startswith('meta'):
        round_id = int(phase[-1])
        fixtures = [('coldplate_fouling', 180, 6400), ('dc_contact_hotspot', 900, 6401)] if round_id == 0 else [
            ('sensor_bias', 900, 6402), ('aux_feeder_loss+cdu_fouling', 180, 6403)]
        return [specification(phase, a, b, c) for a, b, c in fixtures]
    if phase == 'adapt':
        return [specification(phase, a, b, seed) for a, b in DEVELOPMENT for seed in (6500, 6501)]
    if phase == 'test':
        return [specification(phase, a, b, seed, noise) for a, b in TEST for seed in (6600, 6601) for noise in (.2, 1.)]
    raise ValueError('Unknown split')


class Caller(ModelCaller):
    def generate(self, user, seed):
        user = json.loads(json.dumps(user))
        system = ('Return one JSON object using exactly the response_keys in the request. '
            'This is offline patrol-program or improver-protocol adaptation with a fixed validator. '
            'Use only supplied public diagnostics and allowed feedback. Preserve models, measurement validity, '
            'coverage, evaluation, grammar, data and budgets. Never emit source code. '
            'diagnosis and expected_effect must each be a short STRING, never a list. ')
        system += ('workflow.rules must contain at most THREE entries; choose a subset, never four rules.'
            if user['task'] == 'propose_patrol_program' else
            'protocol.diagnostics must contain at most THREE tool names. Replace a tool instead of adding a fourth. diagnosis is a separate explanatory STRING.')
        rendered = self.tokenizer.apply_chat_template([{'role': 'system', 'content': system},
            {'role': 'user', 'content': json.dumps(user)}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(rendered, return_tensors='pt').to(self.device)
        if inputs['input_ids'].numel() > 6144:
            raise RuntimeError('Prompt token cap exceeded')
        torch.manual_seed(seed)
        started = time.perf_counter()
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=768, do_sample=True, temperature=.7, top_p=.8, top_k=20)
        self.calls += 1
        return {'prompt': user, 'rendered_prompt': rendered,
            'raw_response': self.tokenizer.decode(ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True),
            'input_tokens': int(inputs['input_ids'].numel()), 'output_tokens': int(ids.shape[1] - inputs['input_ids'].shape[1]),
            'generation_seconds': time.perf_counter() - started}


def parse(raw, key):
    start, end = raw.find('{'), raw.rfind('}')
    if start < 0 or end < start:
        raise ValueError('No JSON object')
    value = json.loads(raw[start:end + 1])
    if set(value) != {'diagnosis', key, 'expected_effect'}:
        raise ValueError('Response envelope mismatch')
    if any(not isinstance(value[k], str) or not 1 <= len(value[k]) <= 500 for k in ('diagnosis', 'expected_effect')):
        raise ValueError('Response text length/type mismatch')
    return (PatrolProgram.from_dict(value[key]) if key == 'workflow' else Improver.from_dict(value[key])), value


def random_program(seed):
    rng = np.random.default_rng(seed)
    rules = []
    for _ in range(int(rng.integers(0, 4))):
        signal = str(rng.choice(tuple(DOMAINS)))
        rules.append(TypedRule(signal, float(rng.choice(DOMAINS[signal])), str(rng.choice(ACTIONS))))
    return PatrolProgram(int(rng.choice((30, 60, 120))), int(rng.choice((0, 30, 60, 120))),
        str(rng.choice(('tour', 'ppo'))), tuple(rules))


def random_protocol(seed):
    rng = np.random.default_rng(seed)
    tools = tuple(str(v) for v in rng.choice(TOOLS, size=int(rng.integers(1, 4)), replace=False))
    return Improver(tools, str(rng.choice(MEMORY)), str(rng.choice(RANDOM_INSTRUCTIONS)))


class TrialStore(Store):
    def __init__(self, output, predictor, actors):
        super().__init__(output, predictor, actors)
        (output / 'calls').mkdir()
        (output / 'meta').mkdir()
        self.views, self.evaluation_seconds = {}, 0.

    def evaluate(self, cases, program, *, estimator='history', actor_tool='ppo'):
        keys = []
        for case in cases:
            case = dict(case, estimator=estimator, actor_seed=0)
            signature = {'specification': case, 'workflow': program.to_dict(), 'deployment': {'actor_tool': actor_tool}}
            key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
            if key not in self.lookup:
                start = time.perf_counter()
                record, env = run_program(case, program, self.predictor, actor_scorer(self.actors[0]), actor_tool)
                self.evaluation_seconds += time.perf_counter() - start
                path = self.output / 'cases' / (key + '.json.gz')
                path.write_bytes(gzip.compress((json.dumps(record) + '\n').encode(), mtime=0))
                from .improver_protocol import diagnostics
                self.views[key] = diagnostics(record)
                process_key = f"{case['scenario']}--{case['duration_s']}--{case['seed']}--{case['fault_start_s']}"
                if process_key not in self.processes:
                    destination = self.output / 'process-traces' / (process_key + '.npz')
                    np.savez_compressed(destination, **env.trace.arrays)
                    self.processes[process_key] = {'path': str(destination.relative_to(self.output)),
                        'sha256': sha(destination), 'bytes': destination.stat().st_size, 'process': env.trace.metadata}
                elif self.processes[process_key]['process']['numeric_digest'] != env.trace.fingerprint:
                    raise RuntimeError('Inspection changed physical process')
                row = signature | {'case': key, 'path': str(path.relative_to(self.output)), 'sha256': sha(path),
                    'bytes': path.stat().st_size, 'version_id': program.version_id, 'process_key': process_key,
                    'metrics': record['metrics'], 'public_selection': record['public_selection'],
                    'public_diagnostics': self.views[key],
                    'event_weight_sum': sum(r['weight'] for r in record['anomaly_intervals']),
                    'weighted_delay_sum': sum(r['weight'] * r['delay_or_censor_s'] for r in record['anomaly_intervals'])}
                self.rows.append(row)
                self.lookup[key] = row
                self.event('case_completed', case=key, phase=case['phase'])
            keys.append(key)
        rows = [self.lookup[k] for k in keys]
        # Reuse the recursive numeric/schema reducer without exposing evaluator metrics.
        def merge(values):
            a = values[0]
            if isinstance(a, dict):
                return {k: merge([v[k] for v in values]) for k in a}
            if a is None or isinstance(a, str):
                if not all(v == a for v in values):
                    raise ValueError('Diagnostic schema drift')
                return a
            return float(np.mean(values))
        return {'validator': validation_summary(rows), 'public': merge([self.views[k] for k in keys]), 'case_ids': keys}


def adapt(store, caller, protocol, tag, cases, seed, slots, *, random=False, checkpoints=(), initial=None):
    program = PatrolProgram()
    current = initial if initial is not None else store.evaluate(cases, program)
    starting = {'version_id': program.version_id, 'workflow': program.to_dict(), **current}
    history, seen, saved = [], {program.version_id}, {}
    for slot in range(slots):
        generated = None
        entry = {'tag': tag, 'slot': slot, 'generation_seed': seed + slot, 'parent_id': program.version_id,
            'protocol': protocol.to_dict(), 'protocol_id': protocol.version_id, 'parent_validation': current['validator']}
        candidate, proposal = None, None
        if random:
            candidate = random_program(seed + slot)
            proposal = {'diagnosis': 'Bounded random program search', 'workflow': candidate.to_dict(), 'expected_effect': 'No model hypothesis'}
        else:
            generated = caller.generate(program_prompt(program, current['public'], history, protocol), seed + slot)
            entry['generation'] = generated
            try:
                candidate, proposal = parse(generated['raw_response'], 'workflow')
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
        write(store.output / 'proposals' / f'{tag}-{slot}.json', entry)
        store.event('proposal', tag=tag, slot=slot, valid=entry['valid'], accepted=entry['accepted'])
        if slot + 1 in checkpoints or slot + 1 == slots:
            snapshot = {'tag': tag, 'candidate_slots': slot + 1, 'protocol': protocol.to_dict(), 'protocol_id': protocol.version_id,
                'workflow': program.to_dict(), 'version_id': program.version_id, **current,
                'task_start': starting, 'history_start_length': 0, 'source': 'random' if random else 'open_model'}
            saved[slot + 1] = snapshot
            write(store.output / 'selections' / f'{tag}-{slot + 1}.json', snapshot)
    return {'selected': saved[slots], 'checkpoints': saved,
        'public_report': {'slots': slots, 'accepted': sum(h['accepted'] for h in history),
            'duplicates': sum(h['reason'] == 'duplicate' for h in history), 'invalid': sum(h['reason'] == 'invalid' for h in history)}}


def meta_update(store, caller, method, generation_seed, *, smoke=False):
    incumbent, public_reports, rounds = Improver(), [], []
    for number in range(1 if smoke else 2):
        entry = {'round': number, 'method': method, 'incumbent': incumbent.to_dict(), 'incumbent_id': incumbent.version_id}
        if method == 'random_meta':
            candidate = random_protocol(generation_seed * 10000 + 3000 + number)
            entry.update(valid=True, candidate=candidate.to_dict(), candidate_id=candidate.version_id)
        else:
            generation = caller.generate(protocol_prompt(incumbent, public_reports), generation_seed * 10000 + 3000 + number)
            entry['generation'] = generation
            try:
                candidate, proposal = parse(generation['raw_response'], 'protocol')
                entry.update(valid=True, proposal=proposal, candidate=candidate.to_dict(), candidate_id=candidate.version_id)
            except (ValueError, TypeError, KeyError) as error:
                candidate = incumbent
                entry.update(valid=False, candidate=incumbent.to_dict(), candidate_id=incumbent.version_id, error=str(error))
        cases = specs('smoke' if smoke else f'meta{number}')
        shared_start = store.evaluate(cases, PatrolProgram())
        pair_seed = generation_seed * 10000 + number * 10
        arms = {}
        for label, protocol in [('incumbent', incumbent), ('candidate', candidate)]:
            arms[label] = adapt(store, caller, protocol, f'{method}-round{number}-{label}', cases,
                pair_seed, 1 if smoke else 2, initial=shared_start)
        old, new = arms['incumbent']['selected']['validator'], arms['candidate']['selected']['validator']
        accepted = entry['valid'] and candidate.version_id != incumbent.version_id and validation_rank(new) < validation_rank(old)
        entry.update(arms=arms, accepted=accepted, retained_id=candidate.version_id if accepted else incumbent.version_id,
            matched_inner_generation_seed=pair_seed, task_program_and_history_reset_for_each_arm=True)
        if accepted:
            incumbent = candidate
        public_reports.append({'round': number, 'protocol_update_accepted': accepted,
            'incumbent_search': arms['incumbent']['public_report'], 'candidate_search': arms['candidate']['public_report']})
        write(store.output / 'meta' / f'{method}-round{number}.json', entry)
        rounds.append(entry)
        store.event('meta_round_completed', method=method, round=number, accepted=accepted, retained_id=incumbent.version_id)
    return incumbent, rounds


def main():
    p = argparse.ArgumentParser()
    for name in ('output', 'model', 'actors', 'estimator-weights', 'estimator-manifest'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--trial', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('Allocated CUDA is required for real open-model generation')
    start = time.perf_counter()
    predictor = frozen_predictor(args.estimator_weights, args.estimator_manifest)
    actors, actor_info = load_actors(args.actors)
    model_info = verify_open_model(args.model)
    caller = Caller(args.model, 'cuda')
    model_versions = tuple(p._version for p in caller.model.parameters())
    store = TrialStore(args.output, predictor, actors)
    write(args.output / 'contract.json', contract())
    write(args.output / 'immutable-inputs.json', {'model': model_info, 'actors': actor_info,
        'estimator_sha256': sha(args.estimator_weights), 'job_id': os.environ.get('SLURM_JOB_ID'),
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__})
    seed = 7199 if args.smoke else GENERATION_SEEDS[args.trial]
    if args.smoke:
        initial = store.evaluate(specs('smoke'), PatrolProgram())
        direct = adapt(store, caller, Improver(), 'smoke-direct', specs('smoke'), seed * 10000 + 8000, 1, initial=initial)
        protocol, rounds = meta_update(store, caller, 'model_meta', seed, smoke=True)
        transferred = adapt(store, caller, protocol, 'smoke-transfer', specs('smoke'), seed * 10000 + 9000, 1, initial=initial)
        selections, groups = {'direct': direct['selected'], 'transfer': transferred['selected']}, {}
    else:
        adaptation = specs('adapt')
        initial = store.evaluate(adaptation, PatrolProgram())
        direct = adapt(store, caller, Improver(), 'direct', adaptation, seed * 10000 + 8000, 13,
            checkpoints=(3,), initial=initial)
        random_search = adapt(store, caller, Improver(), 'random_search', adaptation, seed * 10000 + 9500, 13,
            random=True, initial=initial)
        random_improver, _ = meta_update(store, caller, 'random_meta', seed)
        random_final = adapt(store, caller, random_improver, 'random_meta-transfer', adaptation,
            seed * 10000 + 8000, 5, checkpoints=(3,), initial=initial)
        model_improver, _ = meta_update(store, caller, 'model_meta', seed)
        model_final = adapt(store, caller, model_improver, 'model_meta-transfer', adaptation,
            seed * 10000 + 8000, 3, initial=initial)
        selections = {'direct3': direct['checkpoints'][3], 'direct13': direct['selected'],
            'random_search13': random_search['selected'], 'random_meta3': random_final['checkpoints'][3],
            'random_meta5': random_final['selected'], 'model_meta3': model_final['selected']}
        write(args.output / 'selections-frozen.json', selections)
        store.event('all_selections_frozen', sha256=sha(args.output / 'selections-frozen.json'))
        store.event('test_begin')
        cases, groups = specs('test'), {}
        human = PatrolProgram(rules=(TypedRule('thermal_excess', .05, 'signal'), TypedRule('age', .75, 'return_tour')))
        for name, program in [('tour', PatrolProgram()), ('actor', PatrolProgram(fallback='ppo')), ('human', human)]:
            groups[name] = store.evaluate(cases, program)['case_ids']
        for name, selected in selections.items():
            groups[name] = store.evaluate(cases, PatrolProgram.from_dict(selected['workflow']))['case_ids']
        selected = PatrolProgram.from_dict(selections['model_meta3']['workflow'])
        groups['model_meta3/no_actor'] = store.evaluate(cases, selected, actor_tool='urgency')['case_ids']
        groups['model_meta3/nominal'] = store.evaluate(cases, selected, estimator='nominal')['case_ids']
    store.save_indexes()
    write(args.output / 'test-groups.json', groups)
    for s, model in actors.items():
        if seed_digest(model) != actor_info[str(s)]['parameters_sha256']:
            raise RuntimeError('Frozen actor changed')
    if tuple(p._version for p in caller.model.parameters()) != model_versions:
        raise RuntimeError('Language-model weights changed')
    if not args.smoke and caller.calls != 39:
        raise RuntimeError('Predeclared model-call envelope mismatch')
    write(args.output / 'summary.json', {'trial': args.trial, 'generation_seed': seed, 'smoke': args.smoke,
        'job_id': os.environ.get('SLURM_JOB_ID'), 'unique_cases': len(store.rows), 'unique_processes': len(store.processes),
        'test_group_case_references': sum(map(len, groups.values())), 'actual_model_calls': caller.calls,
        'model_weights_and_all_actors_unchanged': True, 'new_neural_training': False,
        'seconds_including_input_verification_and_model_load': time.perf_counter() - start,
        'actual_case_evaluation_seconds': store.evaluation_seconds, 'peak_GPU_bytes': torch.cuda.max_memory_allocated()})
    print('M12B_COMPLETED', args.trial, 'cases', len(store.rows), 'calls', caller.calls, flush=True)


if __name__ == '__main__':
    main()
