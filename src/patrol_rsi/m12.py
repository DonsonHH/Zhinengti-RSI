"""M12-A: public-evidence workflow adaptation on the frozen facility backbone."""
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

from .facility_workflow import (COOLDOWN, RANKERS, REFRESH, ROLES, SIGNALS, THRESHOLDS,
    FacilityWorkflow, Rule, run_workflow, selection_rank)
from .inspection_model import TaskPolicy
from .m11 import frozen_predictor, seed_digest
from .m4 import ModelCaller, json_object

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('random_search', 'llm_no_memory', 'llm_memory')
GENERATION_SEEDS = (6101, 6102, 6103)
DEV_FIXTURES = (('healthy', 900), ('coldplate_fouling', 180),
                ('dc_contact_hotspot', 900), ('aux_feeder_loss+cdu_fouling', 180))
TEST_FIXTURES = (('healthy', 900), ('coldplate_fouling', 60), ('coldplate_fouling', 180),
    ('coldplate_fouling', 900), ('dc_contact_hotspot', 900), ('it_feeder_loss', 180),
    ('sensor_bias', 900), ('aux_feeder_loss+cdu_fouling', 180))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def contract():
    return {'stage': 'M12-A frozen-improver integration pilot', 'new_neural_training': False,
        'recursive_improver_updated': False, 'generation_seeds': list(GENERATION_SEEDS),
        'candidate_slots_per_method_seed': 3, 'methods': list(METHODS),
        'development': {'seeds': [6201, 6202], 'fixtures': [list(v) for v in DEV_FIXTURES], 'cases': 8},
        'test': {'seeds': [6300, 6301], 'fixtures': [list(v) for v in TEST_FIXTURES], 'cases': 16},
        'actor_for_adaptation': 'Existing M11 history-seed0', 'actor_transfer_seeds': [1, 2],
        'same_grammar_and_candidate_evaluation_budget': True, 'random_search_uses_no_LLM': True,
        'total_GPU_cost_equal_claimed': False, 'input_token_cap': 6144, 'output_token_cap': 768,
        'model_id': 'Qwen/Qwen3-8B', 'model_revision': 'b968826d9c46dd6066d109eabc6255188de91218',
        'model_precision': 'float16', 'model_generation': {'temperature': .7, 'top_p': .8, 'top_k': 20, 'thinking': False},
        'scope': 'Known static128rack simulation; original paper adapted to temperatureCVAE and high-levelPPO; no field/optimality/recursive-improvement claim',
        'selection': 'Only public valid-measurement outcomes and event-time ages; hidden intervals and test data excluded',
        'test_order': 'All candidate selections frozen before test execution',
        'predeclared_demo': {'seed': 6301, 'scenario': 'coldplate_fouling', 'duration_s': 180,
            'groups': ['fixed_actor', 'human_risk_rules', 'random_search/6101', 'llm_memory/6101']}}


def case_specs(phase):
    if phase == 'smoke':
        fixtures, seeds = (('healthy', 900), ('coldplate_fouling', 180)), (6200,)
    elif phase == 'development':
        fixtures, seeds = DEV_FIXTURES, (6201, 6202)
    elif phase == 'test':
        fixtures, seeds = TEST_FIXTURES, (6300, 6301)
    else:
        raise ValueError('Unknown split')
    return [dict(phase=phase, seed=seed, scenario=scenario, duration_s=duration,
        control='feedback', resources='provisioned', variant='open', estimator='history')
        for scenario, duration in fixtures for seed in seeds]


def load_actors(folder):
    manifest = json.loads((ROOT / 'research/2026-10-27-m11-scheduling/evidence/policy-weight-retrieval.json').read_text())
    models, fingerprints = {}, {}
    for seed in (0, 1, 2):
        relative = f'history-seed-{seed}/selected.pt'
        path = folder / relative
        if sha(path) != manifest['files'][relative]['sha256']:
            raise RuntimeError('Previously delivered actor bytes changed')
        pack = torch.load(path, map_location='cpu', weights_only=True)
        if pack['estimator'] != 'history' or pack['seed'] != seed:
            raise RuntimeError('Actor identity mismatch')
        model = TaskPolicy().eval()
        model.load_state_dict(pack['state_dict'])
        models[seed] = model
        fingerprints[str(seed)] = {'path': str(path), 'sha256': sha(path), 'parameters_sha256': seed_digest(model)}
    return models, fingerprints


def actor_scorer(model):
    def score(features):
        with torch.inference_mode():
            return model.actor(torch.as_tensor(features)).squeeze(-1).numpy()
    return score


def verify_open_model(folder):
    manifest = json.loads((folder / 'verified_manifest.json').read_text())
    if manifest['id'] != contract()['model_id'] or manifest['revision'] != contract()['model_revision']:
        raise RuntimeError('Open model revision changed')
    for row in manifest['files']:
        p = folder / row['file']
        digest = hashlib.sha256()
        with p.open('rb') as f:
            while data := f.read(8 * 1024 * 1024):
                digest.update(data)
        if p.stat().st_size != row['bytes'] or digest.hexdigest() != row['sha256']:
            raise RuntimeError('Open model file hash mismatch: ' + p.name)
    return {'id': manifest['id'], 'revision': manifest['revision'], 'files_verified': len(manifest['files']),
        'manifest_sha256': sha(folder / 'verified_manifest.json')}


def summarize_public(rows):
    average_keys = ('coverage', 'first_alert_tasks', 'first_alert_task_weight', 'weighted_age_ratio', 'travel_m',
        'service_s', 'estimator_calls', 'dispatches', 'extra_checks', 'rule_dispatches', 'ppo_dispatches',
        'forced_single_candidate', 'different_from_actor')
    result = {key: float(np.mean([r[key] for r in rows])) for key in average_keys}
    result['violating_tasks'] = sum(r['violating_tasks'] for r in rows)
    result['measurement_transitions'] = dict(sum((Counter(r['measurement_transitions']) for r in rows), Counter()))
    result['cases'] = len(rows)
    return result


def random_workflow(seed):
    rng = np.random.default_rng(seed)
    rules = []
    for _ in range(int(rng.integers(0, 4))):
        signal = str(rng.choice(SIGNALS))
        rules.append(Rule(str(rng.choice(ROLES)), signal, float(rng.choice(THRESHOLDS[signal])), str(rng.choice(RANKERS))))
    return FacilityWorkflow(int(rng.choice(REFRESH)), int(rng.choice(COOLDOWN)), tuple(rules))


def prompt(parent, score, history, *, memory):
    # Explicit allowlist: this object never receives hidden metrics/specifications.
    return {'task': 'Propose one executable facility-inspection workflow from public development evidence.',
        'current_workflow': parent.to_dict(), 'public_development_summary': score,
        'recent_candidates': history[-2:] if memory else [],
        'grammar': {'top_level': ['diagnosis', 'workflow', 'expected_effect'],
            'text_fields': 'diagnosis and expected_effect are strings of at most600 characters',
            'workflow': {'refresh_s': list(REFRESH), 'cooldown_s': list(COOLDOWN),
                'rules': '0..3 ordered rules; each has role,signal,threshold,ranker'},
            'rule_keys': ['role', 'signal', 'threshold', 'ranker'],
            'rule_domains': {'role': list(ROLES), 'signal': list(SIGNALS),
                'threshold': 'ONE NUMBER, not an object; choose from allowed_thresholds_by_signal for this signal',
                'ranker': list(RANKERS)},
            'allowed_thresholds_by_signal': THRESHOLDS},
        'semantics': {'refresh_s': 'History temperatureCVAE inference period; causal feature frames still every30s.',
            'cooldown_s': 'Only extra checks; never prevents the next protected tour obligation.',
            'risk': 'Public model or measured severity feature, not hidden fault truth or calibrated probability.',
            'uncertainty': 'Absolute history versus nominal model disagreement /20, not calibrated uncertainty.',
            'age': 'Public time since task completion divided by task deadline.',
            'ordered_rules': 'For the first role-matching rule with qualifying candidates, use its ranker on those candidates.',
            'rankers': 'risk or disagreement times public asset weight divided by ETA; urgency is age plus ETA ratio.',
            'fallback': 'Frozen trainedPPO scores all remaining candidates if no rule triggers.',
            'protection': 'Complete-return gate, two extra checks per owned round, qualification and physical service are immutable.'},
        'selection': 'Minimize violating tasks; maximize coverage; maximize weight of distinct first measured-alert tasks; minimize public weighted age; minimize travel. Lexicographic strict improvement. Persistent rechecks add no discovery reward.',
        'rules': ['Return JSON only. Exact grammar; no code, new fields, labels, equipment IDs or metric changes.',
            'Weights, physical processes, measurement validity, interpreter, budgets and hidden evaluator stay fixed.',
            'Changing a workflow is task-level adaptation; the improver itself is fixed in this pilot.']}


def parse_proposal(raw):
    value = json_object(raw)
    if set(value) != {'diagnosis', 'workflow', 'expected_effect'}:
        raise ValueError('Proposal envelope keys mismatch')
    if any(not isinstance(value[k], str) or not 1 <= len(value[k]) <= 600 for k in ('diagnosis', 'expected_effect')):
        raise ValueError('Proposal text size/type mismatch')
    return FacilityWorkflow.from_dict(value['workflow']), value


class FacilityCaller(ModelCaller):
    """Reuse the verified weight loader, not the old grid-task prompt router."""
    def generate(self, user, seed):
        user = json.loads(json.dumps(user))
        system = ('Generate one bounded JSON facility workflow. Read only the supplied public development evidence. '
            'Do not change frozen models, measurement validity, coverage protection, evaluation, data or budgets. '
            'Return diagnosis, workflow and expected_effect using the exact supplied grammar.')
        rendered = self.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(user)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(rendered, return_tensors='pt').to(self.device)
        if inputs['input_ids'].numel() > contract()['input_token_cap']:
            raise RuntimeError('Facility prompt token budget exceeded')
        torch.manual_seed(seed)
        start = time.perf_counter()
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=contract()['output_token_cap'],
                do_sample=True, temperature=.7, top_p=.8, top_k=20)
        answer = self.tokenizer.decode(ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        self.calls += 1
        return {'prompt': user, 'rendered_prompt': rendered, 'raw_response': answer,
            'input_tokens': int(inputs['input_ids'].numel()),
            'output_tokens': int(ids.shape[1] - inputs['input_ids'].shape[1]),
            'generation_seconds': time.perf_counter() - start}


class Store:
    def __init__(self, output, predictor, actors):
        self.output, self.predictor, self.actors = output, predictor, actors
        self.rows, self.lookup, self.processes, self.journal = [], {}, {}, []
        output.mkdir(parents=True, exist_ok=False)
        for name in ('cases', 'process-traces', 'proposals', 'selections'):
            (output / name).mkdir()

    def event(self, kind, **values):
        row = {'sequence': len(self.journal), 'event': kind, **values}
        self.journal.append(row)
        with (self.output / 'journal.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')

    def evaluate(self, specs, program, *, actor_seed=0, estimator='history', **deployment):
        keys, public = [], []
        deployment = {'control': 'workflow', 'actor_fallback': 'ppo', 'signals_permuted': False, **deployment}
        for spec in specs:
            spec = dict(spec, estimator=estimator, actor_seed=actor_seed)
            signature = {'specification': spec, 'workflow': program.to_dict(), 'deployment': deployment}
            key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
            if key not in self.lookup:
                record, env = run_workflow(spec, program, self.predictor, actor_scorer(self.actors[actor_seed]), **deployment)
                path = self.output / 'cases' / (key + '.json.gz')
                path.write_bytes(gzip.compress((json.dumps(record) + '\n').encode(), mtime=0))
                process_key = f"{spec['scenario']}--{spec['duration_s']}--{spec['seed']}"
                if process_key not in self.processes:
                    p = self.output / 'process-traces' / (process_key + '.npz')
                    np.savez_compressed(p, **env.trace.arrays)
                    self.processes[process_key] = {'path': str(p.relative_to(self.output)), 'sha256': sha(p),
                        'bytes': p.stat().st_size, 'process': env.trace.metadata}
                elif self.processes[process_key]['process']['numeric_digest'] != env.trace.fingerprint:
                    raise RuntimeError('Workflow changed exogenous physical process')
                row = signature | {'case': key, 'path': str(path.relative_to(self.output)), 'sha256': sha(path),
                    'bytes': path.stat().st_size, 'version_id': program.version_id, 'process_key': process_key,
                    'metrics': record['metrics'], 'public_selection': record['public_selection'],
                    'event_weight_sum': sum(r['weight'] for r in record['anomaly_intervals']),
                    'weighted_delay_sum': sum(r['weight'] * r['delay_or_censor_s'] for r in record['anomaly_intervals'])}
                self.rows.append(row)
                self.lookup[key] = row
                self.event('case_completed', case=key, phase=spec['phase'])
            keys.append(key)
            public.append(self.lookup[key]['public_selection'])
        return summarize_public(public), keys

    def save_indexes(self):
        write(self.output / 'case-index.json', self.rows)
        write(self.output / 'process-index.json', self.processes)


def adapt(store, caller, method, generation_seed, specs, *, slots=3):
    parent = FacilityWorkflow()
    score, keys = store.evaluate(specs, parent)
    initial = {'workflow': parent.to_dict(), 'public_summary': score, 'case_ids': keys}
    history, seen = [], {parent.version_id}
    for slot in range(slots):
        seed = generation_seed * 100 + slot
        entry = {'method': method, 'generation_seed': generation_seed, 'slot': slot,
            'parent_id': parent.version_id, 'parent_public_summary': score, 'candidate_seed': seed}
        if method == 'random_search':
            candidate = random_workflow(seed)
            proposal = {'diagnosis': 'Bounded random search control', 'workflow': candidate.to_dict(), 'expected_effect': 'No model-generated hypothesis'}
            entry['source'] = 'random_generator'
        else:
            entry['source'] = 'Qwen3-8B'
            entry['generation'] = caller.generate(prompt(parent, score, history, memory=method == 'llm_memory'), seed)
            try:
                candidate, proposal = parse_proposal(entry['generation']['raw_response'])
            except (ValueError, TypeError, KeyError) as error:
                entry.update(valid=False, accepted=False, reason=str(error))
                write(store.output / 'proposals' / f'{method}-{generation_seed}-{slot}.json', entry)
                store.event('proposal', method=method, generation_seed=generation_seed, slot=slot, accepted=False, valid=False)
                history.append({'valid': False, 'accepted': False, 'reason': str(error)})
                continue
        entry.update(valid=True, proposal=proposal, candidate_id=candidate.version_id)
        if candidate.version_id in seen:
            entry.update(accepted=False, reason='duplicate_consumes_slot')
            history.append({'workflow': candidate.to_dict(), 'accepted': False, 'reason': entry['reason']})
        else:
            seen.add(candidate.version_id)
            candidate_score, candidate_keys = store.evaluate(specs, candidate)
            accepted = selection_rank(candidate_score) < selection_rank(score)
            entry.update(candidate_public_summary=candidate_score, case_ids=candidate_keys, accepted=accepted,
                reason='strict_public_development_improvement' if accepted else 'no_strict_public_improvement')
            history.append({'workflow': candidate.to_dict(), 'public_summary': candidate_score, 'accepted': accepted})
            if accepted:
                parent, score, keys = candidate, candidate_score, candidate_keys
        write(store.output / 'proposals' / f'{method}-{generation_seed}-{slot}.json', entry)
        store.event('proposal', method=method, generation_seed=generation_seed, slot=slot,
            valid=True, accepted=entry['accepted'], version_id=candidate.version_id)
        print('M12_PROPOSAL', method, generation_seed, slot, entry['reason'], flush=True)
    selected = {'method': method, 'generation_seed': generation_seed, 'workflow': parent.to_dict(),
        'version_id': parent.version_id, 'public_summary': score, 'case_ids': keys, 'initial': initial,
        'candidate_slots': slots, 'new_neural_training': False, 'improver_updated': False}
    write(store.output / 'selections' / f'{method}-{generation_seed}.json', selected)
    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=('smoke', 'pilot'), required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--actors', type=Path, required=True)
    p.add_argument('--estimator-weights', type=Path, required=True)
    p.add_argument('--estimator-manifest', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('Real open-model generation requires allocated CUDA')
    predictor = frozen_predictor(args.estimator_weights, args.estimator_manifest)
    actors, actor_info = load_actors(args.actors)
    model_info = verify_open_model(args.model)
    caller = FacilityCaller(args.model, 'cuda')
    model_versions = tuple(p._version for p in caller.model.parameters())
    store = Store(args.output, predictor, actors)
    started = time.perf_counter()
    write(args.output / 'contract.json', contract())
    write(args.output / 'immutable-inputs.json', {'open_model': model_info, 'actors': actor_info,
        'estimator_sha256': sha(args.estimator_weights), 'job_id': os.environ.get('SLURM_JOB_ID'),
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__})
    baseline = FacilityWorkflow()
    human = FacilityWorkflow(rules=(Rule('all', 'risk', .05, 'risk'),))
    if args.phase == 'smoke':
        specs = case_specs('smoke')
        for name, program, deployment in [('fixed_actor', baseline, {}), ('cycle', baseline, {'control': 'cycle'}),
                ('human_risk_rules', human, {})]:
            _, keys = store.evaluate(specs, program, **deployment)
            store.event('smoke_control', group=name, case_ids=keys)
        selected = adapt(store, caller, 'llm_memory', 6199, specs, slots=1)
        groups = {'smoke_selected': selected['case_ids']}
        selections = [selected]
    else:
        dev = case_specs('development')
        selections = [adapt(store, caller, method, seed, dev) for method in METHODS for seed in GENERATION_SEEDS]
        write(args.output / 'selections-frozen.json', selections)
        store.event('all_selections_frozen', sha256=sha(args.output / 'selections-frozen.json'))
        store.event('test_begin')
        test = case_specs('test')
        groups = {}
        for name, program, deployment in [('fixed_actor', baseline, {}), ('cycle', baseline, {'control': 'cycle'}),
                ('human_risk_rules', human, {})]:
            _, groups[name] = store.evaluate(test, program, **deployment)
        for selected in selections:
            program = FacilityWorkflow.from_dict(selected['workflow'])
            name = f"{selected['method']}/{selected['generation_seed']}"
            _, groups[name] = store.evaluate(test, program)
            if selected['method'] == 'llm_memory':
                for label, settings in [('nominal_estimator', {'estimator': 'nominal'}),
                    ('heuristic_fallback', {'actor_fallback': 'urgency'}),
                    ('workflow_signals_permuted', {'signals_permuted': True}),
                    ('actor_transfer1', {'actor_seed': 1}), ('actor_transfer2', {'actor_seed': 2})]:
                    _, groups[name + '/' + label] = store.evaluate(test, program, **settings)
        for seed in (1, 2):
            _, groups[f'fixed_actor/actor_transfer{seed}'] = store.evaluate(test, baseline, actor_seed=seed)
    store.save_indexes()
    write(args.output / 'test-groups.json', groups)
    for seed, model in actors.items():
        if seed_digest(model) != actor_info[str(seed)]['parameters_sha256']:
            raise RuntimeError('Frozen actor parameters updated')
    if tuple(p._version for p in caller.model.parameters()) != model_versions:
        raise RuntimeError('Language-model parameters updated')
    write(args.output / 'summary.json', {'phase': args.phase, 'job_id': os.environ.get('SLURM_JOB_ID'),
        'unique_cases': len(store.rows), 'unique_processes': len(store.processes),
        'group_case_references': sum(len(v) for v in groups.values()), 'model_calls': caller.calls,
        'selected_workflows': len(selections), 'new_neural_training': False, 'improver_updated': False,
        'all_frozen_actor_and_language_model_versions_unchanged': True,
        'seconds': time.perf_counter() - started, 'peak_GPU_bytes': torch.cuda.max_memory_allocated()})
    print('M12_COMPLETED', args.phase, 'unique_cases', len(store.rows), 'calls', caller.calls, flush=True)


if __name__ == '__main__':
    main()
