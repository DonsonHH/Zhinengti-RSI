"""Read-only reconstruction of M12-F prompts, treatment provenance and charges.

Uses only the standard library. Never imports the campaign, its validators,
tokenizer or models. Run after model search allocations are quiescent.
"""
from __future__ import annotations
import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

KINDS = ('thermal', 'electrical', 'cooling', 'control')
DECISION_FIELDS = ('time_s', 'task_public_id', 'agent', 'features', 'skill_action', 'executed_action',
                   'team_action', 'candidate_finish_s', 'Tour_target_finish_s', 'matched_rule_index', 'check_intent')
MEASUREMENT_FIELDS = ('time_s', 'kind', 'value', 'field', 'unit', 'qualified_role', 'calibrated', 'observed_alarm', 'transition')
SYSTEM = ('Return exactly one JSON object with the requested response_keys. Follow the bounded typed schema. '
          'No text outside JSON. Use only supplied public evidence.')


def need(value, message):
    if not value:
        raise ValueError(message)


def strict(text):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, 'Duplicate JSON key')
            value[key] = item
        return value
    def number(text):
        value = float(text)
        need(math.isfinite(value), 'Nonfinite JSON number')
        return value
    return json.loads(text, object_pairs_hook=pairs, parse_float=number,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON constant')))


def read(path):
    return strict(Path(path).read_text(encoding='utf-8'))


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def skill_id(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def validate_skill(value, space):
    need(isinstance(value, dict) and set(value) == {'schema_version', 'rules', 'default_action'}, 'Closed skill envelope')
    need(value['schema_version'] == 'timing-skill-v1' and value['default_action'] == 'CONTINUE_TOUR', 'Skill version/default')
    need(isinstance(value['rules'], list) and len(value['rules']) <= 3, 'At most three OR branches')
    numeric, boolean = space['numeric_domains'], set(space['boolean_fields'])
    for branch in value['rules']:
        need(isinstance(branch, dict) and set(branch) == {'all'} and isinstance(branch['all'], list)
             and 1 <= len(branch['all']) <= 6, 'One to six AND atoms')
        for atom in branch['all']:
            need(isinstance(atom, dict) and set(atom) == {'field', 'op', 'value'}, 'Closed atom')
            field, op, v = (atom[k] for k in ('field', 'op', 'value'))
            need(type(field) is str and type(op) is str and field in set(numeric) | boolean | {'kind', 'evidence_margin'}, 'Registered field/operator')
            if field == 'evidence_margin':
                need(op == 'positive' and type(v) in (int, float) and v in space['special_margin_coefficients'], 'Margin domain')
            elif op == 'is_missing':
                need(v is None, 'Missingness requires null')
            elif field in boolean:
                need(op == 'eq' and type(v) is bool, 'Literal Boolean atom')
            elif field == 'kind':
                need(op == 'eq' and v in KINDS, 'Profession atom')
            else:
                need(op in ('ge', 'gt', 'le', 'lt') and ((field == 'professional_signal' and v == 'kind_threshold')
                     or type(v) in (int, float) and v in numeric[field]), 'Numeric threshold domain')
    return value


def reply(text, meta, audit):
    value = strict(text)
    key = 'protocol' if meta else 'skill'
    need(isinstance(value, dict) and set(value) == {'diagnosis', key, 'expected_effect'}, 'Closed original reply')
    need(all(type(value[k]) is str and 1 <= len(value[k]) <= 1500 for k in ('diagnosis', 'expected_effect')), 'Bounded explanation')
    if not meta:
        return validate_skill(value[key], audit.space)
    protocol = value[key]
    need(isinstance(protocol, dict) and set(protocol) == {'protocol', 'generator', 'dedup', 'acceptance', 'failure_fallback'}, 'Nested controlled protocol')
    inner = protocol['protocol']
    need(isinstance(inner, dict) and set(inner) == {'diagnostics', 'memory', 'instruction'}, 'Nested improver')
    schema = audit.templates['meta_empty_context_template']['schema']
    tools = inner['diagnostics']
    need(isinstance(tools, list) and 1 <= len(tools) <= 3 and all(type(v) is str for v in tools)
         and len(set(tools)) == len(tools) and set(tools) <= set(schema['diagnostic_choices']), 'Diagnostic allowlist')
    for key, choices in [('memory', 'memory_choices'), ('instruction', 'instruction_choices')]:
        need(inner[key] in schema[choices], 'Protocol ' + key)
    for key in ('generator', 'dedup', 'acceptance', 'failure_fallback'):
        need(protocol[key] in schema[key], 'Protocol enum ' + key)
    return protocol


def project_card(card, space):
    decision, measurement = card['decision'], card['measurement']
    features = decision['features']; fields = set(space['numeric_domains']) | set(space['boolean_fields']) | {'kind'}
    need(isinstance(features, dict) and set(features) <= fields, 'Only registered public features')
    for field, value in features.items():
        if value is None:
            continue
        need(type(value) is bool if field in space['boolean_fields'] else value in KINDS if field == 'kind'
             else type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Typed finite public feature')
    if features.get('direct_available') is False:
        need(all(features.get(k) is None for k in ('direct_age_s', 'direct_age_ratio', 'direct_reading_normal', 'direct_is_qualified')), 'Missing direct reading is not known normal/abnormal')
    need(features.get('direct_reading_normal') is None or features.get('direct_available') is True, 'Known direct normality has a reading')
    need(features.get('professional_available') is not False or features.get('professional_signal') is None, 'Missing professional signal remains null')
    need(features.get('estimate_available') is not False or features.get('estimate_age_s') is None, 'Missing estimate age remains null')
    kind = measurement['kind']
    field, unit, abnormal = {'thermal': ('temperature_c', 'degC', lambda v: v > 65),
        'electrical': ('contact_temperature_c', 'degC', lambda v: v > 60),
        'cooling': ('pod_flow_kg_s', 'kg/s', lambda v: v < 33.6),
        'control': ('controller_sensor_error_c', 'degC', lambda v: abs(v) > 1)}[kind]
    need(features['kind'] == kind and re.fullmatch(kind + r':[0-9]+', decision['task_public_id']), 'Same public profession/task')
    need(measurement['field'] == field and measurement['unit'] == unit and measurement['qualified_role'] == kind
         and measurement['calibrated'] is True, 'Actual qualified reference')
    start, end, value = decision['time_s'], measurement['time_s'], measurement['value']
    need(all(type(v) in (int, float) and math.isfinite(v) for v in (start, end, value)) and end >= start, 'Causal reading times')
    need(abs(card['elapsed_to_actual_measurement_s'] - (end - start)) <= 1e-9, 'Actual elapsed time')
    need(type(measurement['observed_alarm']) is bool and measurement['observed_alarm'] == abnormal(value), 'Reference alarm arithmetic')
    need(measurement['transition'] in (('new_measured_alert', 'persistent_recheck') if abnormal(value)
         else ('normal_check', 'recovery_confirmation')), 'Public alarm transition')
    return {'decision': {k: decision[k] for k in DECISION_FIELDS if k in decision},
            'measurement': {k: measurement[k] for k in MEASUREMENT_FIELDS}, 'elapsed_to_actual_measurement_s': end - start}


class Auditor:
    def __init__(self, root, source):
        self.root, self.source = root.resolve(), source.resolve()
        self.identity_path = root / 'campaign-execution-identity.json'; self.identity = read(self.identity_path)
        need(Path(self.identity['source_root']).resolve() == self.source, 'Actual source tree')
        indexed = {v['path']: v['sha256'] for v in self.identity['artifacts']}
        need(len(indexed) == len(self.identity['artifacts']), 'Unique artifact paths')
        for path, expected in indexed.items():
            need(sha(path) == expected, 'Unchanged bound artifact: ' + path)
        bound = self.identity['bound_inputs']; need(set(bound.values()) <= set(indexed), 'All consumed inputs hash-bound')
        self.study = read(bound['study']); self.space = self.study['skill_space']
        self.templates = read(bound['prompts']); self.manifest = read(bound['source_manifest']); self.cases = read(bound['cases'])
        need(self.identity['baseline'] == self.study['baseline'] == self.manifest['baseline'] == '39d06ce5edb4dc7f7fff18bedc76909350a946d4', 'Frozen study baseline')
        review = read(bound['review']); self.review = review; need(review['approved'] and review['source_archive_SHA256'] == self.manifest['sha256'], 'Approved exact source')
        for name, value in review['reviewed_input_SHA256'].items(): need(sha(bound[name]) == value, 'Reviewed input bytes')
        need(sha(bound['source_archive']) == self.manifest['sha256'], 'Frozen archive bytes')
        for group in ('src', 'scripts', 'tests'):
            for path in (self.source / group).rglob('*.py'):
                need(sha(path) == self.manifest['files'][path.relative_to(self.source).as_posix()]['sha256'], 'Reviewed source member')
        need(self.study['cases_SHA256'] == sha(bound['cases']) and self.study['prompt_templates_SHA256'] == sha(bound['prompts']), 'Study/input links')
        model = read(bound['Qwen_manifest']); need(model['id'] == 'Qwen/Qwen3-8B' and model['revision'] == 'b968826d9c46dd6066d109eabc6255188de91218', 'Fixed model manifest')
        for item in model['files']:
            relative = Path(item['file']); need(not relative.is_absolute() and '..' not in relative.parts, 'Manifest-relative model filename')
            path = Path(bound['Qwen_manifest']).parent / relative
            need(path.stat().st_size == item['bytes'], 'Frozen model file size; trusted model cache symlinks are permitted')
            need((indexed[str(path)] if str(path) in indexed else sha(path)) == item['sha256'], 'Frozen model bytes; hash only, never load tensors')
        self.model_files = len(model['files'])
        self.initial = self.space['initial_skill']; self.incumbent = self.templates['meta_empty_context_template']['current_protocol']
        self.calls = self.slots = self.invalid = self.duplicates = self.exposed = 0; self.groups = {}; self.invalid_reasons = {}; self.independent_errors = {}
        self.professions = set(); self.card_sets = {}; self.record_count = 0; self.jobs = set(); self.input_tokens = self.output_tokens = 0

    def contained(self, path):
        path = Path(path).resolve(); need(path.is_relative_to(self.root), 'Never read an old/private out-of-run record'); return path

    def records_for(self, ledger, partition):
        results = {v['key']: v for v in ledger['results']}; need(len(results) == len(ledger['results']), 'Unique evaluations')
        cohort = self.cases['known_development'] if partition == 'known' else [v for v in self.cases['future'] if v['partition'] == partition]
        expected = sorted(encode(v['specification']) for v in cohort)
        for result in results.values():
            first = self.contained(result['rows'][0]['record_path']).parent
            started = read(first / 'started.json')
            need(sorted(encode(v['specification']) for v in started['cases']) == expected == sorted(encode(v['specification']) for v in result['rows']), 'Only exact frozen own-partition cases')
            need(encode(read(first / 'complete.json')) == encode(result), 'Ledger equals completed evaluation')
            need(result['key'] == digest({k: started[k] for k in ('cases', 'policy', 'context')}), 'Exact source-bound cache key')
            need(started['context']['source_archive_SHA256'] == self.manifest['sha256'], 'Evaluation source')
        return results

    def card_hashes(self, result):
        identity = (result['key'], tuple((v['record_path'], v['record_SHA256']) for v in result['rows']))
        if identity not in self.card_sets:
            values = set()
            for row in result['rows']:
                path = self.contained(row['record_path']); need(sha(path) == row['record_SHA256'], 'Actual own-run record hash')
                raw = strict(gzip.decompress(path.read_bytes()).decode()); self.record_count += 1
                need(raw['specification'] == row['specification'] and encode(raw['policy_spec']) == encode(result['policy']), 'Own policy/cohort provenance')
                need(raw['specification']['phase'] in ('development', 'F2_meta', 'F2_adapt'), 'Only development/adaptation feedback')
                for card in raw['causal_experience_cards']:
                    values.add(digest(project_card(card, self.space)))
                if row['probe_path'] is not None:
                    need(sha(self.contained(row['probe_path'])) == row['probe_SHA256'], 'Frozen probe metadata')
            self.card_sets[identity] = values
        return self.card_sets[identity]

    def context(self, actual, history, protocol, prior, results, initial):
        need(set(actual) == {'initial_skill', 'proposal_history', 'diagnostics', 'instruction', 'experience_cards', 'acceptance_feedback'}, 'Closed prompt context')
        need(encode(actual['initial_skill']) == encode(self.initial), 'Same initial parent in every query')
        need(actual['diagnostics'] == protocol['protocol']['diagnostics'] and actual['instruction'] == protocol['protocol']['instruction'], 'Declared fixed/proposed protocol')
        neutral = [{'skill': v['skill']} if v.get('skill') is not None else {'schema_error': 'previous output was not a valid TimingSkill'} for v in history[-2:]]
        need(encode(actual['proposal_history']) == encode(neutral), 'Only own last two neutral proposals')
        if prior:
            need(actual['experience_cards'] == actual['acceptance_feedback'] == [], 'No experience, selection feedback or selected-parent leak to prior')
            return
        indexed = list(enumerate(history)); memory = protocol['protocol']['memory']
        selected = [] if memory == 'none' else indexed[-2:] if memory == 'recent' else ([v for v in indexed if v[1].get('accepted')][-1:] + [v for v in indexed if not v[1].get('accepted')][-1:])
        feedback = [{'candidate_id': skill_id(v['skill']), 'previous_proposal_offset': len(history) - i, 'accepted': bool(v['accepted'])}
                    for i, v in selected if v.get('skill') is not None and 'accepted' in v]
        need(encode(actual['acceptance_feedback']) == encode(feedback), 'Only declared own-run acceptance bits')
        allowed = set(self.card_hashes(initial))
        for _, item in selected:
            if item.get('evaluation_key') is not None:
                allowed.update(self.card_hashes(results[item['evaluation_key']]))
        need(isinstance(actual['experience_cards'], list) and len(actual['experience_cards']) <= 8, 'Bounded exposed cards')
        for card in actual['experience_cards']:
            projected = project_card(card, self.space)
            need(encode(projected) == encode(card) and digest(card) in allowed, 'Exposed card belongs to permitted completed own-run/initial pool')
            self.professions.add(card['measurement']['kind']); self.exposed += 1

    def call(self, area, seed, history, protocol, prior, results, initial, *, meta=False):
        number = self.call_number; path = self.call_area / 'calls' / f'call-{number:02d}'
        start, end = read(path.with_suffix('.start.json')), read(path.with_suffix('.result.json'))
        need(not path.with_suffix('.failed.json').exists(), 'Successful phase cannot contain a failed generation')
        need(start['seed'] == seed and start['thinking'] is False and type(start['input_tokens']) is int and 0 < start['input_tokens'] <= 6144 and start['output_cap'] == 1024, 'Fixed actual call seed/limits')
        need(start['input_tokens'] == end['input_tokens'] and type(end['output_tokens']) is int and 0 <= end['output_tokens'] <= 1024, 'Receipt token costs')
        prompt = start['prompt']; name = 'meta_empty_context_template' if meta else 'task_empty_context_template'; template = copy.deepcopy(self.templates[name]); key = 'completed_causal_experience' if meta else 'data'
        template[key] = prompt[key]; need(encode(template) == encode(prompt), 'Only context differs from frozen prompt template')
        self.context(prompt[key], history, protocol, prior, results, initial)
        need(end['prompt_SHA256'] == digest(prompt) and start['dropped_last_cards'] == end['dropped_last_cards'], 'Unedited actual exposed prompt')
        rendered = start['rendered_prompt']; body = json.dumps(prompt, ensure_ascii=False)
        prefix = '<|im_start|>system\n' + SYSTEM + '<|im_end|>\n<|im_start|>user\n' + body + '<|im_end|>\n<|im_start|>assistant\n'
        need(rendered.startswith(prefix) and re.fullmatch(r'\s*(?:<think>\s*</think>\s*)?', rendered[len(prefix):]), 'Declared messages and empty nonthinking assistant prefix only')
        need(rendered.count('<|im_start|>system\n') == rendered.count('<|im_start|>user\n') == 1, 'No extra hidden system/user message')
        need(start['reservation']['job_id'] == self.job and start['reservation']['amount'] == 1, 'Every actual generation charged')
        self.calls += 1; self.call_number += 1; self.input_tokens += start['input_tokens']; self.output_tokens += end['output_tokens']
        return end

    def group(self, area, name, summary, seeds, results, *, prior=False, simple=False, call_numbers=None):
        protocol = summary['protocol']; history = summary['history']; entries = summary['entries']; n = len(seeds)
        need(summary['slots'] == len(history) == len(entries) == n and summary['prior_only'] is prior, 'Exact bounded proposal slots')
        need(len(list((area / name).glob('slot-*.attempt.json'))) == len(list((area / name).glob('slot-*.complete.json'))) == n, 'Every slot has one attempt/completion')
        candidates = [v for v in results.values() if encode(v['policy']) == encode(summary['initial_policy'])]
        need(len(candidates) == 1 and encode(summary['initial_policy']['skill']) == encode(self.initial), 'Same evaluated initial skill')
        initial = candidates[0]; seen = {self.dedup(self.initial, protocol['dedup'])}; selected = initial
        need(summary['initial_policy']['executor'] == 'timing-skill-v1' and summary['initial_policy']['ranker'] == 'urgency', 'Same initial executor/ranker')
        for i, seed in enumerate(seeds):
            prefix = area / name / f'slot-{i:02d}'; attempt, entry = read(prefix.with_suffix('.attempt.json')), read(prefix.with_suffix('.complete.json'))
            need(encode(entry) == encode(entries[i]) and attempt['slot'] == entry['slot'] == i and attempt['generation_seed'] == entry['generation_seed'] == seed and attempt['one_opportunity_consumed'], 'Immutable charged slot identity')
            need(encode(attempt['initial_parent']) == encode(self.initial) and attempt['prior_only'] is prior, 'Fixed parent/treatment')
            need(attempt['generator'] == ('random_program' if simple else protocol['generator']), 'Declared generator')
            model = not simple and protocol['generator'] == 'qwen_once'; valid = True
            if model:
                if call_numbers is not None: self.call_number = call_numbers[i]
                generated = self.call(area, seed, history[:i], protocol, prior, results, initial)
                need(encode(entry['generation']) == encode(generated), 'Unedited original generated response')
                try:
                    skill = reply(generated['raw_response'], False, self)
                except (ValueError, TypeError, KeyError, OverflowError) as error:
                    valid = False; category = type(error).__name__ + ': ' + str(error); self.independent_errors[category] = self.independent_errors.get(category, 0) + 1
            else:
                need('generation' not in entry and entry['source'] == 'simple_same_grammar_sampler', 'No hidden model call in random slot')
                skill = validate_skill(entry['skill'], self.space)
            need(entry['valid'] is valid and type(entry['accepted']) is bool and history[i]['accepted'] is entry['accepted'], 'Original parse result and retention bit')
            self.slots += 1
            if not valid:
                need(not entry['accepted'] and history[i]['skill'] is None and entry['reason'] == 'invalid_schema', 'Invalid output consumes slot without repair')
                self.invalid += 1; reason = entry.get('schema_error', 'unspecified invalid reply'); self.invalid_reasons[reason] = self.invalid_reasons.get(reason, 0) + 1
                continue
            need(encode(skill) == encode(entry['skill']) == encode(history[i]['skill']) and entry['candidate_id'] == skill_id(skill), 'Typed skill is the raw proposal')
            key = self.dedup(skill, protocol['dedup']); duplicate = key in seen; seen.add(key)
            need((entry['reason'] == 'duplicate_consumes_slot') is duplicate, 'Fixed duplicate accounting')
            if duplicate:
                need(not entry['accepted'] and history[i]['evaluation_key'] is None, 'Duplicate cannot be retained or resampled'); self.duplicates += 1
            else:
                need(entry['evaluation_key'] == history[i]['evaluation_key'] and encode(results[entry['evaluation_key']]['policy']['skill']) == encode(skill), 'Actual candidate evaluation')
            if entry['accepted']:
                selected = results[entry['evaluation_key']]
        need(encode(selected['policy']) == encode(summary['selected_policy']) and selected['key'] == summary['selected_evaluation_key'] and encode(selected['statistics']) == encode(summary['selected_statistics']), 'Selected policy reconstructed from actual retention sequence')
        need(summary['valid_proposals'] == sum(v['valid'] for v in entries), 'Valid-slot count')
        self.groups[str(area / name)] = {'slots': n, 'valid': sum(v['valid'] for v in entries), 'invalid': sum(not v['valid'] for v in entries), 'duplicates': sum(v['reason'] == 'duplicate_consumes_slot' for v in entries), 'behavior_changes_receipted_not_replayed': sum(v.get('behavior', {}).get('behavior_changed', False) for v in entries)}

    def dedup(self, skill, mode):
        value = copy.deepcopy(skill)
        need(mode in ('exact_version', 'canonical_rule_order'), 'Fixed dedup method')
        if mode == 'canonical_rule_order':
            rules = []
            for branch in value['rules']:
                atoms = []
                for atom in branch['all']:
                    if atom not in atoms: atoms.append(atom)
                if {'all': atoms} not in rules: rules.append({'all': atoms})
            value['rules'] = rules
        return digest(value)

    def allocation(self, area, mode, trial=-1):
        startup, loaded = read(area / 'calls/model-startup.json'), read(area / 'calls/model-loaded.json')
        need(mode in self.review['approved_scopes'], 'Approved actual search mode')
        claim = read(self.root / 'launches' / f'{mode}--trial{trial}--batch-1.json'); self.job = startup['job_id']
        need(self.job not in self.jobs, 'Unique recorded GPU allocation per search'); self.jobs.add(self.job); self.call_number = 0; self.call_area = area
        need(self.job == claim['job_id'] and claim['identity_SHA256'] == sha(self.identity_path), 'One source-bound GPU launch')
        proofs = [v for v in read(self.root / 'launch-proofs.json') if v['mode'] == mode and v['trial'] == trial and v['batch'] == -1]
        need(len(proofs) == 1, 'One normalized held-script proof'); proof = proofs[0]
        need(proof['held_verified'] and proof['job_id'] == self.job and proof['identity_SHA256'] == sha(self.identity_path) and Path(proof['source_root']).resolve() == self.source and Path(proof['run_root']).resolve() == self.root, 'Held job/source/run identity')
        need(sha(self.contained(proof['script_path'])) == proof['script_SHA256'] == self.manifest['files']['cluster/m12f_gpu.sbatch']['sha256'], 'Actual held Slurm bytes equal frozen wrapper')
        need(sha(self.contained(proof['original_submission_path'])) == proof['original_submission_SHA256'] and sha(self.contained(proof['launch_claim_path'])) == proof['launch_claim_SHA256'], 'Original submission and runtime claim hashes')
        submitted = read(proof['original_submission_path']); need(submitted['job_id'] == self.job and submitted['identity_SHA256'] == sha(self.identity_path) and submitted['actual_script_SHA256'] == proof['script_SHA256'], 'Original proof joins normalized proof')
        expected = f'{self.source}/cluster/m12f_gpu.sbatch {self.source} {self.root} {mode} {trial}'
        held = self.contained(proof['held_job_path']).read_text(); need(proof['expected_command'] == expected and expected in held and f'JobId={self.job} ' in held and 'JobState=PENDING' in held and 'Reason=JobHeldUser' in held, 'Correct command verified while held')
        need('V100S' in startup['GPU'] and startup['total_GPU_bytes'] >= 32000000000 and loaded['parameter_tensors'] > 0, 'Frozen model startup class')
        need(loaded['frozen_manifest_SHA256'] == sha(self.identity['bound_inputs']['Qwen_manifest']) and loaded['model_verified']['revision'] == 'b968826d9c46dd6066d109eabc6255188de91218', 'Frozen model revision/files')

    def end_allocation(self, area, count):
        expected = {f'call-{i:02d}.start.json' for i in range(self.call_number)}
        need({p.name for p in (area / 'calls').glob('*.start.json')} == expected and count == self.call_number, 'No extra generation/retry')
        need(len(list((area / 'calls').glob('*.result.json'))) == self.call_number and not list((area / 'calls').glob('*.failed.json')), 'Every successful call returned exactly once')

    def run_phase(self, phase):
        if phase == 'F1':
            area = self.root / 'F1/search'; done = read(area / 'complete.json'); self.allocation(area, 'F1_search'); results = self.records_for(done['evaluation_ledger'], 'known')
            need(done['complete'] and done['identity_SHA256'] == sha(self.identity_path), 'Completed same-source F1 search')
            repeat_gates = []
            for mapping, g in zip(done['mappings'], self.study['F1']['generation_repeats']):
                need(mapping['generation_seed'] == g, 'Fixed generation repeat')
                base = self.call_number
                for name in ('simple', 'prior', 'experienced'):
                    need(encode(mapping[name]['protocol']) == encode(self.incumbent), 'Same fixed improver for F1 controls')
                    indices = [base + 2 * s + (name == 'experienced') for s in range(3)]
                    self.group(area / f'generation-{g}', name, mapping[name], [g * 10000 + (200 if name == 'simple' else 100) + s for s in range(3)], results, prior=name == 'prior', simple=name == 'simple', call_numbers=indices)
                self.call_number = base + 6
                prior, experienced = (self.groups[str(area / f'generation-{g}' / name)] for name in ('prior', 'experienced'))
                repeat_gates.append({'generation_seed': g, 'prior_valid': prior['valid'], 'experienced_valid': experienced['valid'], 'schema_gate': prior['valid'] >= 2 and experienced['valid'] >= 2, 'experienced_behavior_changes_receipted': experienced['behavior_changes_receipted_not_replayed']})
            need(len(done['mappings']) == 3 and self.professions == set(KINDS), 'All repeats and four actually exposed professions')
            self.repeat_gates = repeat_gates
            schema_gate = all(v['schema_gate'] for v in repeat_gates)
            combined = schema_gate and all(v['experienced_behavior_changes_receipted'] >= 1 for v in repeat_gates) and self.professions == set(KINDS)
            need(done['mechanism_gate']['passed'] is combined, 'Registered mechanism gate from reconstructed schema counts and receipted behavior flags')
            self.closed_by_schema_alone = not schema_gate
            self.end_allocation(area, done['model_calls']); need(self.call_number == 18, 'Exactly eighteen F1 calls')
        else:
            for trial, g in enumerate(self.study['F2']['generation_repeats']):
                area = self.root / f'F2/trial-{trial}/adaptation'; done = read(area / 'complete.json'); self.allocation(area, 'F2_adapt', trial)
                need(done['complete'] and done['identity_SHA256'] == sha(self.identity_path) and done['trial'] == trial and done['generation_seed'] == g, 'Completed same-source F2 repeat')
                meta = done['meta']; results = self.records_for(meta['ledger'], 'F2_meta'); old = meta['old']; new = meta['new']
                need(encode(old['protocol']) == encode(meta['incumbent']) == encode(self.incumbent), 'Original meta incumbent')
                self.group(area, 'old-inner', old, [g * 10000 + 1000 + s for s in range(2)], results)
                initial = next(v for v in results.values() if encode(v['policy']) == encode(old['initial_policy']))
                generated = self.call(area, g * 10000 + 2000, old['history'], self.incumbent, False, results, initial, meta=True)
                need(encode(generated) == encode(meta['response']), 'Original meta reply')
                try: candidate = reply(generated['raw_response'], True, self); valid = True
                except (ValueError, TypeError, KeyError, OverflowError): candidate = self.incumbent; valid = False
                need(valid is meta['proposal_valid'] and encode(candidate) == encode(meta['candidate']) == encode(new['protocol']), 'Invalid meta restores exact incumbent without repair')
                need(meta['protocol_changed'] is (encode(candidate) != encode(self.incumbent)), 'Actual distinct protocol flag')
                need(not meta['retained'] or valid and encode(candidate) != encode(self.incumbent), 'No unchanged/invalid meta update')
                need(encode(meta['transferred_protocol']) == encode(candidate if meta['retained'] else self.incumbent) == encode(done['meta3']['protocol']), 'Only retained protocol transfers')
                self.group(area, 'candidate-inner', new, [g * 10000 + 1000 + s for s in range(2)], results)
                need(meta['task_reset'] == {'initial_skill': self.initial, 'history': [], 'candidate_archive': [], 'outcomes': []}, 'Task/history/outcome reset')
                results = self.records_for(done['adapt_ledger'], 'F2_adapt')
                for name, summary, offset, n, prior, simple in [('simple8', done['simple'], 5000, 8, False, True), ('fixed8', done['fixed8'], 3000, 8, False, False), ('prior3', done['prior'], 4000, 3, True, False), ('meta3', done['meta3'], 3000, 3, False, False)]:
                    need(encode(summary['protocol']) == encode(meta['transferred_protocol'] if name == 'meta3' else self.incumbent), 'Fixed/prior protocol or retained-only transfer')
                    self.group(area, name, summary, [g * 10000 + offset + s for s in range(n)], results, prior=prior, simple=simple)
                need(encode(done['fixed3']['entries']) == encode(done['fixed8']['entries'][:3]) and done['fixed3']['slots'] == 3, 'Original fixed-three prefix; no new calls')
                need(self.call_number <= 19, 'Per-repeat call cap'); self.end_allocation(area, done['model_calls'])


def costs(root):
    result = {}
    for kind, initial, cap in [('episodes', 20, 6000), ('model_calls', 0, 75)]:
        import fcntl
        with (root / 'cost' / f'{kind}.lock').open('r') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            counter = read(root / 'cost' / f'{kind}-counter.json'); lines = (root / 'cost' / f'{kind}-reservations.jsonl').read_text().splitlines()
        total = initial; count = 0
        for line in lines:
            row = strict(line); need(type(row['amount']) is int and row['amount'] > 0 and row['first'] == total, 'Append-only charged reservation sequence')
            total += row['amount']; count += 1; need(total <= cap, 'Global charged budget cap')
        need(counter['reserved'] == total and counter.get('F0_charged', 0) == initial, 'Reservation journal equals locked counter')
        result[kind] = {'reserved': total, 'cap': cap, 'reservation_receipts': count, 'includes_F0': initial}
    starts = list((root / 'F1/search/calls').glob('call-*.start.json')) + list((root / 'F2').glob('trial-*/adaptation/calls/call-*.start.json'))
    tokens = {'attempt_starts': len(starts), 'input_tokens': 0, 'known_output_tokens': 0, 'unknown_output_calls': 0, 'output_cost_upper_bound': 0}
    for path in starts:
        start = read(path); prefix = path.with_name(path.name.removesuffix('.start.json')); end = prefix.with_suffix('.result.json'); failed = prefix.with_suffix('.failed.json')
        need(0 < start['input_tokens'] <= 6144 and start['unknown_output_charge_upper_bound'] == 1024 and not(end.exists() and failed.exists()), 'Charged call/token envelope')
        output = read(end).get('output_tokens') if end.exists() else read(failed).get('output_tokens') if failed.exists() else None
        need(output is None or type(output) is int and 0 <= output <= 1024, 'Known or explicitly upper-bounded failed output')
        tokens['input_tokens'] += start['input_tokens']; tokens['known_output_tokens'] += output or 0
        tokens['unknown_output_calls'] += output is None; tokens['output_cost_upper_bound'] += 1024 if output is None else output
    need(len(starts) <= result['model_calls']['reserved'], 'Every actual call has a charged reservation')
    result['token_receipts'] = tokens
    return result


def scheduler(root, jobs, path=None):
    path = path or root / 'resource-sacct.psv'
    if not path.exists(): return {'verified_from_captured_ledger': False, 'reason': 'No captured scheduler ledger; GPU retries/time and CPU allocation totals not certified.'}
    with path.open(encoding='utf-8') as stream: rows = list(csv.DictReader(stream, delimiter='|'))
    by_id = {v['JobID']: v for v in rows}; need(len(by_id) == len(rows), 'Unique allocation rows')
    claims = {read(p)['job_id'] for p in (root / 'launches').glob('*.json') if p.name.startswith(('F1_search--', 'F2_adapt--'))}
    if not claims <= set(by_id): return {'verified_from_captured_ledger': False, 'reason': 'Captured ledger omits a recorded model launch.', 'snapshot_SHA256': sha(path)}
    gpu = [v for v in rows if re.search(r'(?:^|,)gres/gpu=([1-9][0-9]*)', v['AllocTRES'])]
    need({v['JobID'] for v in gpu} == claims and len(gpu) <= 4, 'No extra GPU job row in captured census')
    need(all(re.search(r'(?:^|,)gres/gpu=1(?:,|$)', v['AllocTRES']) for v in gpu), 'One GPU per allocation')
    need(all(by_id[j]['State'] == 'COMPLETED' and by_id[j]['ExitCode'] == '0:0' for j in jobs), 'Audited successful search jobs completed normally')
    seconds = sum(int(v['ElapsedRaw']) for v in gpu); cpu = sum(int(v['ElapsedRaw']) * int(v['AllocCPUS']) / 60 for v in rows)
    need(seconds <= 21600 and cpu <= 2400, 'Captured total GPU/CPU allocation caps')
    return {'verified_from_captured_ledger': True, 'snapshot_SHA256': sha(path), 'GPU_job_rows': len(gpu), 'reported_GPU_seconds': seconds, 'reported_allocated_CPU_minutes': cpu, 'scope': 'Captured ledger joined to launch claims; no live scheduler query or independent same-job requeue census.'}


def closed_f2(root):
    path = root / 'F2-not-executed-decision.json'
    if not path.exists(): return None
    value = read(path)
    need(value['executed'] is False and value['F2_model_calls'] == value['F2_environment_episodes'] == value['F2_physical_processes'] == 0 and value['F2_future_sources_ungenerated'], 'Recorded F2 nonexecution')
    need(value['F1_search_complete_SHA256'] == sha(root / 'F1/search/complete.json') and not value['mechanism_gate']['passed'], 'Closed F2 joins completed mechanism gate')
    need(not list((root / 'launches').glob('F2_*')) and not list((root / 'F2').rglob('*.start.json')) and not list((root / 'F2').rglob('*.attempt.json')), 'No recorded F2 launch/call/episode attempt')
    need(not any(p.is_file() for d in (root / 'future-sources').glob('F2*') for p in d.rglob('*')), 'No F2 physical-generation artifacts')
    return {'executed': False, 'decision_SHA256': sha(path), 'model_calls': 0, 'environment_episodes': 0, 'physical_processes': 0, 'scope': 'Decision plus absence of recorded F2 execution artifacts within this run.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run-root', type=Path, required=True); parser.add_argument('--source-root', type=Path, required=True); parser.add_argument('--phase', choices=('F1', 'F2'), required=True); parser.add_argument('--accounting', type=Path, help='Captured sacct PSV; defaults to run-root/resource-sacct.psv.'); parser.add_argument('--output', type=Path, help='Optional new, versioned audit receipt; never overwritten.'); args = parser.parse_args()
    report = {'passed': False, 'phase': args.phase, 'audit_script_SHA256': sha(Path(__file__)), 'model_calls_in_this_audit': 0, 'environment_episodes_in_this_audit': 0}
    try:
        report['global_charged_costs'] = costs(args.run_root)
        audit = Auditor(args.run_root, args.source_root); audit.run_phase(args.phase)
        report.update(passed=True, phase_search_complete=True, scope='Prompt/treatment, own-run feedback membership, source/model byte hashes and charged-counter reconstruction.', identity_SHA256=sha(audit.identity_path), source_archive_SHA256=audit.manifest['sha256'], model_files_hashed=audit.model_files, model_calls=audit.calls, candidate_slots=audit.slots, invalid_slots=audit.invalid, independent_parse_error_counts=audit.independent_errors, runtime_invalid_reason_counts=audit.invalid_reasons, proposal_groups=audit.groups, duplicate_slots=audit.duplicates, input_tokens_receipted=audit.input_tokens, output_tokens_receipted=audit.output_tokens, exposed_cards_checked=audit.exposed, actually_exposed_professions=sorted(audit.professions), own_feedback_record_hashes_checked=audit.record_count, recorded_GPU_jobs=sorted(audit.jobs), captured_scheduler=scheduler(args.run_root, audit.jobs, args.accounting))
        if args.phase == 'F1': report.update(per_repeat_prompt_gates=audit.repeat_gates, F2_closed_by_independent_schema_counts=audit.closed_by_schema_alone, F2_nonexecution=closed_f2(args.run_root))
        report['not_independently_reconstructed'] = ['Tokenizer rendering/counts beyond declared system/user framing and receipt limits; no tokenizer/model execution.', 'Exact diagnostic card ranking and token-trim ordering; checked exposed-card membership in allowed own-run pools.', 'Numeric acceptance decisions and model tensor versions in memory; handled by raw-result audit/frozen source checks.', 'Random-generator stream replay; checked frozen source, seed, slot and legal output.', 'Scheduler allocation census/GPU elapsed time or unrecorded setup retries; requires captured Slurm ledger.']
    except Exception as error:
        report.update(passed=False, error_type=type(error).__name__, error=str(error), phase_search_complete=False)
    target = args.output or args.run_root / f'{args.phase}-independent-prompts-audit.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print('M12F_PROMPT_AUDIT', args.phase, report['passed'], report.get('model_calls', 0), flush=True)
    if not report['passed']: raise SystemExit(1)


if __name__ == '__main__':
    main()
