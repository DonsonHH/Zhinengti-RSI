"""Read-only G0/G1a retained-result audit; no project, neural or physics execution."""
import argparse, collections, copy, csv, datetime, gzip, hashlib, importlib.util, json, math, re, shlex, sys, types
from pathlib import Path
sys.dont_write_bytecode = True

KERNEL_SHA = '5cb08c9f45726e36965915b83565e94ab3ab2265d7ffccefa3a56e1f8db7a9f3'
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8388608), b''): h.update(b)
    return h.hexdigest()
def need(ok, message):
    if not ok: raise ValueError(message)

class Audit:
    def __init__(self, args):
        self.a = args; self.root = args.run_root.resolve(); self.source = args.source_root.resolve(); self.inputs = {}; self.parents = {}; self.jobs = set(); self.modes = {}; self.cards = self.raws = 0
        p = args.f_audit_kernel or self.source / 'scripts/audit_m12f_actual.py'; need(sha(p) == KERNEL_SHA, 'Wrong frozen standalone F audit kernel')
        spec = importlib.util.spec_from_file_location('f_audit_kernel', p); self.k = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.k); self.inputs[str(p.resolve())] = KERNEL_SHA
    def read(self, p):
        p = Path(p).resolve(); b = p.read_bytes(); h = hashlib.sha256(b).hexdigest(); need(str(p) not in self.inputs or self.inputs[str(p)] == h, 'Input changed during audit'); self.inputs[str(p)] = h
        return self.k.strict_json((gzip.decompress(b) if p.suffix == '.gz' else b).decode('utf-8'))
    def local(self, p):
        p = Path(p).resolve(); need(p.is_relative_to(self.root), 'Result path outside G run root'); return p
    def same(self, a, b, message): need(self.k.equal(a, b), message)
    def aggregate(self, rows):
        return self.k.Audit.aggregate(types.SimpleNamespace(raw=lambda *a: None), rows, [v['specification'] for v in self.cases], self.cohort, None)
    def metadata(self):
        self.identity = self.read(self.root / 'G-execution-identity.json'); self.ish = sha(self.root / 'G-execution-identity.json'); b = self.identity['bound_inputs']; self.bound = b
        need(Path(self.identity['source_root']).resolve() == self.source and Path(self.identity['run_root']).resolve() == self.root, 'Foreign source/run identity')
        for v in self.identity['artifacts']: need(sha(v['path']) == v['sha256'], 'Changed bound artifact: ' + v['path'])
        manifest = self.read(b['source_manifest']); need(sha(b['source_archive']) == manifest['sha256'], 'Archive identity')
        for n, v in manifest['files'].items(): need(sha(self.source / n) == v['sha256'], 'Changed frozen source: ' + n)
        review = self.read(b['review']); need(review['approved'] and 'G0' in review['approved_scopes'] and review['source_archive_SHA256'] == manifest['sha256'], 'Missing G0 source approval')
        for n, h in review['reviewed_input_SHA256'].items(): need(sha(b[n]) == h, 'Changed reviewed input')
        gate = self.read(b['gate']); need(gate['success'] and gate['tests'] >= 29 and not (gate['errors'] or gate['failures'] or gate['skipped']) and Path(gate['source_root']).resolve() == self.source and gate['runtime_imports_passed'], 'Incomplete targeted gate')
        for n, h in gate['test_source_SHA256'].items(): need(sha(self.source / n) == h, 'Changed gate test')
        self.study = self.read(b['study']); cat = self.read(b['candidates']); self.candidates = cat['policies']; self.parent = cat['initial_skill']; self.cases = self.read(b['cases'])['cases']; self.prompts = self.read(b['prompts'])
        for n in ('candidates', 'cases', 'prompts'): need(sha(b[n]) == self.study[n + '_SHA256'], 'Study input join')
        need(self.identity['baseline'] == self.study['baseline'] == cat['baseline'] == 'a5ab5880e33e3df3cd9aeffc3a90a63c40273c41', 'Wrong baseline')
        need(len(self.cases) == 20 and len({v['case_id'] for v in self.cases}) == 20 and len(self.candidates) == 24 and [v['slot'] for v in self.candidates] == list(range(1, 25)), 'Frozen slots/cases')
        self.cohort = {self.k.encoded(v['specification']): v for v in self.cases}; need(len(self.cohort) == 20 and all(v['specification']['phase'] == 'development' and v['specification']['seed'] in (33000, 33001) for v in self.cases), 'Known development only')
        for n in ('F_manifest', 'F_search_complete'): need(sha(b[n]) == self.study['reference_authority'][n + '_SHA256'], 'Historical reference authority')
        refs = self.read(b['F_search_complete'])['evaluation_ledger']['results']; policy = {'executor': 'timing-skill-v1', 'ranker': 'urgency', 'skill': self.parent}; matches = [v for v in refs if self.k.encoded(v['policy']) == self.k.encoded(policy)]; need(len(matches) == 1, 'Exactly one historical parent')
        self.initial = matches[0]; self.old = {self.k.encoded(v['specification']): v for v in self.initial['rows']}; self.same(self.aggregate(self.initial['rows']), self.initial['statistics'], 'Historical macro arithmetic')
        need(not (self.root / 'G-STOP.json').exists() and all(not (self.root / n).exists() for n in ('G1b', 'G2', 'future-sources')), 'Stopped or unauthorized phase entered')
    def launch(self, proof, claim, wrapper, suffix):
        job = str(claim['job_id']); need(str(proof['job_id']) == job and proof['held_verified'] and claim['identity_SHA256'] == self.ish, 'Launch identity/job join')
        need(sha(self.local(proof['script_path'])) == proof['script_SHA256'] == sha(self.source / 'cluster' / wrapper), 'Actual held wrapper SHA')
        command = [str(self.source / 'cluster' / wrapper), str(self.source), str(self.root)] + suffix; held = self.local(proof['held_job_path']).read_text()
        need(shlex.split(proof['expected_command']) == command and 'Command=' + proof['expected_command'] in held and 'JobId=' + job in held and ('JobHeldUser' in held or 'JobHeldAdmin' in held), 'Held launch command/arguments'); self.jobs.add(job); self.modes[job] = wrapper
    def feedback(self, raw, candidate, events):
        skill = candidate['policy']['skill']; cards = raw['causal_experience_cards']; diffs = raw['g_difference_cards']; need(len(cards) == len(diffs), 'Omitted difference card'); seen = set(); expected_run = self.k.digest({'policy': candidate['policy'], 'observed_packet_sequence': raw['public_telemetry_sequence_SHA256']}); parent_id, child_id = self.k.digest(self.parent), self.k.digest(skill)
        by_task = collections.defaultdict(list)
        for (j, _), event in events.items(): by_task[j].append(event)
        for c, g in zip(cards, diffs):
            d, m = c['decision'], c['measurement']; key = d['task'], m['time_s']; e = events.get(key); need(e is not None and key not in seen and d['time_s'] < m['time_s'], 'Noncausal/duplicate feedback'); seen.add(key)
            kind, index = self.k.TASKS[d['task']]; field = self.k.FIELDS[self.k.KINDS.index(kind)]; f = d['features']; past = [v for v in by_task[d['task']] if v['time_s'] <= d['time_s']]
            need(m['kind'] == m['qualified_role'] == e['role'] == kind and m['field'] == field and m['unit'] == ('kg/s' if kind == 'cooling' else 'degC') and m['calibrated'] is True and m['value'] == e['measurement'][field] and m['observed_alarm'] == e['alarm'] and m['transition'] == e['measurement_transition'], 'Feedback/actual event mismatch')
            need(d['task_public_id'] == f'{kind}:{index}' and f['already_alarm'] == (past[-1]['alarm'] if past else False), 'Noncausal public alarm/task')
            if not f['direct_available']: need(all(f[n] is None for n in ('direct_age_s', 'direct_age_ratio', 'direct_reading_normal', 'direct_is_qualified')), 'Missing treated as normal')
            if not f['professional_available']: need(f['professional_signal'] is None, 'Missing professional signal treated as zero')
            if f['direct_is_qualified'] is True: need(past and self.k.equal(d['time_s'] - f['direct_age_s'], past[-1]['time_s']), 'Qualified reading timestamp')
            pa, ca = ('CHECK' if self.k.action(s, f) else 'CONTINUE_TOUR' for s in (self.parent, skill)); need(d['skill_action'] == ca and g['same_input_parent_admission'] == pa and g['same_input_child_admission'] == ca and g['admission_differs'] == (pa != ca), 'Independent admission mismatch')
            self.same(g['decision'], {n: d[n] for n in self.k.DECISION_FIELDS if n in d}, 'Projected decision mismatch'); self.same(g['actual_follow_up'], m, 'Follow-up altered'); self.same(g['applied_patch'], candidate['patch'], 'Wrong edit attribution')
            choices = g['same_state_team_choices']; chosen = d['team_selected_pair']; child = {'agent': chosen['agent'], 'task_public_id': ':'.join(map(str, self.k.TASKS[chosen['task']]))} if chosen else None
            need(choices['child_team_choice'] == child and g['team_choice_differs'] == (choices['parent_team_choice'] != child) and g['parent_team_choice_is_shadow'] is True, 'Recorded child/team-shadow join')
            need(g['public_effect'] == ('selected_optional' if d['executed_action'] == 'CHECK_THIS' else 'admitted_not_selected' if ca == 'CHECK' else 'declined'), 'False public action-effect label')
            need(g['trajectory_id'] == expected_run and g['parent_skill_id'] == parent_id and g['child_skill_id'] == child_id and g['counterfactual_measurement'] == g['earlier_temperature_truth'] == 'unobserved', 'False counterfactual/trajectory claim')
            self.same(c['elapsed_to_actual_measurement_s'], m['time_s'] - d['time_s'], 'Causal elapsed time'); self.same(g['elapsed_to_actual_measurement_s'], c['elapsed_to_actual_measurement_s'], 'Difference elapsed time'); self.k.no_hidden(g); self.cards += 1
        for g in raw['g_feedback_sample']: need(g in diffs, 'Retrieved invented feedback')
        need({g['actual_follow_up']['kind'] for g in raw['g_feedback_sample']} == {g['actual_follow_up']['kind'] for g in diffs}, 'Omitted available profession coverage')
        need(len(raw['g_feedback_sample']) <= 8, 'Feedback card cap'); first = raw['g_shadow_counts']['first_difference']
        if first is not None:
            need(first['recorded_before_execution'] is True, 'Shadow recorded after action'); f = first['features']; need(first['parent_admission'] == ('CHECK' if self.k.action(self.parent, f) else 'CONTINUE_TOUR') and first['child_admission'] == ('CHECK' if self.k.action(skill, f) else 'CONTINUE_TOUR'), 'Recorded first-shadow admissions')
    def raw(self, row, candidate, case, folder, number):
        p = self.local(row['record_path']); need(p == folder / f'case-{number:02d}.json.gz' and sha(p) == row['record_SHA256'], 'Raw path/hash'); raw = self.read(p); attempt = self.read(p.with_suffix('.attempt.json')); self.same(self.read(p.with_suffix('.complete.json')), row, 'Detached completion')
        need(attempt['charged_actual_episode_attempt'] and attempt['candidate_id'] == raw['g_candidate_id'] == candidate['id'] and attempt['case_id'] == row['case_id'] == case['case_id'], 'Actual attempt/slot/case'); self.jobs.add(str(attempt['job_id']))
        need(raw['specification'] == row['specification'] == case['specification'] and raw['process_fingerprint'] == row['process_fingerprint'] == case['process_fingerprint'] and self.k.encoded(raw['policy_spec']) == self.k.encoded(candidate['policy']) and raw['policy_version_id'] == self.k.digest(candidate['policy'])[:24], 'Raw executor/policy/condition identity')
        self.same(raw['g_applied_patch'], candidate['patch'], 'Raw applied patch'); self.same(raw['metrics'], row['metrics'], 'Raw metrics detached'); self.same(raw['usage'], row['usage'], 'Raw usage detached'); self.same(raw['g_shadow_counts'], row['shadow_counts'], 'Shadow counters detached'); need(row['difference_cards'] == len(raw['g_difference_cards']), 'Difference-card count'); m = raw['metrics']
        need(row['audit']['passed'] and raw['new_neural_training'] == 0 and m['final_professional_coverage'] == 1 and all(m[n] == 0 for n in ('invalid_assignments', 'duplicate_reservations', 'tasks_violating_exact_gap')), 'Primary safety audit/coverage failure')
        actions = raw['action_log']; need([v['time_s'] for v in actions] == list(range(5, 1800, 5)), 'Full action clocks'); ah = hashlib.sha256(json.dumps(actions, sort_keys=True).encode()).hexdigest(); need(ah == row['action_SHA256'] == raw['action_log_SHA256'], 'Action digest')
        old = self.old[self.k.encoded(case['specification'])]; op = Path(old['record_path']); need(sha(op) == old['record_SHA256'] == raw['g_parent_record_SHA256'] and Path(raw['g_parent_record_path']).resolve() == op.resolve(), 'Known parent trajectory authority')
        if str(op) not in self.parents: self.parents[str(op)] = self.read(op)['action_log']
        need([v['time_s'] for v in self.parents[str(op)]] == list(range(5, 1800, 5)), 'Known parent full paired action clocks')
        first = next(({'time_s': a['time_s'], 'actual_parent_choices': a['choices'], 'actual_child_choices': b['choices'], 'scope': 'Separate real paired runs; not unobserved shadow outcome'} for a, b in zip(self.parents[str(op)], actions) if self.k.encoded(a['choices']) != self.k.encoded(b['choices'])), None)
        self.same(first, raw['g_first_actual_trajectory_difference'], 'First paired action difference'); self.same(first, row['first_actual_difference'], 'First difference receipt')
        extras = [v for v in raw['selected_certificates'] if v['task'] != v['baseline_next']]; need(len(extras) == m['selected_detours'] <= 1, 'Global CHECK quota'); self.same(extras[0] if extras else None, row['first_CHECK'], 'First CHECK receipt'); self.same(row['first_CHECK'], raw['g_first_actual_CHECK'], 'First CHECK raw')
        events = {}; previous = {}
        for e in raw['inspection_events']:
            j, t = e['task'], e['time_s']; kind, _ = self.k.TASKS[j]; value = e['measurement'][self.k.FIELDS[self.k.KINDS.index(kind)]]; alarm = value > (65 if kind == 'thermal' else 60) if kind in ('thermal', 'electrical') else value < 33.6 if kind == 'cooling' else abs(value) > 1
            transition = 'new_measured_alert' if alarm and not previous.get(j, False) else 'persistent_recheck' if alarm else 'recovery_confirmation' if previous.get(j, False) else 'normal_check'; need((j, t) not in events and e['kind'] == e['role'] == kind and e['calibrated'] is True and bool(alarm) == e['alarm'] and transition == e['measurement_transition'], 'Actual qualified alarm/transition'); events[j, t] = e; previous[j] = bool(alarm)
        waits = []; founds = []; windows = raw['anomaly_intervals']
        for w in windows:
            need(0 <= w['onset_s'] < w['end_s'] <= 1800 and w['samples'] >= 3 and self.k.equal(w['end_s'] - w['onset_s'], 5 * w['samples']) and math.isfinite(w['weight']) and w['weight'] > 0, 'Retained window contract')
            found = [t for (j, t), e in events.items() if j == w['task'] and e['alarm'] and w['onset_s'] < t <= w['end_s']]; first = min(found) if found else None; delay = (first if first is not None else w['end_s']) - w['onset_s']; need(w['detected'] == bool(found), 'Once-per-window detected flag'); self.same(first, w['first_confirmation_s'], 'First confirmation arithmetic'); self.same(delay, w['delay_or_censor_s'], 'Delay/censor arithmetic'); waits.append(w['weight'] * delay); founds.append(bool(found))
        need(m['anomaly_segments'] == len(windows) and m['confirmed_segments'] == sum(founds) and m['missed_segments'] == len(windows) - sum(founds) and m['completed_inspections'] == len(events), 'Event count arithmetic'); self.same(m['consequence_weighted_censored_delay_s'], sum(waits) / sum(v['weight'] for v in windows) if windows else None, 'Weighted wait arithmetic')
        self.same(m['miss_rate'], 1 - sum(founds) / len(windows) if windows else None, 'Miss fraction arithmetic')
        for kind in self.k.KINDS:
            selected = [w for w in windows if w['kind'] == kind]; need(raw['by_role'][kind]['segments'] == len(selected) and raw['by_role'][kind]['confirmed'] == sum(w['detected'] for w in selected), 'Professional retained-event counts')
        eligibility = self.k.digest([{n: w[n] for n in ('task', 'onset_s', 'end_s', 'weight', 'samples')} for w in windows]); need(eligibility == row['event_eligibility_SHA256'] == old['event_eligibility_SHA256'], 'Retained eligibility differs from frozen parent')
        if candidate['policy']['executor'] == 'timing-skill-v1': self.feedback(raw, candidate, events)
        else: need(raw['g_shadow_counts'] is None and not raw['g_difference_cards'], 'H01 falsely represented as typed shadow')
        self.raws += 1
    def reply(self, text):
        v = self.k.strict_json(text); need(set(v) == {'diagnosis', 'patch', 'expected_effect'} and all(type(v[n]) is str and 1 <= len(v[n]) <= 500 for n in ('diagnosis', 'expected_effect')), 'Raw patch envelope'); p = v['patch']; need(set(p) == {'schema_version', 'parent_id', 'edits'} and p['schema_version'] == 'timing-patch-v1' and p['parent_id'] == self.k.digest(self.parent) and isinstance(p['edits'], list) and 1 <= len(p['edits']) <= 2, 'Raw patch identity/count')
        q = self.prompts['rows'][0]['prompt']; view = q['parent_view']; rules = {f'r{i}': {f'r{i}.p{j}': copy.deepcopy(a) for j, a in enumerate(r['all'])} for i, r in enumerate(self.parent['rules'])}; schemas = q['operations']
        validator = types.SimpleNamespace(study={'skill_space': {'numeric_domains': q['numeric_domains'], 'boolean_fields': q['booleans'], 'special_margin_coefficients': q['margin_coefficients']}})
        for i, e in enumerate(p['edits']):
            op = e['op']; need(op in schemas and set(e) == set(schemas[op]), 'Closed patch operation')
            if op == 'add_rule': rules[f'new{i}'] = {f'new{i}.p{j}': a for j, a in enumerate(e['rule']['all'])}; need(set(e['rule']) == {'all'}, 'Closed new rule'); continue
            target = e['target']; table = view['predicates' if op in ('replace_threshold', 'delete_predicate') else 'rules']; need(target in table and e['expected_hash'] == table[target]['hash'], 'Original-node precondition'); rid = target.split('.')[0]; need(rid in rules, 'Deleted rule target')
            if op == 'delete_rule': del rules[rid]
            elif op == 'add_predicate': rules[rid][f'added{i}'] = e['predicate']
            else:
                need(target in rules[rid], 'Deleted predicate target')
                if op == 'delete_predicate': del rules[rid][target]
                else:
                    need(rules[rid][target]['field'] in set(q['numeric_domains']) | {'evidence_margin'} and rules[rid][target]['op'] != 'is_missing', 'Numeric threshold only'); rules[rid][target]['value'] = e['value']
                    self.k.Audit.skill(validator, {'schema_version': 'timing-skill-v1', 'rules': [{'all': [rules[rid][target]]}], 'default_action': 'CONTINUE_TOUR'})
        child = {'schema_version': 'timing-skill-v1', 'rules': [{'all': list(v.values())} for v in rules.values()], 'default_action': 'CONTINUE_TOUR'}; return self.k.Audit.skill(validator, child), p
    def precheck(self, supported):
        area = self.root / 'G1a-precheck'
        if not area.exists(): return {'status': 'eligible_not_executed' if supported else 'verified_not_executed_gate_closed', 'real_model_calls': 0, 'registered_execution_complete': not supported}
        need(supported, 'Model precheck entered despite closed G0 gate'); claim = self.read(area / 'launch.json'); self.launch(self.read(self.root / 'G1a-launch-proof.json'), claim, 'm12g_precheck.sbatch', []); need(claim['G0_receipt_SHA256'] == sha(self.root / 'G0-complete.json'), 'Precheck gate seal'); value = self.read(area / 'complete.json'); rows = []
        need('G1a' in self.read(self.bound['review'])['approved_scopes'] and value['complete'] and value['identity_SHA256'] == self.ish and len(self.prompts['rows']) == 6, 'Incomplete/unapproved precheck')
        for i, prompt in enumerate(self.prompts['rows']):
            base = area / 'calls' / f'call-{i:02d}'; start = self.read(base.with_suffix('.start.json')); result = self.read(base.with_suffix('.result.json')); need(start['case'] == prompt and start['attempt_charged'] and 0 < start['input_tokens'] == result['input_tokens'] <= 6144 and 0 <= result['output_tokens'] <= 1024 and start['output_token_cap'] == 1024 and json.dumps(prompt['prompt']) in start['rendered_prompt'], 'Fixed synthetic call/token receipt'); self.k.no_hidden(prompt['prompt'])
            try: child, patch = self.reply(result['raw_response']); valid = True
            except (ValueError, TypeError, KeyError, OverflowError, IndexError): valid = False
            need(result['raw_patch_valid'] == result['compiled_valid'] == valid and result['id'] == prompt['id'], 'Raw/compiled verdict mismatch')
            if valid: need(self.k.encoded(child) == self.k.encoded(result['compiled_skill']) and self.k.encoded(patch) == self.k.encoded(result['patch']) and result['different_parent'] == (self.k.digest(child) != self.k.digest(self.parent)), 'Patch compilation/no-op verdict')
            rows.append(result)
        self.same(value['rows'], rows, 'Precheck rows detached'); need(len(list((area / 'calls').glob('*.start.json'))) == len(list((area / 'calls').glob('*.result.json'))) == value['real_model_calls'] == 6 and value['raw_compiled_valid'] == sum(v['compiled_valid'] for v in rows) and value['interface_passed'] == (value['raw_compiled_valid'] >= 5) and value['synthetic_only'] and value['new_environment_episodes'] == 0 and value['no_repair_or_resampling'] and value['no_skill_gain_claim'], 'Precheck call cap/claim'); return {'status': 'completed_synthetic_only', 'real_model_calls': 6, 'interface_passed': value['interface_passed'], 'registered_execution_complete': True, 'input_tokens': sum(v['input_tokens'] for v in rows), 'output_tokens': sum(v['output_tokens'] for v in rows)}
    def run(self):
        self.metadata(); joined = self.read(self.root / 'G0-complete.json'); need(joined['complete'] and joined['identity_SHA256'] == self.ish and len(joined['sealed_batches']) == 2, 'Incomplete G0 join'); proofs = self.read(self.root / 'G0-launch-proofs.json'); need(len(proofs) == len({str(v['job_id']) for v in proofs}) == 2, 'Exactly two distinct G0 launches'); results = []; supported = []; best = None
        for batch in (0, 1):
            area = self.root / 'G0' / f'batch-{batch}'; claim = self.read(area / 'launch.json'); proof = [v for v in proofs if v['batch'] == batch]; need(len(proof) == 1 and claim['batch'] == batch and claim['episode_cap'] == 240, 'Batch launch claim'); self.launch(proof[0], claim, 'm12g_g0.sbatch', [str(batch)]); completed = self.read(area / 'complete.json'); need(completed['complete'] and completed['identity_SHA256'] == self.ish and completed['actual_environment_evaluations'] == 240 and completed['new_physical_processes'] == completed['new_model_calls'] == 0 and len(completed['results']) == 12, 'Incomplete G0 batch')
            seal = [v for v in joined['sealed_batches'] if Path(v['path']).resolve() == (area / 'complete.json').resolve()]; need(len(seal) == 1 and seal[0]['sha256'] == sha(area / 'complete.json'), 'Changed batch seal')
            for candidate, result in zip(self.candidates[batch * 12:(batch + 1) * 12], completed['results']):
                folder = area / candidate['id']; self.same(result, self.read(folder / 'complete.json'), 'Policy completion detached'); self.same(result['candidate'], candidate, 'Frozen candidate changed'); need(result['identity_SHA256'] == self.ish and result['eligible_local_mechanism'] == candidate['eligible_local_mechanism'] and len(result['rows']) == 20, 'Candidate identity/cohort')
                for i, (case, row) in enumerate(zip(self.cases, result['rows'])): self.raw(row, candidate, case, folder, i); need(str(self.read(Path(row['record_path']).with_suffix('.attempt.json'))['job_id']) == str(claim['job_id']), 'Episode produced by foreign batch job')
                stats = self.aggregate(result['rows']); self.same(stats, result['statistics'], 'Independent hierarchical macro statistics'); gain = self.k.gain(stats, self.initial['statistics']); need(gain == result['gain_vs_initial'] and result['actual_trajectory_differences_vs_initial'] == sum(v['action_SHA256'] != self.old[self.k.encoded(v['specification'])]['action_SHA256'] for v in result['rows']), 'Gain/trajectory verdict')
                if gain and candidate['eligible_local_mechanism']:
                    supported.append(candidate['id'])
                    if best is None or self.k.better(stats, best['statistics']): best = result
                results.append(result)
        self.same(joined['results'], results, 'Joined results detached'); self.same(joined['initial_reference_statistics'], self.initial['statistics'], 'Changed parent summary'); need(joined['local_mechanism_supported'] == bool(supported) and joined['supported_local_candidates'] == supported and joined['selected_local_reference'] == (best['candidate'] if best else None), 'Independent mechanism/selection verdict')
        need(joined['actual_environment_evaluations'] == self.raws == len(list((self.root / 'G0').rglob('case-*.attempt.json'))) == len(list((self.root / 'G0').rglob('case-*.json.gz'))) == 480 and joined['candidate_slots'] == 24 and joined['all_slots_actual'] and joined['new_physical_processes'] == joined['model_calls'] == 0 and joined['H01_only_gain_does_not_open_precheck'], 'Episode/slot/call cap')
        self.gpu_allowed = bool(supported); precheck = self.precheck(bool(supported)); return {'G0_complete': True, 'supported_local_candidates': supported, 'selected_local_reference': best['candidate']['id'] if best else None, 'conditional_G1a': precheck}
    def costs(self, path):
        rows = list(csv.DictReader(Path(path).read_text().splitlines(), delimiter='|')); need(len({v['JobID'] for v in rows}) == len(rows) and self.jobs <= {v['JobID'] for v in rows} and all(v['State'] in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT') for v in rows), 'Incomplete accounting prefix'); need(all(v['State'] == 'COMPLETED' and v['ExitCode'] == '0:0' for v in rows if v['JobID'] in self.jobs), 'Producing job failed'); events = []
        for v in rows:
            if int(v['ElapsedRaw']): events.extend([(datetime.datetime.fromisoformat(v['Start']), 1), (datetime.datetime.fromisoformat(v['End']), -1)])
            if v['JobID'] in self.modes:
                gpu = self.modes[v['JobID']] == 'm12g_precheck.sbatch'; need(int(v['AllocCPUS']) == (4 if gpu else 2) and v['Partition'] == ('4gpu' if gpu else 'cu') and int(v['ElapsedRaw']) <= (900 if gpu else 2700), 'Actual mode allocation differs from frozen wrapper')
        active = peak = 0
        for _, change in sorted(events, key=lambda v: (v[0], v[1])): active += change; peak = max(peak, active)
        cpu = sum(int(v['ElapsedRaw']) * int(v['AllocCPUS']) / 60 for v in rows); gpu = [v for v in rows if 'gres/gpu=' in v['AllocTRES']]; seconds = sum(int(v['ElapsedRaw']) for v in gpu); storage = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file()); need(cpu <= 360 and len(gpu) <= 1 and (not gpu or self.gpu_allowed) and seconds <= 900 and peak <= 2 and storage <= 2000000000 and all(int(re.search(r'(?:^|,)gres/gpu=(\d+)', v['AllocTRES']).group(1)) == 1 for v in gpu), 'Resource cap'); self.inputs[str(Path(path).resolve())] = sha(path)
        return {'scope': 'Completed-allocation prefix; current audit and subsequent allocations excluded until final ledger refresh.', 'CPU_core_minutes': cpu, 'GPU_allocations': len(gpu), 'GPU_seconds': seconds, 'max_concurrent_allocations': peak, 'new_storage_bytes_before_report': storage, 'producer_job_ids': sorted(self.jobs), 'whole_goal_post_audit_costs_verified': False}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('run-root', 'source-root', 'accounting', 'output'): p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--f-audit-kernel', type=Path); a = p.parse_args(); need(not a.output.exists(), 'Choose a new immutable report path'); audit = None
    try:
        audit = Audit(a); result = audit.run(); costs = audit.costs(a.accounting)
        for path, h in audit.inputs.items(): need(sha(path) == h, 'Consumed input changed before report')
        report = {'passed': True, 'result': result, 'raw_records_checked': audit.raws, 'feedback_cards_checked': audit.cards, 'accounting_prefix': costs, 'input_SHA256': audit.inputs, 'auditor_SHA256': sha(__file__), 'new_model_calls': 0, 'new_environment_episodes': 0, 'new_physical_processes': 0, 'scope': 'Retained-window/event arithmetic, raw/action hashes, all card admissions and event joins, paired macro/gain/selection and source/launch/cost joins. No NPZ, neural, policy or physics replay. Full physical labels, schedule/return geometry and event measurement truth rely on the unchanged primary per-episode independent audits. Complete team-pool shadows and earliest same-state difference are source-bound recorded evidence, not independently reconstructed without a complete candidate ledger. First actual trajectory difference is independently compared over both real action logs. selected_optional labels the earlier decision; a later follow-up can be a Tour revisit and is not counted as CHECK success. Token counts use receipts, not a new tokenizer. G1a compilation is independently checked against the registered JSON contract; synthetic challenge-specific usefulness is not claimed. Execution completion does not assert delivery/report completion.', 'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        need(not a.output.resolve().is_relative_to(audit.root) or costs['new_storage_bytes_before_report'] + len((json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode()) <= 2000000000, 'Report would exceed storage cap')
    except Exception as e: report = {'passed': False, 'error_type': type(e).__name__, 'error': str(e), 'auditor_SHA256': sha(__file__), 'input_SHA256': audit.inputs if audit else {}, 'new_model_calls': 0, 'new_environment_episodes': 0}
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with a.output.open('x', encoding='utf-8', newline='\n') as f: f.write(text)
    need(json.loads(a.output.read_text(encoding='utf-8')) == report, 'Report read-back failure'); print('M12G_RESULT_AUDIT', report['passed'], sha(a.output)); return 0 if report['passed'] else 1
if __name__ == '__main__': sys.exit(main())
