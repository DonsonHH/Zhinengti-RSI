"""Independent M12-F receipt/array audit. No project, Torch or simulator imports.

Run this sidecar OUTSIDE code-v6. Inputs are read-only; only a new report is written.
Accounting is a completed-allocation prefix, excluding this audit's unfinished job.
"""
import argparse
import collections
import csv
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import sys

import numpy as np

TOL = 1e-9
KINDS = ('thermal', 'electrical', 'cooling', 'control')
FIELDS = ('temperature_c', 'contact_temperature_c', 'pod_flow_kg_s', 'controller_sensor_error_c')
DEADLINES = np.array([300.] * 128 + [600.] * 48)
TASKS = [('thermal', i) for i in range(128)] + [(kind, i) for kind in KINDS[1:] for i in range(16)]
DECISION_FIELDS = ('time_s', 'task_public_id', 'agent', 'features', 'skill_action', 'executed_action', 'team_action', 'candidate_finish_s', 'Tour_target_finish_s', 'matched_rule_index', 'check_intent')
MEASUREMENT_FIELDS = ('time_s', 'kind', 'value', 'field', 'unit', 'qualified_role', 'calibrated', 'observed_alarm', 'transition')
FORBIDDEN = {'seed', 'scenario', 'fault_start_s', 'fault_end_s', 'onset_s', 'end_s', 'oracle', 'truth', 'anomaly_intervals', 'metrics', 'macro_missed_count', 'macro_wait_s', 'process_fingerprint', 'record_path', 'final_test'}


def need(value, message):
    if not value:
        raise ValueError(message)


def encoded(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
def digest(value): return hashlib.sha256(encoded(value).encode()).hexdigest()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()
def mean(values): return sum(values) / len(values) if values else None
def equal(a, b):
    if isinstance(a, dict): return isinstance(b, dict) and set(a) == set(b) and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, list): return isinstance(b, list) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    if a is None or b is None: return a is b
    if type(a) is bool or type(b) is bool: return type(a) is type(b) and a == b
    return math.isclose(a, b, rel_tol=0, abs_tol=TOL) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else a == b


def strict_json(text):
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, 'Duplicate JSON key'); out[key] = value
        return out
    def number(value):
        result = float(value); need(math.isfinite(result), 'Nonfinite JSON number'); return result
    return json.loads(text, object_pairs_hook=pairs, parse_float=number, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON constant')))


def no_hidden(value):
    if isinstance(value, dict):
        need(not set(value).intersection(FORBIDDEN), 'Hidden field in actual model prompt')
        for item in value.values(): no_hidden(item)
    elif isinstance(value, list):
        for item in value: no_hidden(item)


def action(skill, public):
    def test(atom):
        field, op, value = atom['field'], atom['op'], atom['value']; actual = public.get(field)
        if field == 'evidence_margin':
            terms = [public.get(k) for k in ('professional_signal', 'importance', 'lead_s', 'shift_s')]
            return all(x is not None for x in terms) and terms[0] * terms[1] * terms[2] > value * terms[3]
        if op == 'is_missing': return actual is None
        if actual is None: return False
        if value == 'kind_threshold':
            if public.get('kind') not in KINDS: return False
            value = .5 if public['kind'] == 'control' else .05
        return {'eq': lambda: actual == value, 'ge': lambda: actual >= value, 'gt': lambda: actual > value, 'le': lambda: actual <= value, 'lt': lambda: actual < value}[op]()
    return any(all(test(atom) for atom in row['all']) for row in skill['rules'])


def paired(candidate, baseline):
    need(candidate['cohort_SHA256'] == baseline['cohort_SHA256'] and candidate['process_event_identities'] == baseline['process_event_identities'], 'Unpaired comparison')
    old = {v['seed']: v['macro_missed_count'] for v in baseline['seeds']}
    need({v['seed'] for v in candidate['seeds']} == set(old), 'Different comparison seeds')
    return all(v['macro_missed_count'] <= old[v['seed']] + TOL for v in candidate['seeds'])


def nonregression(c, b):
    return paired(c, b) and (c['macro_missed_count'] < b['macro_missed_count'] - TOL or abs(c['macro_missed_count'] - b['macro_missed_count']) <= TOL and (c['macro_wait_s'] is None and b['macro_wait_s'] is None or c['macro_wait_s'] is not None and b['macro_wait_s'] is not None and c['macro_wait_s'] <= b['macro_wait_s'] + TOL))


def gain(c, b):
    if not paired(c, b): return False
    if c['macro_missed_count'] < b['macro_missed_count'] - TOL: return True
    if abs(c['macro_missed_count'] - b['macro_missed_count']) > TOL: return False
    cw, bw = c['macro_wait_s'], b['macro_wait_s']
    if cw is not None and bw is not None and bw > TOL and (bw - cw) / bw >= .05 - TOL: return True
    return equal(cw, bw) and b['mean_CHECK'] > TOL and (b['mean_CHECK'] - c['mean_CHECK']) / b['mean_CHECK'] >= .05 - TOL and c['mean_walk_m'] <= b['mean_walk_m'] + TOL


def better(c, b):
    if (c['macro_wait_s'] is None) != (b['macro_wait_s'] is None): return False
    for key in ('macro_missed_count', 'macro_wait_s', 'mean_CHECK', 'mean_walk_m'):
        if c[key] is not None and abs(c[key] - b[key]) > TOL: return c[key] < b[key]
    return False


class Audit:
    def __init__(self, root, source):
        self.root, self.source = root.resolve(), source.resolve(); self.inputs = {}; self.raw_seen = {}; self.traces = {}; self.feedback = {}; self.probes = {}; self.jobs = set(); self.launches = {}; self.reservations = {'episodes': [], 'model_calls': []}; self.call_count = self.episodes = self.cards_checked = 0

    def read(self, path):
        path = Path(path).resolve(); data = path.read_bytes(); h = hashlib.sha256(data).hexdigest()
        need(str(path) not in self.inputs or self.inputs[str(path)] == h, 'Input changed during audit: ' + str(path)); self.inputs[str(path)] = h
        return strict_json(gzip.decompress(data).decode() if path.suffix == '.gz' else data.decode('utf-8'))

    def local(self, path):
        path = Path(path).resolve(); need(path.is_relative_to(self.root), 'Raw/output path outside this campaign'); return path

    def receipt(self, path):
        value = self.read(path); need(value['complete'] and value['identity_SHA256'] == self.identity_sha, 'Incomplete or foreign phase: ' + str(path))
        for row in value.get('sealed_artifacts', []): need(sha(self.local(row['path'])) == row['sha256'], 'Changed consumed phase seal')
        return value

    def seals(self, value, required):
        paths = {Path(v['path']).resolve() for v in value['sealed_artifacts']}; need({Path(v).resolve() for v in required} <= paths, 'Consumed phase input omitted from seal')

    def launch(self, mode, trial=None, batch=None):
        proofs = self.read(self.root / 'launch-proofs.json'); found = [v for v in proofs if v['mode'] == mode and (trial is None or v['trial'] == trial) and (batch is None or v['batch'] == batch)]; need(len(found) == 1, 'Exactly one held launch proof required'); proof = found[0]; claim_path = self.local(proof['launch_claim_path']); need(sha(claim_path) == proof['launch_claim_SHA256'], 'Launch claim hash mismatch'); claim = self.read(claim_path)
        need(proof['held_verified'] and proof['identity_SHA256'] == claim['identity_SHA256'] == self.identity_sha and proof['job_id'] == claim['job_id'] and (mode, proof['trial'], proof['batch']) == (claim['mode'], claim['trial'], claim['batch']) and claim['single_launch_only'] and Path(proof['source_root']).resolve() == self.source and Path(proof['run_root']).resolve() == self.root, 'Foreign or repeated launch')
        wrapper = self.source / 'cluster' / ('m12f_gpu.sbatch' if mode in ('F1_search', 'F2_adapt') else 'm12f_cpu.sbatch'); need(sha(self.local(proof['script_path'])) == proof['script_SHA256'] == sha(wrapper), 'Actual held script differs from frozen wrapper'); held = self.local(proof['held_job_path']).read_text(); command = shlex.split(proof['expected_command']); parameter = proof['trial'] if mode in ('F1_search', 'F2_adapt') else proof['batch']
        need(command == [str(wrapper), str(self.source), str(self.root), mode, str(parameter)] and 'Command=' + proof['expected_command'] in held and 'JobId=' + proof['job_id'] in held and ('JobHeldUser' in held or 'JobHeldAdmin' in held), 'Held command/job/arguments not verified')
        original_path = self.local(proof['original_submission_path']); need(sha(original_path) == proof['original_submission_SHA256'], 'Original submission hash mismatch'); original = self.read(original_path); need(str(original['job_id']) == proof['job_id'] and original.get('actual_script_SHA256', original.get('script_SHA256')) == proof['script_SHA256'], 'Normalized launch does not match original submission')
        for key in ('identity_SHA256', 'identitySHA'):
            if key in original: need(original[key] == self.identity_sha, 'Original submission identity mismatch')
        if 'mode' in original: need(original['mode'] == mode, 'Original submission mode mismatch')
        if 'registered_mode' in original: need(original['registered_mode'] == mode, 'Original registered mode mismatch')
        if 'held_verified' in original: need(original['held_verified'] is True, 'Original held verification failed')
        self.jobs.add(proof['job_id']); self.launches[proof['job_id']] = proof; return proof

    def metadata(self):
        self.identity = self.read(self.root / 'campaign-execution-identity.json'); self.identity_sha = sha(self.root / 'campaign-execution-identity.json'); bound = self.identity['bound_inputs']; self.history = self.read(bound['history'])
        need(Path(self.identity['source_root']).resolve() == self.source and Path(self.identity['run_root']).resolve() == self.root, 'Actual source/run-root mismatch'); self.study = self.read(bound['study']); self.cases = self.read(bound['cases']); self.templates = self.read(bound['prompts']); self.initial = self.study['skill_space']['initial_skill']; need(self.identity['baseline'] == self.study['baseline'] == self.cases['baseline'] == '39d06ce5edb4dc7f7fff18bedc76909350a946d4', 'Foreign study baseline')
        need(sha(bound['cases']) == self.study['cases_SHA256'], 'Study/cases mismatch'); manifest = self.read(bound['source_manifest']); need(sha(bound['source_archive']) == manifest['sha256'], 'Source archive mismatch')
        for name, row in manifest['files'].items(): need(sha(self.source / name) == row['sha256'], 'Changed snapshot member: ' + name)
        indexed = {v['path']: v['sha256'] for v in self.identity['artifacts']}
        for path, h in indexed.items(): need(sha(path) == h, 'Changed runtime artifact: ' + path)
        review = self.read(bound['review']); need(review['approved'] and review['source_archive_SHA256'] == manifest['sha256'], 'Unapproved source')
        for key, h in review['reviewed_input_SHA256'].items(): need(sha(bound[key]) == h, 'Changed reviewed input')
        gate = self.read(bound['gate']); need(Path(gate['source_root']).resolve() == self.source and gate['success'] and gate['tests'] >= 341 and len(set(gate['F_test_ids'])) >= 24 and not (gate['errors'] or gate['failures'] or gate['skipped']), 'Incomplete/foreign actual Torch gate')
        for name, h in gate['F_test_sources'].items(): need(sha(self.source / name) == h, 'Gate/test hash mismatch')
        proof, old, audit = (self.read(bound[k]) for k in ('F0_complete', 'F0_manifest', 'F0_audit'))
        need(proof['complete'] and proof['all_action_hashes_and_metrics_equal'] and proof['initial_E1_candidate_parity'] == 298152 and proof['charged_known_closed_loop_episodes'] == 20 and proof['predicate_mismatches'] == 0 and audit['passed'] and audit['complete_receipt_SHA256'] == sha(bound['F0_complete']) and audit['source_archive_SHA256'] == old['sha256'], 'F0 proof mismatch')
        for name in ('src/patrol_rsi/timing_skill.py', 'src/patrol_rsi/timing_skill_features.py', 'src/patrol_rsi/timing_skill_control.py', 'scripts/timing_skill_execution.py'): need(manifest['files'][name]['sha256'] == old['files'][name]['sha256'], 'Changed F0 interpreter')
        self.context = {'source_archive_SHA256': manifest['sha256'], 'CVAE_SHA256': indexed[bound['CVAE']], 'actor0_SHA256': indexed[bound['actor0']], 'study_SHA256': sha(bound['study'])}

    def skill(self, value):
        need(isinstance(value, dict) and set(value) == {'schema_version', 'rules', 'default_action'} and value['schema_version'] == 'timing-skill-v1' and value['default_action'] == 'CONTINUE_TOUR', 'Invalid skill envelope')
        need(isinstance(value['rules'], list) and len(value['rules']) <= 3, 'Invalid skill OR count'); space = self.study['skill_space']; numeric = space['numeric_domains']; boolean = space['boolean_fields']
        for row in value['rules']:
            need(set(row) == {'all'} and isinstance(row['all'], list) and 1 <= len(row['all']) <= 6, 'Invalid AND rule')
            for atom in row['all']:
                need(isinstance(atom, dict) and set(atom) == {'field', 'op', 'value'}, 'Invalid predicate'); f, op, v = (atom[k] for k in ('field', 'op', 'value'))
                need(isinstance(f, str) and f in set(numeric) | set(boolean) | {'kind', 'evidence_margin'}, 'Unknown skill field')
                valid = op == 'positive' and type(v) in (int, float) and v in space['special_margin_coefficients'] if f == 'evidence_margin' else v is None if op == 'is_missing' else op == 'eq' and type(v) is bool if f in boolean else op == 'eq' and v in KINDS if f == 'kind' else op in ('ge', 'gt', 'le', 'lt') and (f == 'professional_signal' and v == 'kind_threshold' or type(v) in (int, float) and v in numeric[f])
                need(valid, 'Invalid predicate operator/value')
        return value

    def reply(self, text, meta=False):
        value = strict_json(text); key = 'protocol' if meta else 'skill'; need(isinstance(value, dict) and set(value) == {'diagnosis', key, 'expected_effect'}, 'Invalid reply envelope')
        need(all(type(value[k]) is str and 1 <= len(value[k]) <= 1500 for k in ('diagnosis', 'expected_effect')), 'Invalid explanation')
        if not meta: return self.skill(value[key])
        p = value[key]; schema = self.templates['meta_empty_context_template']['schema']; need(set(p) == {'protocol', 'generator', 'dedup', 'acceptance', 'failure_fallback'} and set(p['protocol']) == {'diagnostics', 'memory', 'instruction'}, 'Invalid meta nesting')
        inner = p['protocol']; need(isinstance(inner['diagnostics'], list) and 1 <= len(inner['diagnostics']) <= 3 and len(set(inner['diagnostics'])) == len(inner['diagnostics']) and set(inner['diagnostics']) <= set(schema['diagnostic_choices']) and inner['memory'] in schema['memory_choices'] and inner['instruction'] in schema['instruction_choices'], 'Invalid meta values')
        need(p['generator'] in schema['generator'] and p['dedup'] in schema['dedup'] and p['acceptance'] == 'strict_improvement' and p['failure_fallback'] == 'keep_parent', 'Illegal protocol edit'); return p

    def trace(self, case):
        folder = Path(case['trace_folder']).resolve(); spec = case['specification']; known = spec['phase'] == 'development'
        if known: need(any(v['specification'] == spec and Path(v['trace_folder']).resolve() == folder for v in self.cases['known_development']), 'Old source is not a frozen known-development case')
        else: need(folder.is_relative_to(self.root / 'future-sources' / case['partition']), 'Foreign/final-old trace path')
        meta = self.read(folder / 'complete.json'); config = meta['metadata']['configuration']; need(sha(folder / 'complete.json') == case['actual_source_meta_SHA256'] and meta['fingerprint'] == case['process_fingerprint'] and sha(folder / 'trace.npz') == meta['trace_file_SHA256'] and meta['seed'] == spec['seed'] and meta['scenario'] == spec['scenario'] and config['fault_start_s'] == spec['fault_start_s'] and config['fault_end_s'] == spec['fault_start_s'] + spec['duration_s'], 'Changed/mislabelled physical source')
        if not known:
            need(spec['seed'] not in self.history['seed_groups'] and meta['fingerprint'] not in self.history['numeric_fingerprints'], 'Historical confirmation-source overlap'); index = self.read(self.root / 'future-numeric-process-index.json'); need(index[meta['fingerprint']] == {'partition': case['partition'], 'physical_key': '--'.join(str(spec[k]) for k in ('seed', 'scenario', 'duration_s', 'fault_start_s'))}, 'Cross-partition numerical identity mismatch')
        fp = meta['fingerprint']
        if fp not in self.traces:
            with np.load(folder / 'trace.npz', allow_pickle=False) as archive: arrays = {k: archive[k] for k in meta['array_names']}
            h = hashlib.sha256()
            for name, array in sorted(arrays.items()): h.update(name.encode() + str(array.shape).encode() + array.dtype.str.encode()); h.update(array.tobytes())
            need(h.hexdigest() == fp, 'Numerical process digest mismatch'); temp = arrays['rack_temperature_c']; flow = arrays['rack_flow_kg_s'].reshape(360, 16, 8).sum(2); error = (arrays['public_sensor_temperature_c'] - temp).reshape(360, 16, 8).mean(2)
            values = np.concatenate((temp, arrays['contact_temperature_c'], flow, error), axis=1); labels = np.concatenate((temp > 65, arrays['contact_temperature_c'] > 60, flow < 33.6, abs(error) > 1), axis=1)
            excess = np.concatenate((np.maximum(temp - 65, 0) / 20, np.maximum(arrays['contact_temperature_c'] - 60, 0) / 30, np.maximum(1 - flow / 33.6, 0), abs(error) / 18), axis=1); weights = np.r_[arrays['asset_weights'], np.tile(arrays['asset_weights'].reshape(16, 8).mean(1), 3)]; windows = []
            for j in range(176):
                changes = np.diff(np.r_[False, labels[:, j], False].astype(int))
                for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
                    if end - start >= 3: windows.append({'task': j, 'onset_s': float(start * 5), 'end_s': float(end * 5), 'samples': int(end - start), 'weight': float(weights[j] * (1 + excess[start:end, j].mean()))})
            self.traces[fp] = values, labels, weights, windows
        return self.traces[fp]

    def raw(self, row, policy, case):
        path = self.local(row['record_path']); need(sha(path) == row['record_SHA256'], 'Raw record hash mismatch')
        if str(path) in self.raw_seen: return self.raw_seen[str(path)]
        raw = self.read(path); attempt, completed = self.read(path.with_suffix('.attempt.json')), self.read(path.with_suffix('.completed.json')); need(encoded(completed) == encoded(row) and attempt['charged_actual_episode_attempt'] and attempt['specification'] == case['specification'] and encoded(attempt['policy_spec']) == encoded(policy), 'Attempt/completion mismatch'); need(encoded(raw['policy_spec']) == encoded(policy) and raw['specification'] == row['specification'] == case['specification'] and raw['process_fingerprint'] == case['process_fingerprint'], 'Raw policy/case mismatch')
        need(raw['new_neural_training'] == 0 and equal(raw['metrics'], row['metrics']) and row['audit']['passed'] and raw['metrics']['final_professional_coverage'] == 1 and all(raw['metrics'][k] == 0 for k in ('invalid_assignments', 'duplicate_reservations', 'tasks_violating_exact_gap')), 'Invalid raw execution'); actions = {v['time_s']: {int(k): x for k, x in v['choices'].items()} for v in raw['action_log']}; need(set(actions) == set(range(5, 1800, 5)), 'Incomplete action trajectory')
        shell = {'refresh_s': 30, 'cooldown_s': 60, 'fallback': 'tour', 'rules': [{'signal': 'thermal_excess', 'threshold': .05, 'action': 'signal'}, {'signal': 'age', 'threshold': .75, 'action': 'return_tour'}] if policy['executor'] == 'original-human-v1' else []}; need(encoded(raw['workflow']) == encoded(raw['physical_shell']['program']) == encoded(shell) and raw['policy_version_id'] == digest(policy)[:24], 'Changed physical shell/policy identity')
        ah = hashlib.sha256(json.dumps(raw['action_log'], sort_keys=True).encode()).hexdigest(); need(ah == row['action_SHA256'] == raw['action_log_SHA256'], 'Action hash mismatch'); values, labels, weights, windows = self.trace(case); need(equal(raw['workflow_contract']['task_weights'], weights.tolist()) and equal(raw['workflow_contract']['deadlines_s'], DEADLINES.tolist()), 'Changed task contract')
        team = {v['id']: v for v in raw['team']['per_agent']}; owners = {j: i for i, c in enumerate(raw['initial_certificates']) for j in c['fallback_order']}; need(len(team) == 18 and len(owners) == sum(len(c['fallback_order']) for c in raw['initial_certificates']) == 176 and collections.Counter(v['role'] for v in team.values()) == {'thermal': 12, 'electrical': 2, 'cooling': 2, 'control': 2} and all(c['ok'] for c in raw['initial_certificates']), 'Changed fleet/ownership'); events = {}; by_task = collections.defaultdict(list); previous = {}
        need(all(equal(v['travel_m'] + v['service_s'] + v['idle_s'], 1795.) for v in team.values()) and all(equal(sum(v[field] for v in team.values()), raw['metrics'][field]) for field in ('travel_m', 'service_s', 'idle_s')), 'Physical work/time conservation failure')
        for event in raw['inspection_events']:
            j, t = event['task'], event['time_s']; kind, index = TASKS[j]; field = FIELDS[KINDS.index(kind)]; tick = int(t / 5) - 1; measured = values[tick, j] + (np.random.default_rng(3000000 + case['specification']['seed'] * 10000 + index * 361 + tick).normal(0, .1) if kind == 'thermal' else 0.)
            alarm = measured > (65 if kind == 'thermal' else 60) if kind in ('thermal', 'electrical') else measured < 33.6 if kind == 'cooling' else abs(measured) > 1; transition = 'new_measured_alert' if alarm and not previous.get(j, False) else 'persistent_recheck' if alarm else 'recovery_confirmation' if previous.get(j, False) else 'normal_check'
            need((j, t) not in events and kind == event['kind'] == event['role'] == team[event['agent']]['role'] and owners[j] == event['agent'] and event['calibrated'] is True and event['access'] == 'external_sensing' and equal(measured, event['measurement'][field]) and bool(alarm) == event['alarm'] and transition == event['measurement_transition'], 'Invalid qualified/reference event'); previous[j] = bool(alarm); events[j, t] = event; by_task[j].append(event)
        extras = [v for v in raw['workflow_decisions'] if v['task'] != v['baseline_next']]; need(len(extras) == raw['metrics']['selected_detours'] <= 1 and all(not v['one_extra_already_used'] and v['detours_used_before'] < 2 for v in extras), 'Quota failure')
        for extra in extras: need(not [e for e in by_task[extra['task']] if extra['time_s'] - 60 < e['time_s'] <= extra['time_s']], 'Cooldown failure')
        for selected in raw['selected_certificates']:
            cert = selected['certificate']; need(cert['ok'] and len(cert['fallback_order']) == len(set(cert['fallback_order'])) and all(t <= due + TOL for t, due in zip(cert['fallback_completion_s'], cert['fallback_due_s'])) and cert['candidate_finish_s'] <= cert['candidate_due_s'] + TOL, 'Full-return certificate failure')
        for gap in raw['exact_task_gaps']:
            times = sorted(v['time_s'] for v in by_task[gap['task']]); maximum = max(np.diff([0.] + times + [1800.])); need(times == gap['completion_times_s'] and equal(maximum, gap['maximum_gap_including_initial_and_terminal_s']) and maximum <= DEADLINES[gap['task']] + TOL, 'Exact head/tail revisit failure')
        need(len(windows) == len(raw['anomaly_intervals']), 'Changed retained event eligibility'); waits = []; confirmed = 0
        for expected, actual in zip(windows, raw['anomaly_intervals']):
            need(all(equal(actual[k], v) for k, v in expected.items()), 'Physical event window/weight mismatch'); found = [e['time_s'] for e in by_task[expected['task']] if e['alarm'] and expected['onset_s'] <= e['time_s'] <= expected['end_s'] and labels[int(e['time_s'] / 5) - 1, expected['task']]]; first = min(found) if found else None; delay = (first if first is not None else expected['end_s']) - expected['onset_s']
            need(equal(first, actual['first_confirmation_s']) and bool(found) == actual['detected'] and equal(delay, actual['delay_or_censor_s']), 'Once-per-event confirmation arithmetic'); confirmed += bool(found); waits.append(expected['weight'] * delay)
        need(raw['metrics']['anomaly_segments'] == len(windows) and raw['metrics']['confirmed_segments'] == confirmed and raw['metrics']['missed_segments'] == len(windows) - confirmed and equal(raw['metrics']['consequence_weighted_censored_delay_s'], sum(waits) / sum(v['weight'] for v in windows) if windows else None), 'Miss/censored-wait arithmetic')
        seen_cards = set()
        for card in raw['causal_experience_cards']:
            d, m = card['decision'], card['measurement']; event = events.get((d['task'], m['time_s'])); need(event is not None and (d['task'], m['time_s']) not in seen_cards and d['time_s'] < m['time_s'] and equal(card['elapsed_to_actual_measurement_s'], m['time_s'] - d['time_s']), 'Noncausal/duplicate feedback'); seen_cards.add((d['task'], m['time_s']))
            kind = event['kind']; need(m['kind'] == kind == m['qualified_role'] and m['field'] == FIELDS[KINDS.index(kind)] and m['unit'] == ('kg/s' if kind == 'cooling' else 'degC') and m['calibrated'] is True and m['value'] == event['measurement'][m['field']] and m['observed_alarm'] == event['alarm'] and m['transition'] == event['measurement_transition'], 'Feedback/event mismatch'); last = [e for e in by_task[d['task']] if e['time_s'] <= d['time_s']]; f = d['features']; need(f['already_alarm'] == (last[-1]['alarm'] if last else False), 'Noncausal alarm feature')
            if not f['direct_available']: need(all(f[k] is None for k in ('direct_age_s', 'direct_age_ratio', 'direct_reading_normal', 'direct_is_qualified')), 'Missing is not normal/zero')
            if not f['professional_available']: need(f['professional_signal'] is None, 'Missing professional signal is not zero')
            if f['direct_is_qualified'] is True: need(last and equal(d['time_s'] - f['direct_age_s'], last[-1]['time_s']), 'False qualified-reading provenance')
            self.cards_checked += 1
        eligibility = digest([{k: v[k] for k in ('task', 'onset_s', 'end_s', 'weight', 'samples')} for v in raw['anomaly_intervals']]); need(eligibility == row['event_eligibility_SHA256'], 'Event identity mismatch'); result = {'metrics': raw['metrics'], 'eligibility': eligibility, 'action': ah}; self.raw_seen[str(path)] = result; self.episodes += 1; return result

    def aggregate(self, rows, expected, cases, policy):
        need(collections.Counter(encoded(v['specification']) for v in rows) == collections.Counter(encoded(v) for v in expected), 'Wrong condition cohort'); groups = collections.defaultdict(list); identities = []; seeds = collections.defaultdict(list)
        for row in rows: self.raw(row, policy, cases[encoded(row['specification'])]); spec = row['specification']; groups[tuple(spec[k] for k in ('seed', 'scenario', 'duration_s', 'fault_start_s'))].append(row)
        for key, pair in groups.items():
            need(len(pair) == 2 and {v['specification']['noise_c'] for v in pair} == {.2, 1.} and len({v['process_fingerprint'] for v in pair}) == len({v['event_eligibility_SHA256'] for v in pair}) == 1, 'Unpaired physical/noise cohort'); waits = [v['metrics']['consequence_weighted_censored_delay_s'] for v in pair]; need((waits[0] is None) == (waits[1] is None), 'Wait eligibility mismatch'); seeds[key[0]].append((mean([v['metrics']['missed_segments'] for v in pair]), mean(waits) if waits[0] is not None else None)); identities.append([list(key), pair[0]['process_fingerprint'], pair[0]['event_eligibility_SHA256']])
        need(all(len(v) == 5 for v in seeds.values()), 'Exactly five physical processes per seed'); per_seed = [{'seed': seed, 'macro_missed_count': mean([x[0] for x in values]), 'macro_wait_s': mean([x[1] for x in values if x[1] is not None])} for seed, values in sorted(seeds.items())]
        return {'seeds': per_seed, 'macro_missed_count': mean([v['macro_missed_count'] for v in per_seed]), 'macro_wait_s': mean([v['macro_wait_s'] for v in per_seed if v['macro_wait_s'] is not None]), 'mean_CHECK': mean([v['metrics']['selected_detours'] for v in rows]), 'mean_walk_m': mean([v['metrics']['travel_m'] for v in rows]), 'conditions': len(rows), 'physical_processes': len(groups), 'cohort_SHA256': hashlib.sha256(json.dumps(sorted(encoded(v) for v in expected)).encode()).hexdigest(), 'process_event_identities': sorted(identities)}

    def store(self, area, expected, cap):
        ledger = self.read(area / 'store-complete.json'); need(ledger['cap'] == cap and ledger['attempted_episode_reservations'] == len(ledger['results']) * len(expected) <= cap, 'Store episode cap'); results = {}; cases = None
        for result in ledger['results']:
            folder = self.local(result['rows'][0]['record_path']).parent; start = self.read(folder / 'started.json'); need(encoded(start['context']) == encoded(self.context) and equal(self.read(folder / 'complete.json'), result), 'Evaluation receipt mismatch'); cohort = {encoded(v['specification']): v for v in start['cases']}; need(set(cohort) == {encoded(v) for v in expected}, 'Foreign evaluator cohort'); cases = cohort
            need(result['key'] == digest({'cases': start['cases'], 'policy': result['policy'], 'context': start['context']}), 'Evaluator identity mismatch'); stats = self.aggregate(result['rows'], expected, cohort, result['policy']); need(equal(stats, result['statistics']), 'Independent hierarchical statistics mismatch'); result['statistics'] = stats; results[result['key']] = result; self.jobs.add(start['reservation']['job_id']); self.reservations['episodes'].append(start['reservation']); need(start['reservation']['amount'] == len(expected), 'Incorrect episode reservation'); need(len(list(folder.glob('case-*.attempt.json'))) == len(list(folder.glob('case-*.completed.json'))) == len(result['rows']), 'Uncompleted/extra actual episode attempt')
        seen = set()
        for query in ledger['requests']: need(query['key'] in results and query['cache_hit'] == (query['key'] in seen) and encoded(query['policy']) == encoded(results[query['key']]['policy']), 'False cache/reference receipt'); seen.add(query['key'])
        need(len(results) == len(ledger['results']) and len(ledger['requests']) == ledger['logical_requests'] and sum(v['cache_hit'] for v in ledger['requests']) == ledger['cache_hits'], 'Cache accounting mismatch'); return {'ledger': ledger, 'results': results, 'expected': expected, 'cases': cases, 'queries': []}

    def result(self, store, policy, request=False):
        matches = [v for v in store['results'].values() if encoded(v['policy']) == encoded(policy)]; need(len(matches) == 1, 'Exact policy evaluation required')
        if request: store['queries'].append(encoded(policy))
        return matches[0]

    def retrieve(self, cards, diagnostics):
        def order(c):
            f = c['decision']['features']; values = {'professional_evidence': -int(c['measurement']['observed_alarm']), 'coverage': -float(f.get('inspection_age_ratio') or 0), 'efficiency': -float(f.get('shift_s') or 0), 'thermal_disagreement': -float(f.get('thermal_disagreement') or 0), 'novelty': -sum(v is None for v in f.values())}; return tuple(values[k] for k in diagnostics) + (-float(f.get('direct_age_s') or 0), c['decision']['time_s'], c['decision']['task_public_id'])
        primary, contrasts = [], []
        for kind in KINDS:
            chosen = [min(pool, key=order) for flag in (True, False) if (pool := [c for c in cards if c['measurement']['kind'] == kind and c['measurement']['observed_alarm'] is flag])]
            if chosen: primary.append(chosen[0]); contrasts.extend(chosen[1:])
        return primary + contrasts

    def cards(self, result, diagnostics):
        key = result['key'], tuple(diagnostics)
        if key not in self.feedback:
            cards = []
            for row in result['rows']:
                raw = self.read(self.local(row['record_path']))
                for card in raw['causal_experience_cards']: cards.append({'decision': {k: card['decision'][k] for k in DECISION_FIELDS if k in card['decision']}, 'measurement': {k: card['measurement'][k] for k in MEASUREMENT_FIELDS}, 'elapsed_to_actual_measurement_s': card['elapsed_to_actual_measurement_s']})
            self.feedback[key] = self.retrieve(cards, diagnostics)
        return self.feedback[key]

    def context_for(self, initial, history, protocol, prior):
        neutral = [{'skill': v['skill']} if v['skill'] is not None else {'schema_error': 'previous output was not a valid TimingSkill'} for v in history[-2:]]; value = {'initial_skill': self.initial, 'proposal_history': neutral, 'diagnostics': protocol['protocol']['diagnostics'], 'instruction': protocol['protocol']['instruction'], 'experience_cards': [], 'acceptance_feedback': []}
        if prior: return value
        pairs = list(enumerate(history)); memory = [] if protocol['protocol']['memory'] == 'none' else pairs[-2:] if protocol['protocol']['memory'] == 'recent' else [(i, v) for i, v in pairs if v['accepted']][-1:] + [(i, v) for i, v in pairs if not v['accepted']][-1:]
        cards = self.cards(initial, value['diagnostics']) + [c for _, v in memory if v.get('result') for c in self.cards(v['result'], value['diagnostics'])]; value['experience_cards'] = self.retrieve(cards, value['diagnostics']); value['acceptance_feedback'] = [{'candidate_id': hashlib.sha256(json.dumps(v['skill'], sort_keys=True).encode()).hexdigest()[:16], 'previous_proposal_offset': len(history) - i, 'accepted': v['accepted']} for i, v in memory if v['skill'] is not None]; return value

    def book(self, area, cap):
        starts = sorted(area.glob('call-*.start.json')); need(not list(area.glob('*.failed.json')) and len(starts) == len(list(area.glob('call-*.result.json'))) <= cap, 'Missing/failed actual model receipts'); book = []
        startup, loaded = self.read(area / 'model-startup.json'), self.read(area / 'model-loaded.json'); need('V100S' in startup['GPU'] and startup['total_GPU_bytes'] >= 32000000000 and loaded['frozen_manifest_SHA256'] == loaded['model_verified']['manifest_sha256'] == sha(self.identity['bound_inputs']['Qwen_manifest']), 'Frozen actual model/device mismatch')
        for i, path in enumerate(starts):
            need(path.name == f'call-{i:02d}.start.json', 'Noncontiguous call receipts'); start = self.read(path); result = self.read(path.with_name(path.name.replace('.start.', '.result.'))); need(start['thinking'] is False and start['output_cap'] == 1024 and 0 < start['input_tokens'] <= 6144 and start['input_tokens'] == result['input_tokens'] and 0 <= result['output_tokens'] <= 1024 and result['prompt_SHA256'] == digest(start['prompt']) and start['dropped_last_cards'] == result['dropped_last_cards'], 'Call identity/token contract'); no_hidden(start['prompt']); self.jobs.add(start['reservation']['job_id']); self.reservations['model_calls'].append(start['reservation']); need(startup['job_id'] == start['reservation']['job_id'], 'Model allocation mismatch'); book.append({'start': start, 'result': result, 'used': False})
        self.call_count += len(book); return book

    def claim_call(self, book, seed, response, context, meta=False):
        matches = [v for v in book if not v['used'] and v['start']['seed'] == seed and encoded(v['result']) == encoded(response)]; need(matches, 'Proposal not from a unique charged actual call'); call = matches[0]; call['used'] = True; prompt = json.loads(json.dumps(self.templates['meta_empty_context_template' if meta else 'task_empty_context_template'])); prompt['completed_causal_experience' if meta else 'data'] = context; cards = context['experience_cards']; dropped = call['start']['dropped_last_cards']; need(0 <= dropped <= len(cards), 'Invalid token truncation'); prompt['completed_causal_experience' if meta else 'data']['experience_cards'] = cards[:len(cards) - dropped] if dropped else cards
        need(encoded(prompt) == encoded(call['start']['prompt']) and json.dumps(call['start']['prompt'], ensure_ascii=False) in call['start']['rendered_prompt'], 'Causal treatment/prompt mismatch'); return call['start']['prompt']

    def search(self, value, store, area, seeds, book, prior=False, simple=False):
        protocol = value['protocol']; initial = self.result(store, self.refs['initial_E1_skill']); best = initial; history = []; seen = {self.skill_key(self.initial, protocol['dedup'])}; need(value['slots'] == len(value['entries']) == len(seeds) and value['prior_only'] == prior, 'Wrong search opportunity count')
        need(encoded(value['initial_policy']) == encoded(initial['policy']) and len(list(area.glob('slot-*.attempt.json'))) == len(list(area.glob('slot-*.complete.json'))) == len(seeds), 'Changed/extra search opportunities')
        for i, (entry, seed) in enumerate(zip(value['entries'], seeds)):
            attempt = self.read(area / f'slot-{i:02d}.attempt.json'); need(entry['generation_seed'] == attempt['generation_seed'] == seed and attempt['initial_parent'] == self.initial and attempt['one_opportunity_consumed'] and encoded(entry) == encoded(self.read(area / f'slot-{i:02d}.complete.json')), 'Search slot receipt mismatch')
            if simple or protocol['generator'] == 'random_program': skill = self.sample(seed)
            else:
                self.claim_call(book, seed, entry['generation'], self.context_for(initial, history, protocol, prior))
                try: skill = self.reply(entry['generation']['raw_response'])
                except (ValueError, TypeError, KeyError, OverflowError): skill = None
            need(entry['slot'] == attempt['slot'] == i and attempt['prior_only'] == prior and attempt['generator'] == ('random_program' if simple else protocol['generator']) and entry['valid'] == (skill is not None), 'False schema/generator verdict')
            if skill is None: need(not entry['accepted'] and entry['reason'] == 'invalid_schema', 'Accepted/mislabelled malformed output'); history.append({'skill': None, 'accepted': False}); continue
            need(encoded(skill) == encoded(entry['skill']) and entry['candidate_id'] == hashlib.sha256(json.dumps(skill, sort_keys=True).encode()).hexdigest()[:16], 'Edited candidate skill'); key = self.skill_key(skill, protocol['dedup']); accepted = False; result = None
            if key in seen: need(entry['reason'] == 'duplicate_consumes_slot' and not entry['accepted'], 'Duplicate retry/retention')
            else:
                seen.add(key); result = self.result(store, {'executor': 'timing-skill-v1', 'ranker': 'urgency', 'skill': skill}, True); accepted = paired(result['statistics'], initial['statistics']) and better(result['statistics'], best['statistics']); need(entry['evaluation_key'] == result['key'] and equal(entry['statistics'], result['statistics']) and entry['accepted'] == accepted, 'Independent retention mismatch'); behavior = self.behavior(result, initial); need(equal(entry['behavior'], behavior), 'Behavior claim mismatch')
                if accepted: best = result
            history.append({'skill': skill, 'accepted': accepted, 'result': result})
        published = [{'skill': v['skill'], 'accepted': v['accepted'], 'evaluation_key': v['result']['key'] if v.get('result') else None} if v['skill'] is not None else {'skill': None, 'accepted': False, 'schema_error': True} for v in history]
        need(value['valid_proposals'] == sum(v['skill'] is not None for v in history) and encoded(published) == encoded(value['history']) and encoded(best['policy']) == encoded(value['selected_policy']) and best['key'] == value['selected_evaluation_key'] and equal(best['statistics'], value['selected_statistics']), 'Search winner/history/count mismatch'); return self.context_for(initial, history, protocol, prior)

    def sample(self, seed):
        rng = np.random.default_rng(seed); space = self.study['skill_space']; fields = sorted(set(space['numeric_domains']) | set(space['boolean_fields']) | {'kind', 'evidence_margin'}); rules = []
        for _ in range(int(rng.integers(0, 4))):
            atoms = []
            for _ in range(int(rng.integers(1, 7))):
                f = fields[int(rng.integers(len(fields)))]; op, values = ('positive', space['special_margin_coefficients']) if f == 'evidence_margin' else (('eq', 'is_missing')[int(rng.integers(2))], (False, True)) if f in space['boolean_fields'] else (('eq', 'is_missing')[int(rng.integers(2))], KINDS) if f == 'kind' else (('ge', 'gt', 'le', 'lt', 'is_missing')[int(rng.integers(5))], space['numeric_domains'][f] + (['kind_threshold'] if f == 'professional_signal' else [])); atoms.append({'field': f, 'op': op, 'value': None if op == 'is_missing' else values[int(rng.integers(len(values)))]})
            rules.append({'all': atoms})
        return self.skill({'schema_version': 'timing-skill-v1', 'rules': rules, 'default_action': 'CONTINUE_TOUR'})

    def skill_key(self, skill, mode):
        value = json.loads(json.dumps(skill))
        if mode == 'canonical_rule_order':
            rules = []
            for row in value['rules']:
                atoms = [v for i, v in enumerate(row['all']) if v not in row['all'][:i]]
                if {'all': atoms} not in rules: rules.append({'all': atoms})
            value['rules'] = rules
        return digest(value)

    def behavior(self, candidate, initial):
        key = initial['key']
        if key not in self.probes:
            values = []
            for row in initial['rows']:
                path = self.local(row['probe_path']); need(sha(path) == row['probe_SHA256'], 'Changed public probe'); values.extend(strict_json(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines())
            self.probes[key] = values
        different = sum(action(candidate['policy']['skill'], v) != action(self.initial, v) for v in self.probes[key]); old = {encoded(v['specification']): v['action_SHA256'] for v in initial['rows']}; changed = sum(v['action_SHA256'] != old[encoded(v['specification'])] for v in candidate['rows']); return {'same_input_rows': len(self.probes[key]), 'decision_differences': different, 'actual_trajectory_differences': changed, 'behavior_changed': different > 0 and changed > 0}

    def check_queries(self, store): need(collections.Counter(store['queries']) == collections.Counter(encoded(v['policy']) for v in store['ledger']['requests']), 'Unexpected/omitted evaluation request')
    def cohort(self, phase, batch=None): return [v['specification'] for v in self.cases['future'] if v['partition'] == phase and (batch is None or v['specification']['seed'] in ((36140, 36141) if batch == 0 else (36142, 36143)))]

    def run(self, phase):
        self.metadata(); self.refs = {'Tour': {'executor': 'timing-skill-v1', 'ranker': 'urgency', 'skill': dict(self.initial, rules=[])}, 'human_professional': {'executor': 'original-human-v1', 'ranker': 'rules', 'skill': None}, 'initial_E1_skill': {'executor': 'timing-skill-v1', 'ranker': 'urgency', 'skill': self.initial}, 'manual_information': {'executor': 'timing-skill-v1', 'ranker': 'urgency', 'skill': self.study['skill_space']['manual_information_skill']}}
        return self.f1() if phase == 'F1' else self.f2()

    def f1(self):
        self.launch('F1_search'); base = self.root / 'F1'; search = self.receipt(base / 'search/complete.json'); need(len(search['mappings']) == 3, 'Exactly three F1 mappings'); self.seals(search, [base / f'search/mapping-{g}.json' for g in self.study['F1']['generation_repeats']]); store = self.store(base / 'search/evaluations', [v['specification'] for v in self.cases['known_development']], 680); self.result(store, self.refs['initial_E1_skill'], True)
        for policy in self.refs.values(): self.result(store, policy, True)
        book = self.book(base / 'search/calls', 18); need(len(book) == 18, 'Completed F1 requires its 18 fixed model attempts'); changed = []
        for mapping, generation in zip(search['mappings'], self.study['F1']['generation_repeats']):
            area = base / f'search/generation-{generation}'; need(mapping['generation_seed'] == generation and encoded(mapping) == encoded(self.read(base / f'search/mapping-{generation}.json')), 'Detached F1 mapping')
            for name in ('simple', 'prior', 'experienced'):
                need(encoded(mapping[name]['protocol']) == encoded(self.templates['meta_empty_context_template']['current_protocol']), 'F1 improver changed'); self.search(mapping[name], store, area / name, [generation * 10000 + (200 if name == 'simple' else 100) + s for s in range(3)], book, prior=name == 'prior', simple=name == 'simple')
            methods = dict(self.refs, **{name: mapping[key]['selected_policy'] for name, key in (('simple3', 'simple'), ('prior3', 'prior'), ('experienced3', 'experienced'))}); methods['experienced3_PPOtool'] = dict(methods['experienced3'], ranker='ppo'); need(encoded(methods) == encoded(mapping['methods']), 'F1 reference/tool reselection'); self.result(store, methods['experienced3_PPOtool'], True); changed.append(mapping['prior']['valid_proposals'] >= 2 and mapping['experienced']['valid_proposals'] >= 2 and sum(e.get('behavior', {}).get('behavior_changed', False) for e in mapping['experienced']['entries']) >= 1)
        self.check_queries(store); need(all(v['used'] for v in book) and search['model_calls'] == len(book) and equal(search['evaluation_ledger'], store['ledger']), 'Unassigned/incorrect model/evaluation count'); roles = {c['measurement']['kind'] for v in book for c in v['start']['prompt']['data']['experience_cards'] if c['elapsed_to_actual_measurement_s'] > 0}; mechanism = all(changed) and roles == set(KINDS); need(mechanism == search['mechanism_gate']['passed'], 'F2 mechanism verdict mismatch')
        validation = self.confirm_f1(base / 'validation', search, False); final = self.confirm_f1(base / 'final', search, True) if validation['final_open'] else None
        if final is None: need(not (base / 'final').exists() and not (self.root / 'future-sources/F1_final').exists(), 'Closed F1 final was opened')
        closed = None
        if not mechanism:
            need(not (self.root / 'F2').exists() and not list((self.root / 'future-sources').glob('F2*')) and not list((self.root / 'launches').glob('F2*.json')), 'Mechanism-closed F2 was entered'); closed = {'status': 'verified_not_executed', 'F2_executed': False, 'reason': 'Registered mechanism gate false; no F2 task/meta result exists.'}
        return {'status': 'audited_completed_F1', 'mechanism_gate': mechanism, 'validation': {k: validation[k] for k in ('TaskGO', 'ExperienceGO', 'final_open')}, 'final_opened': final is not None, 'confirmed_TaskGO': final['confirmed_TaskGO'] if final else False, 'confirmed_ExperienceGO': final['confirmed_ExperienceGO'] if final else False, 'conditional_F2': closed}

    def confirm_f1(self, area, search, final):
        self.launch('F1_final' if final else 'F1_validation'); value = self.receipt(area / 'complete.json'); self.seals(value, [self.root / 'F1/search/complete.json'] + [self.root / f'F1/search/mapping-{g}.json' for g in self.study['F1']['generation_repeats']] + ([self.root / 'F1/validation/complete.json'] if final else [])); store = self.store(area / 'evaluations', self.cohort('F1_final' if final else 'F1_validation'), 640 if final else 320); flags = []
        for mapping, reported in zip(search['mappings'], value['comparisons']):
            stats = {name: self.result(store, policy, True)['statistics'] for name, policy in mapping['methods'].items()}; need(mapping['generation_seed'] == reported['generation_seed'] and equal(stats, reported['statistics']), 'F1 confirmation statistics mismatch'); task = all(gain(stats['experienced3'], stats[n]) for n in ('Tour', 'human_professional', 'initial_E1_skill', 'manual_information', 'simple3')); experience = gain(stats['experienced3'], stats['prior3']) and all(nonregression(stats['experienced3'], stats[n]) for n in self.refs); need((task, experience) == (reported['TaskGO'], reported['ExperienceGO']), 'F1 GO mismatch'); flags.append((task, experience))
        self.check_queries(store); task, experience = all(v[0] for v in flags), all(v[1] for v in flags); need(equal(value['evaluation_ledger'], store['ledger']) and len(flags) == 3 and (task, experience) == (value['TaskGO'], value['ExperienceGO']), 'Incomplete F1 comparisons/evaluation accounting')
        if final:
            val = self.receipt(self.root / 'F1/validation/complete.json'); need(val['final_open'] and value['confirmed_TaskGO'] == (task and val['TaskGO']) and value['confirmed_ExperienceGO'] == (experience and val['ExperienceGO']) and value['registered_success'] == (value['confirmed_TaskGO'] or value['confirmed_ExperienceGO']), 'Unreplicated F1 claim')
        else: need(value['final_open'] == (task or experience), 'Incorrect F1 final gate')
        return value

    def f2(self):
        first = self.receipt(self.root / 'F1/search/complete.json'); self.receipt(self.root / 'F1/validation/complete.json')
        if not first['mechanism_gate']['passed']:
            need(not (self.root / 'F2').exists() and not list((self.root / 'future-sources').glob('F2*')), 'Mechanism-closed F2 was entered'); return {'status': 'verified_not_executed', 'F2_executed': False}
        mappings = []
        for trial, generation in enumerate(self.study['F2']['generation_repeats']):
            self.launch('F2_adapt', trial=trial); area = self.root / f'F2/trial-{trial}/adaptation'; value = self.receipt(area / 'complete.json'); meta = value['meta']; need(value['trial'] == trial and value['generation_seed'] == generation and encoded(meta) == encoded(self.read(area / 'meta-complete.json')) and encoded(meta['incumbent']) == encoded(self.templates['meta_empty_context_template']['current_protocol']) and encoded(meta['old']['protocol']) == encoded(meta['incumbent']), 'Detached meta receipt/incumbent'); book = self.book(area / 'calls', 19); store = self.store(area / 'meta-evaluations', self.cohort('F2_meta'), 100); self.result(store, self.refs['initial_E1_skill'], True)
            context = self.search(meta['old'], store, area / 'old-inner', [generation * 10000 + 1000 + s for s in range(2)], book); self.claim_call(book, generation * 10000 + 2000, meta['response'], context, True)
            try: protocol = self.reply(meta['response']['raw_response'], True); valid = True
            except (ValueError, TypeError, KeyError, OverflowError): protocol = meta['incumbent']; valid = False
            need(encoded(protocol) == encoded(meta['candidate']) == encoded(meta['new']['protocol']) and valid == meta['proposal_valid'], 'False meta schema/fallback'); self.search(meta['new'], store, area / 'candidate-inner', [generation * 10000 + 1000 + s for s in range(2)], book); retained = valid and encoded(protocol) != encoded(meta['incumbent']) and better(meta['new']['selected_statistics'], meta['old']['selected_statistics']); need(retained == meta['retained'] and encoded(meta['transferred_protocol']) == encoded(protocol if retained else meta['incumbent']) and meta['task_reset'] == {'initial_skill': self.initial, 'history': [], 'candidate_archive': [], 'outcomes': []} and equal(meta['ledger'], store['ledger']), 'Meta retention/reset/accounting mismatch'); self.check_queries(store)
            store = self.store(area / 'adapt-evaluations', self.cohort('F2_adapt'), 540); self.result(store, self.refs['initial_E1_skill'], True)
            for policy in self.refs.values(): self.result(store, policy, True)
            for key, folder, count, offset, prior, simple in (('simple', 'simple8', 8, 5000, False, True), ('fixed8', 'fixed8', 8, 3000, False, False), ('prior', 'prior3', 3, 4000, True, False), ('meta3', 'meta3', 3, 3000, False, False)):
                need(encoded(value[key]['protocol']) == encoded(meta['transferred_protocol'] if key == 'meta3' else meta['incumbent']), 'Task protocol transfer mismatch'); self.search(value[key], store, area / folder, [generation * 10000 + offset + s for s in range(count)], book, prior, simple)
            prefix = value['fixed3']; need(encoded(prefix) == encoded(self.read(area / 'fixed-prefix3.json')) and prefix['entries'] == value['fixed8']['entries'][:3], 'Fixed-prefix rescue'); selected = self.refs['initial_E1_skill']
            for entry in prefix['entries']:
                if entry['accepted']: selected = {'executor': 'timing-skill-v1', 'ranker': 'urgency', 'skill': entry['skill']}
            need(encoded(prefix['selected_policy']) == encoded(selected) and encoded(prefix['history']) == encoded(value['fixed8']['history'][:3]) and prefix['selected_evaluation_key'] == self.result(store, selected)['key'] and equal(prefix['selected_statistics'], self.result(store, selected)['statistics']), 'Wrong fixed3 prefix'); methods = dict(self.refs, simple8=value['simple']['selected_policy'], prior3=value['prior']['selected_policy'], fixed_experience3=selected, fixed_experience8=value['fixed8']['selected_policy'], meta3=value['meta3']['selected_policy'], meta3_PPOtool=dict(value['meta3']['selected_policy'], ranker='ppo')); need(encoded(methods) == encoded(value['methods']), 'F2 method/tool mapping mismatch'); self.result(store, methods['meta3_PPOtool'], True); self.check_queries(store); need(all(v['used'] for v in book) and value['model_calls'] == len(book) and equal(value['adapt_ledger'], store['ledger']), 'Unassigned/incorrect F2 model/evaluation count'); mappings.append(value)
        validation = self.confirm_f2(self.root / 'F2/validation', mappings); final = None
        if validation['MetaGO']:
            batches = [self.confirm_f2(self.root / f'F2/final-batch-{b}', mappings, b) for b in (0, 1)]; final = self.receipt(self.root / 'F2/final-complete.json'); flags = []
            for trial, mapping in enumerate(mappings):
                stats = {}
                for name, policy in mapping['methods'].items():
                    rows = [row for b in batches for result in b['ledgers'][trial]['results'] if encoded(result['policy']) == encoded(policy) for row in result['rows']]; cases = {encoded(v['specification']): v for result in batches[0]['ledgers'][trial]['results'] + batches[1]['ledgers'][trial]['results'] for v in self.store_cases_from_result(result)}; stats[name] = self.aggregate(rows, self.cohort('F2_final'), cases, policy)
                need(equal(stats, final['comparisons'][trial]['statistics']), 'Four-seed final merge mismatch'); flags.append(gain(stats['meta3'], stats['fixed_experience3']) and all(nonregression(stats['meta3'], stats[n]) for n in (*self.refs, 'simple8')))
            need(final['MetaGO'] == all(flags) and final['original_environment_evaluations_added_by_merge'] == 0, 'False final meta claim')
        else: need(not list((self.root / 'F2').glob('final*')) and not (self.root / 'future-sources/F2_final').exists(), 'Closed F2 final opened')
        return {'status': 'audited_completed_F2', 'F2_executed': True, 'validation_MetaGO': validation['MetaGO'], 'final_opened': final is not None, 'confirmed_MetaGO': bool(final and final['MetaGO'])}

    def store_cases_from_result(self, result): return self.read(self.local(result['rows'][0]['record_path']).parent / 'started.json')['cases']
    def confirm_f2(self, area, mappings, batch=None):
        self.launch('F2_final' if batch is not None else 'F2_validation', batch=batch if batch is not None else None); value = self.receipt(area / 'complete.json'); self.seals(value, [self.root / f'F2/trial-{trial}/adaptation/complete.json' for trial in range(3)]); flags = []
        for mapping, reported in zip(mappings, value['comparisons']):
            store = self.store(area / f'trial-{mapping["trial"]}', self.cohort('F2_final' if batch is not None else 'F2_validation', batch), 200); stats = {name: self.result(store, policy, True)['statistics'] for name, policy in mapping['methods'].items()}; need((mapping['trial'], mapping['generation_seed']) == (reported['trial'], reported['generation_seed']) and equal(stats, reported['statistics']) and equal(value['ledgers'][mapping['trial']], store['ledger']), 'F2 confirmation statistics/ledger mismatch'); passed = gain(stats['meta3'], stats['fixed_experience3']) and all(nonregression(stats['meta3'], stats[n]) for n in (*self.refs, 'simple8')); need(passed == reported['MetaGO'], 'F2 point GO mismatch'); flags.append(passed); self.check_queries(store)
        need(len(flags) == 3 and value['MetaGO'] == all(flags), 'Incomplete F2 comparisons'); return value

    def costs(self, accounting):
        for kind, initial, cap in (('episodes', 20, 6000), ('model_calls', 0, 75)):
            path = self.root / 'cost' / f'{kind}-reservations.jsonl'; rows = [strict_json(v) for v in path.read_text().splitlines()] if path.exists() else []; counter = self.read(self.root / 'cost' / f'{kind}-counter.json') if rows else {'reserved': initial}; total = initial
            for row in rows: need(row['first'] == total and row['amount'] > 0, 'Noncontiguous global reservation journal'); total += row['amount']
            journal = {encoded(v) for v in rows}; need(total == counter['reserved'] <= cap and all(encoded(v) in journal for v in self.reservations[kind]), 'Global budget counter/reservation mismatch')
        jobs = list(csv.DictReader(Path(accounting).read_text().splitlines(), delimiter='|')); ids = {v['JobID'] for v in jobs}; need(len(ids) == len(jobs) and self.jobs <= ids and all(v['State'] in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT') for v in jobs), 'Accounting prefix missing completed producing allocations'); need(all(v['State'] == 'COMPLETED' and v['ExitCode'] == '0:0' for v in jobs if v['JobID'] in self.jobs), 'Producing allocation failed')
        events = []
        for job in jobs:
            if int(job['ElapsedRaw']) > 0: events.extend([(datetime.datetime.fromisoformat(job['Start']), 1), (datetime.datetime.fromisoformat(job['End']), -1)])
            if job['JobID'] in self.launches:
                gpu_mode = self.launches[job['JobID']]['mode'] in ('F1_search', 'F2_adapt'); need(int(job['AllocCPUS']) == (4 if gpu_mode else 2) and job['Partition'] == ('4gpu' if gpu_mode else 'cu') and int(job['ElapsedRaw']) <= (5400 if gpu_mode else 3600), 'Mode/allocation mismatch')
        active = peak = 0
        for _, change in sorted(events, key=lambda v: (v[0], v[1])): active += change; peak = max(peak, active)
        cpu = sum(int(v['ElapsedRaw']) * int(v['AllocCPUS']) / 60 for v in jobs); gpu = [v for v in jobs if 'gres/gpu=' in v['AllocTRES']]; seconds = sum(int(v['ElapsedRaw']) for v in gpu); need(cpu <= 2400 and len(gpu) <= 4 and seconds <= 21600 and peak <= 2 and all(int(re.search(r'(?:^|,)gres/gpu=(\d+)', v['AllocTRES']).group(1)) == 1 and int(v['ElapsedRaw']) <= 5400 for v in gpu), 'Allocation prefix exceeds caps'); need(self.jobs <= set(self.launches), 'Producing job lacks a held-launch proof'); self.inputs[str(Path(accounting).resolve())] = sha(accounting); storage = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file()); need(storage <= 6000000000, 'New-storage cap exceeded'); return {'scope': 'Completed-allocation prefix supplied by caller; excludes this audit allocation until the final post-audit ledger refresh.', 'CPU_minutes': cpu, 'GPU_allocations': len(gpu), 'GPU_seconds': seconds, 'max_concurrent_allocations': peak, 'storage_bytes_before_report': storage, 'required_data_job_ids': sorted(self.jobs), 'whole_campaign_post_audit_costs_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run-root', type=Path, required=True); parser.add_argument('--source-root', type=Path, required=True); parser.add_argument('--phase', choices=('F1', 'F2'), required=True); parser.add_argument('--accounting', type=Path); parser.add_argument('--output', type=Path); args = parser.parse_args()
    audit = Audit(args.run_root, args.source_root); result = audit.run(args.phase); costs = audit.costs(args.accounting or args.run_root / 'resource-sacct.psv'); report = {'passed': True, 'phase': args.phase, 'result': result, 'actual_unique_raw_records_checked': audit.episodes, 'actual_causal_cards_checked': audit.cards_checked, 'actual_model_call_receipts_checked': audit.call_count, 'accounting_prefix': costs, 'input_SHA256': audit.inputs, 'auditor_SHA256': sha(__file__), 'new_model_calls': 0, 'new_environment_episodes': 0, 'new_physical_processes': 0, 'whole_neural_or_physical_rerun': False, 'physical_scope': 'Existing stored arrays and registered counter-based reference noise are checked; the simulator and policy are not rerun.', 'token_cost_scope': 'Source-bound recorded token counts; no independent tokenizer or model execution.', 'scientific_scope': 'Only the preregistered paired pilot decisions; no independent-room or population/general-RSI claim.', 'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}; output = args.output or args.run_root / f'{args.phase}-independent-actual-audit.json'
    need(not output.exists(), 'Audit reports are immutable; choose a new output path'); text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'; need(not output.resolve().is_relative_to(audit.root) or costs['storage_bytes_before_report'] + len(text.encode()) <= 6000000000, 'Report would exceed storage cap')
    with output.open('x', encoding='utf-8', newline='\n') as stream: stream.write(text)
    need(strict_json(output.read_text()) == report, 'Report read-back failed'); print('M12F_ACTUAL_AUDIT_PASSED', args.phase, audit.episodes, audit.call_count, sha(output))


if __name__ == '__main__':
    try: main()
    except Exception as error: print('M12F_ACTUAL_AUDIT_FAILED:', error, file=sys.stderr); sys.exit(1)
