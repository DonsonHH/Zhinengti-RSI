"""Editable diagnostic/retrieval instructions with a fixed validator contract."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from .improver_task import PatrolProgram, TypedRule, DOMAINS, ACTIONS, SIGNAL_ROLES

TOOLS = ('coverage', 'professional_evidence', 'efficiency', 'thermal_disagreement', 'novelty')
MEMORY = ('none', 'recent', 'best_and_failures')


@dataclass(frozen=True)
class Improver:
    diagnostics: tuple[str, ...] = ('coverage', 'professional_evidence', 'efficiency')
    memory: str = 'recent'
    instruction: str = 'Use typed measured evidence to propose one distinct program that may confirm anomalies earlier without unnecessary detours. Return to the protected tour when extra checking lacks evidence.'

    def __post_init__(self):
        if not isinstance(self.diagnostics, tuple) or not 1 <= len(self.diagnostics) <= 3 or len(set(self.diagnostics)) != len(self.diagnostics):
            raise ValueError('Choose1..3 distinct diagnostics')
        if any(d not in TOOLS for d in self.diagnostics) or self.memory not in MEMORY:
            raise ValueError('Unknown diagnostic or memory mode')
        if not isinstance(self.instruction, str) or not 1 <= len(self.instruction) <= 500:
            raise ValueError('Instruction size/type mismatch')

    def to_dict(self):
        value = asdict(self)
        value['diagnostics'] = list(self.diagnostics)
        return value

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {'diagnostics', 'memory', 'instruction'} or not isinstance(value['diagnostics'], list):
            raise ValueError('Protocol keys mismatch')
        return cls(tuple(value['diagnostics']), value['memory'], value['instruction'])

    @property
    def version_id(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


def diagnostics(record):
    public, events = record['public_selection'], record['inspection_events']
    result = {'coverage': {k: public[k] for k in ('coverage', 'violating_tasks', 'weighted_age_ratio', 'first_alert_tasks')},
        'efficiency': {k: public[k] for k in ('travel_m', 'service_s', 'extra_checks', 'estimator_calls', 'dispatches', 'different_from_actor')}}
    role_info = {}
    schemas = {'thermal': ('temperature_c', 'degC'), 'electrical': ('contact_temperature_c', 'degC'),
        'cooling': ('pod_flow_kg_s', 'kg/s'), 'control': ('controller_sensor_error_c', 'degC')}
    for role, (field, unit) in schemas.items():
        checks = [e for e in events if e['kind'] == role]
        measured = [e['measurement'][field] for e in checks]
        transitions = Counter(e['measurement_transition'] for e in checks)
        role_info[role] = {'checks': len(checks), 'distinct_checked_tasks': len({e['task'] for e in checks}),
            'distinct_measured_alarm_tasks': len({e['task'] for e in checks if e['alarm']}),
            'new_measured_alerts': transitions['new_measured_alert'],
            'persistent_rechecks': transitions['persistent_recheck'], 'recovery_confirmations': transitions['recovery_confirmation'],
            'measurement_field': field, 'unit': unit, 'mean_measured_value': float(np.mean(measured)) if measured else None,
            'largest_public_completion_gap_s': max(g['maximum_gap_including_initial_and_terminal_s']
                for g in record['exact_task_gaps'] if g['task_id'].startswith(role + ':'))}
    result['professional_evidence'] = role_info
    values = record['thermal_tool_observations']
    result['thermal_disagreement'] = {'meaning': 'TemperatureCVAE versus nominal temperature; thermal units only, not other professions',
        'mean_abs_difference_degC': float(np.mean([r['mean_abs_difference_degC'] for r in values])),
        'peak_abs_difference_degC': max(r['peak_abs_difference_degC'] for r in values)}
    return result


def combine_diagnostics(records):
    views = [diagnostics(r) for r in records]
    def merge(values):
        first = values[0]
        if isinstance(first, dict):
            return {k: merge([v[k] for v in values]) for k in first}
        if isinstance(first, str) or first is None:
            if not all(v == first for v in values):
                raise ValueError('Inconsistent diagnostic schema')
            return first
        return float(np.mean(values))
    return merge(views)


def validation_summary(rows):
    count = sum(r['metrics']['anomaly_segments'] for r in rows)
    weight = sum(r['event_weight_sum'] for r in rows)
    return {'cases': len(rows), 'anomaly_segments': count,
        'missed_segments': sum(r['metrics']['missed_segments'] for r in rows),
        'weighted_censored_delay_s': sum(r['weighted_delay_sum'] for r in rows) / weight if weight else None,
        'violating_tasks': sum(r['metrics']['tasks_violating_exact_gap'] for r in rows),
        'coverage': float(np.mean([r['metrics']['final_professional_coverage'] for r in rows])),
        'mean_travel_m': float(np.mean([r['metrics']['travel_m'] for r in rows])),
        'scope': 'Development-only annotated evaluator; not raw labels supplied to the language model'}


def validation_rank(value):
    return tuple(round(float(v), 8) for v in (value['violating_tasks'], 1 - value['coverage'],
        value['missed_segments'], value['weighted_censored_delay_s'] if value['weighted_censored_delay_s'] is not None else 0., value['mean_travel_m']))


def retrieved(history, mode):
    if mode == 'none':
        return []
    if mode == 'recent':
        return history[-2:]
    accepted = [r for r in history if r.get('accepted')]
    failed = [r for r in history if not r.get('accepted')]
    return (accepted[-1:] if accepted else []) + failed[-1:]


def program_prompt(program, views, history, protocol):
    tools = []
    for name in protocol.diagnostics:
        value = {'prior_candidate_count': len(history), 'duplicates': sum(r.get('reason') == 'duplicate' for r in history),
            'seen_versions': [r.get('candidate_id') for r in history if r.get('candidate_id')]} if name == 'novelty' else views[name]
        tools.append({'tool': name, 'result': value})
    return {'task': 'propose_patrol_program', 'current_workflow': program.to_dict(),
        'improver_protocol': protocol.to_dict(), 'public_diagnostics': tools,
        'retrieved_candidates': retrieved(history, protocol.memory),
        'response_keys': ['diagnosis', 'workflow', 'expected_effect'],
        'response_types': {'diagnosis': 'ONE explanatory string, never a list', 'workflow': 'object', 'expected_effect': 'ONE string'},
        'maximum_rule_count': 3,
        'rule_count_policy': 'Choose at most three conditions, not one per profession. Four rules is invalid.',
        'workflow_keys': ['refresh_s', 'cooldown_s', 'fallback', 'rules'],
        'domains': {'refresh_s': [30, 60, 120], 'cooldown_s': [0, 30, 60, 120], 'fallback': ['tour', 'ppo'],
            'rules': '0..3 ordered rules; each has exactly signal, threshold, action',
            'signal': list(DOMAINS), 'threshold': 'ONE NUMBER from the signal threshold domain',
            'threshold_domains': DOMAINS, 'action': list(ACTIONS)},
        'semantics': {'signal_profession': SIGNAL_ROLES,
            'thermal_excess': 'max(estimatedTemp-65,0)/20; degC scale; thermal only',
            'thermal_disagreement': 'abs(history-nominalTemp)/20; thermal only, not calibrated uncertainty',
            'contact_heat': 'max(measuredContactTemp-60,0)/30; electrical only, not supply availability',
            'flow_deficit': 'max(1-measuredFlow/33.6,0); cooling only',
            'verified_sensor_bias': 'Previously valid reference comparison above1degC; control only',
            'age': 'Known task age/public revisit period',
            'rules': 'First rule with a qualifying candidate; return_tour selects next tour item; other actions rank qualifying candidates.',
            'signal_action': 'Typed signal times public asset weight divided by completion cost',
            'fallback': 'tour returns to baseline; ppo uses the frozen trained actor if no rule matches.',
            'tool_use': 'refresh_s changes actual CVAE calls; cooldown applies only to optional detours.'},
        'constraints': ['Fixed qualification, complete return certificate, two detours per owned round and measured-task semantics.',
            'No changes to hidden evaluator, weights, data, costs or budgets; no source code.',
            'Raw truth, onset and test results are not provided. Development accept/rollback feedback is available.',
            'Do not repeat the current workflow. Return JSON only; diagnosis and expected_effect1..500characters.']}


def protocol_prompt(protocol, reports):
    return {'task': 'propose_improver_protocol', 'current_protocol': protocol.to_dict(),
        'prior_meta_round_reports': reports,
        'response_keys': ['diagnosis', 'protocol', 'expected_effect'],
        'response_types': {'diagnosis': 'ONE explanatory string, not the diagnostics list', 'protocol': 'object', 'expected_effect': 'ONE string'},
        'protocol_keys': ['diagnostics', 'memory', 'instruction'],
        'maximum_diagnostic_count': 3,
        'tool_replacement_rule': 'If the current list already has three tools, REMOVE or REPLACE a tool to add another. Never append a fourth tool.',
        'domains': {'diagnostics': '1..3 distinct tools, in order', 'tools': list(TOOLS),
            'memory': list(MEMORY), 'instruction': '1..500characters'},
        'objective': 'Improve how the fixed model proposes useful distinct patrol workflows from public observations.',
        'rules': ['Only tool selection/order, memory retrieval and proposal instruction can change.',
            'Candidate and incumbent are evaluated from the same task start with matched candidate seeds before retention.',
            'Only the protocol transfers to new tasks; weights, evaluator, task grammar and budgets stay fixed.',
            'Return JSON only. No raw simulator truth or final-test information is available.']}
