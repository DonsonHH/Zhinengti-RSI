"""M2: latched revisit tools, paired improvers and equal-slot random search."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .environment import PatrolEnv
from .experiment import RouteGreedy, room
from .revisit import PatrolSkill, RevisitSkill, trajectory_diagnostics


TOOLS = ('coverage', 'revisit', 'oscillation')
METRICS = ('target_coverage', 'mean_overdue_fraction', 'mean_normalized_weighted_idleness',
           'peak_unobserved_steps', 'max_unobserved_steps', 'never_observed_targets',
           'estimator_calls', 'prior_samples', 'skill_executor_fraction', 'total_movement_distance')


def splits():
    return {'development': [{'split': 'development', 'layout_seed': 6000+i,
                             'field_seed': 700000+i, 'noise': 0.0} for i in range(6)],
            'test': [{'split': 'm2_test', 'layout_seed': 7000+i, 'field_seed': 800000+i,
                      'noise': 0.0 if i < 12 else .03} for i in range(24)],
            'policy_seeds': [0, 1, 2], 'proposal_seeds': [71, 173, 269],
            'candidate_slots_per_method': 6, 'fixed_evaluation_deadline_steps': 24,
            'ranking': 'lexicographic: unseen fraction, overdue fraction, weighted idleness, prior samples',
            'input_token_cap': 4096, 'output_token_cap': 512,
            'methods': ['random_search', 'fixed_improver', 'updated_improver']}


def mean(rows):
    return {key: float(np.mean([r[key] for r in rows])) for key in METRICS}


def rank(metrics):
    return (round(1-metrics['target_coverage'], 6), round(metrics['mean_overdue_fraction'], 6),
            round(metrics['mean_normalized_weighted_idleness'], 6), metrics['prior_samples'])


@dataclass(frozen=True)
class DiagnosticPlan:
    tool: str = 'coverage'
    instruction: str = 'Diagnose missed targets; propose one testable tool-policy change.'

    def __post_init__(self):
        if self.tool not in TOOLS or not isinstance(self.instruction, str) or not 1 <= len(self.instruction) <= 600:
            raise ValueError('Invalid diagnostic plan')


def parse(content, parent):
    left, right = content.find('{'), content.rfind('}')
    if left < 0 or right < left:
        raise ValueError('No JSON proposal')
    obj = json.loads(content[left:right+1])
    if not isinstance(obj, dict) or not isinstance(obj.get('patch'), dict):
        raise ValueError('Proposal must contain a patch object')
    candidate = parent.proposed(obj['patch'])
    d = obj.get('next_diagnostic')
    if not isinstance(d, dict) or set(d) != {'tool', 'instruction'}:
        raise ValueError('next_diagnostic requires tool and instruction only')
    return candidate, DiagnosticPlan(**d), obj


def feedback(record):
    """Task feedback is shared fairly; proposed meta-instructions are excluded."""
    return {k: record[k] for k in ('generation', 'patch', 'accepted', 'metrics', 'error') if k in record}


def evaluate(estimator, policy, skill, cases, device, method='learned', trace=False):
    import torch
    rows, diagnostics = [], []
    for case in cases:
        env = PatrolEnv(room(case['layout_seed'], case['noise']), seed=case['field_seed'])
        recovery = RevisitSkill(env, case['layout_seed'])
        route = RouteGreedy(env) if method == 'route_greedy' else None
        ages = [env.raw_age[env.scenario.targets].copy()]
        positions = [env.positions.copy()]
        idleness, overdue = [], []
        calls = 0
        for t in range(96):
            if t % skill.estimator_refresh_interval == 0:
                # Align prior noise by case/time, instead of shifting the whole
                # random stream when the refresh schedule changes.
                torch.manual_seed(case['field_seed']*1000+t)
                with torch.no_grad():
                    x = torch.as_tensor(env.estimator_input()[None], dtype=torch.float32, device=device)
                    env.set_estimate(estimator.predict(x, skill.estimator_samples)[0, 0].cpu().numpy())
                calls += 1
            if route:
                scores = route.scores(env)
            else:
                with torch.no_grad():
                    scores = policy(torch.as_tensor(env.policy_observations(), device=device)).cpu().numpy()
                scores = recovery.scores(env, scores, skill.revisit_trigger)
            env.step(env.safe_actions(scores, skill.coordination_order))
            ages.append(env.raw_age[env.scenario.targets].copy())
            positions.append(env.positions.copy())
            idleness.append(env.evaluation()['normalized_weighted_idleness'])
            overdue.append(float((ages[-1] > 24).mean()))
        row = env.evaluation()
        row.update(case)
        row.update({'mean_normalized_weighted_idleness': float(np.mean(idleness)),
                    'mean_overdue_fraction': float(np.mean(overdue)),
                    'peak_unobserved_steps': int(np.max(ages)),
                    'never_observed_targets': int((~env.ever_measured[env.scenario.targets]).sum()),
                    'estimator_calls': calls, 'prior_samples': calls*skill.estimator_samples,
                    'skill_executor_fraction': recovery.executor_steps/(96*len(env.positions))})
        rows.append(row)
        views = {tool: trajectory_diagnostics(env, ages, positions, tool) for tool in TOOLS}
        diagnostics.append({'layout_seed': case['layout_seed'], **views})
        if trace:
            row['positions'] = [p.tolist() for p in positions]
            row['target_ages'] = [a.tolist() for a in ages]
            row['targets'] = np.argwhere(env.scenario.targets).tolist()
    return mean(rows), rows, diagnostics


def random_candidate(rng):
    while True:
        try:
            return PatrolSkill(estimator_samples=int(rng.choice([1, 2, 4])),
                               estimator_refresh_interval=int(rng.choice([1, 2, 4, 8])),
                               coordination_order=str(rng.choice(['q_priority', 'cyclic'])),
                               revisit_trigger=int(rng.choice([0, 12, 16, 24, 32])))
        except ValueError:
            pass


def build_prompt(parent, score, plan, diagnostics, history, generation):
    # This value contains freshly constructed copies, not mutable feedback lists.
    return json.loads(json.dumps({
        'generation': generation, 'current_skill': asdict(parent), 'metrics': score,
        'active_diagnostic_plan': asdict(plan),
        'diagnostic_tool_results': [d[plan.tool] | {'layout_seed': d['layout_seed']} for d in diagnostics[:2]],
        'history': history[-3:],
        'metric_meanings': {'target_coverage': 'higher is better; ever visited',
                            'mean_overdue_fraction': 'fraction of target-time pairs with age > fixed 24 steps; lower is better',
                            'mean_normalized_weighted_idleness': 'lower is better',
                            'skill_executor_fraction': 'fraction of DDQN executor commands replaced by the public revisit skill'},
        'fixed_acceptance': 'Prioritize coverage, then reduce overdue fraction, then weighted idleness, then prior samples. Do not change this order.',
        'available_tools': {
            'revisit_trigger': '0 disables; 12/16/24/32 activates latched shortest-path recovery at that raw age, until inspected',
            'estimator_samples': [1, 2, 4], 'estimator_refresh_interval': [1, 2, 4, 8],
            'coordination_order': ['q_priority', 'cyclic'],
            'budget': 'samples * ceil(96/refresh_interval) <= 192; 96 patrol steps; no weight changes'},
        'output_rules': [
            'Return JSON only. patch is an object whose keys are actual PatrolSkill field names, e.g. revisit_trigger or estimator_refresh_interval. Values must be allowed numeric values or the allowed coordination string.',
            'Never output placeholder keys changed_skill_field or allowed_value. Include changed fields only.',
            'next_diagnostic MUST be an object with exactly tool and instruction. It must NOT be a string. tool must be exactly coverage, revisit, or oscillation; instruction is a string <=600 characters.',
            'A larger revisit_trigger waits longer before recovery. A larger estimator_refresh_interval updates less often. These settings do not directly change exploration rewards.'],
        'JSON_shape_example_only': {'diagnosis': 'brief evidence-based hypothesis', 'patch': {'revisit_trigger': 12},
                                   'expected_effect': 'testable metric change',
                                   'next_diagnostic': {'tool': 'revisit', 'instruction': 'Inspect the largest raw ages before choosing the next patch.'}}}))


def run(output, model_directory, checkpoints, trial, slots=6, final_test=True):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from .models import ConditionalVAEUNet, SharedDuelingQ
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError('Allocated CUDA required')
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    contract = splits()
    policy_seed, proposal_seed = contract['policy_seeds'][trial//3], contract['proposal_seeds'][trial%3]
    estimator_path = checkpoints/'vae/best-vae.pt'
    policy_path = checkpoints/f'seed-{policy_seed}/best-policy.pt'
    estimator, policy = ConditionalVAEUNet().to(device).eval(), SharedDuelingQ().to(device).eval()
    estimator.load_state_dict(torch.load(estimator_path, map_location=device, weights_only=True)['model'])
    policy.load_state_dict(torch.load(policy_path, map_location=device, weights_only=True)['model'])
    (output/'contract.json').write_text(json.dumps(contract, indent=2))
    initial = PatrolSkill()
    initial_score, initial_rows, initial_diagnostics = evaluate(estimator, policy, initial, contract['development'], device)
    tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True, trust_remote_code=False)
    lm = AutoModelForCausalLM.from_pretrained(model_directory, torch_dtype=torch.float16, attn_implementation='eager',
                                             local_files_only=True, trust_remote_code=False).to(device).eval()
    records, retained = [], {}
    for method in contract['methods']:
        parent, score, diagnostics = initial, initial_score, initial_diagnostics
        plan, history = DiagnosticPlan(), []
        rng = np.random.default_rng(proposal_seed)
        for generation in range(slots):
            record = {'method': method, 'generation': generation, 'parent': asdict(parent), 'diagnostic_plan': asdict(plan),
                      'accepted': False, 'input_tokens': 0, 'output_tokens': 0, 'generation_seconds': 0.0}
            if method == 'random_search':
                candidate = random_candidate(rng)
                parsed = {'patch': asdict(candidate)}
                record['origin'] = 'search_baseline'
            else:
                user = build_prompt(parent, score, plan, diagnostics, history, generation)
                system = 'You improve a frozen VAE+multiagent DDQN through bounded tool policies. Use development evidence only. Return one JSON object. Do not modify weights, metrics, observations, test data or budgets.'
                rendered = tokenizer.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(user)}],
                                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
                inputs = tokenizer(rendered, return_tensors='pt').to(device)
                if inputs['input_ids'].numel() > contract['input_token_cap']:
                    raise RuntimeError('Prompt cap exceeded; protocol must be reviewed before running')
                torch.manual_seed(proposal_seed*100+generation)
                start = time.monotonic()
                with torch.no_grad():
                    ids = lm.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=.7, top_p=.8, top_k=20)
                answer = tokenizer.decode(ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                record.update({'origin': 'language_model', 'user_prompt': user, 'rendered_prompt': rendered, 'raw_response': answer,
                               'input_tokens': int(inputs['input_ids'].numel()), 'output_tokens': int(ids.shape[1]-inputs['input_ids'].shape[1]),
                               'generation_seconds': time.monotonic()-start})
                try:
                    candidate, next_plan, parsed = parse(answer, parent)
                    if method == 'updated_improver':
                        # Protocol changes alter the actual next diagnostic tool.
                        # Their benefit is tested via future candidate outcomes,
                        # not asserted by adopting the instruction itself.
                        plan = next_plan
                except (ValueError, TypeError) as error:
                    record['error'] = str(error)
                    history.append(feedback(record)); records.append(record)
                    (output/f'{method}-{generation}.json').write_text(json.dumps(record))
                    print(json.dumps({'trial':trial,'method':method,'generation':generation,'error':str(error)}),flush=True)
                    continue
            candidate_score, candidate_rows, candidate_diagnostics = evaluate(estimator, policy, candidate, contract['development'], device)
            accepted = rank(candidate_score) < rank(score)
            record.update({'patch': parsed['patch'], 'candidate': asdict(candidate), 'metrics': candidate_score,
                           'cases': candidate_rows, 'accepted': accepted, 'next_diagnostic_plan': asdict(plan)})
            if method != 'random_search':
                record['parsed_response'] = parsed
            if accepted:
                parent, score, diagnostics = candidate, candidate_score, candidate_diagnostics
            history.append(feedback(record)); records.append(record)
            (output/f'{method}-{generation}.json').write_text(json.dumps(record))
            print(json.dumps({'trial': trial, 'method': method, 'generation': generation, 'accepted': accepted,
                              'coverage': candidate_score['target_coverage'], 'overdue': candidate_score['mean_overdue_fraction']}), flush=True)
        retained[method] = {'skill': asdict(parent), 'development': score, 'final_diagnostic_plan': asdict(plan)}
    # All choices are immutable before any M2 test case is read.
    (output/'selection-frozen.json').write_text(json.dumps(retained, indent=2))
    tests = {}
    if final_test:
        controls = [('plain_ddqn', PatrolSkill(revisit_trigger=0), 'learned'),
                    ('human_revisit_skill', initial, 'learned'),
                    ('route_greedy', PatrolSkill(revisit_trigger=0), 'route_greedy')]
        controls += [(name, PatrolSkill(**value['skill']), 'learned') for name, value in retained.items()]
        for name, version, method in controls:
            average, rows, _ = evaluate(estimator, policy, version, contract['test'], device, method=method, trace=True)
            tests[name] = {'metrics': average, 'cases': rows}
    summary = {'trial': trial, 'policy_seed': policy_seed, 'proposal_seed': proposal_seed, 'slots_per_method': slots,
               'initial_development': initial_score, 'retained': retained, 'test': tests,
               'test_used_for_selection': False, 'job_id': os.environ.get('SLURM_JOB_ID'),
               'estimator_sha256': hashlib.sha256(estimator_path.read_bytes()).hexdigest(),
               'policy_sha256': hashlib.sha256(policy_path.read_bytes()).hexdigest(),
               'model_revision': json.loads((model_directory/'verified_manifest.json').read_text())['revision'],
               'actual_costs': {method: {key: sum(r[key] for r in records if r['method'] == method)
                                       for key in ['input_tokens', 'output_tokens', 'generation_seconds']}
                                for method in contract['methods']},
               'interpretation': 'equal_candidate_slots_not_equal_total_compute; tool gains and improver gains reported separately'}
    (output/'summary.json').write_text(json.dumps(summary))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True);p.add_argument('--model', type=Path, required=True)
    p.add_argument('--checkpoints', type=Path, required=True);p.add_argument('--trial', type=int, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    run(a.output, a.model, a.checkpoints, a.trial, slots=2 if a.smoke else 6, final_test=not a.smoke)


if __name__ == '__main__':
    main()
