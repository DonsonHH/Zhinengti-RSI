"""M3: finite-space anchor and transfer of frozen improver procedures."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import time

import numpy as np

from .m2 import DiagnosticPlan, evaluate, feedback, parse, rank
from .revisit import PatrolSkill


METHODS = ('random_search', 'fixed_coverage', 'fixed_revisit', 'transferred_protocol')


def config_space():
    result = []
    for values in product((1, 2, 4), (1, 2, 4, 8), ('q_priority', 'cyclic'), (0, 12, 16, 24, 32)):
        try:
            result.append(PatrolSkill(*values))
        except ValueError:
            continue
    return result


def contract():
    # Noise is paired within the same layout and simulator field in M3.
    return {'development': [{'split': 'm3_development', 'layout_seed': 8000+i,
                             'field_seed': 900000+i, 'noise': 0.0} for i in range(6)],
            'test': [{'split': 'm3_test', 'layout_seed': 9000+i, 'field_seed': 10000000+i,
                      'noise': noise} for noise in (0.0, .03) for i in range(12)],
            'policy_seeds': [0, 1, 2], 'proposal_seeds': [389, 467, 557],
            'methods': list(METHODS), 'candidate_slots': 6, 'configuration_count': 110,
            'input_token_cap': 4096, 'output_token_cap': 512,
            'source_protocol_rule': 'M2 final updated_improver diagnostic plan from corresponding trial index; all nine, no selection by test',
            'reset_rule': 'Each method starts at PatrolSkill() with empty history; transfer diagnostic plan only; no task patch or source score transfer',
            'test_rule': 'Enumeration and all optimizer selections frozen before test access',
            'enumeration_budget': '110 configurations x 6 development cases per policy, extra-compute anchor; not equal-slot comparison'}


def source_plan(summary):
    # Deliberately whitelist exactly the protocol, excluding task configuration
    # and both source development/test results from the transfer path.
    value = summary['retained']['updated_improver']['final_diagnostic_plan']
    return DiagnosticPlan(**{key: value[key] for key in ('tool', 'instruction')})


def load_models(checkpoints, policy_seed):
    import torch
    from .models import ConditionalVAEUNet, SharedDuelingQ
    if not torch.cuda.is_available():
        raise RuntimeError('Allocated CUDA required; no CPU fallback')
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    estimator_path = checkpoints/'vae/best-vae.pt'
    policy_path = checkpoints/f'seed-{policy_seed}/best-policy.pt'
    estimator, policy = ConditionalVAEUNet().to(device).eval(), SharedDuelingQ().to(device).eval()
    estimator.load_state_dict(torch.load(estimator_path, map_location=device, weights_only=True)['model'])
    policy.load_state_dict(torch.load(policy_path, map_location=device, weights_only=True)['model'])
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in
              [('estimator', estimator_path), ('policy', policy_path)]}
    return estimator, policy, device, hashes


def enumerate_development(output, checkpoints, policy_seed):
    output.mkdir(parents=True, exist_ok=False)
    estimator, policy, device, hashes = load_models(checkpoints, policy_seed)
    c = contract();records = [];start = time.monotonic()
    with (output/'configurations.jsonl').open('x') as log:
        for i, skill in enumerate(config_space()):
            score, rows, _ = evaluate(estimator, policy, skill, c['development'], device)
            record = {'index': i, 'skill': asdict(skill), 'metrics': score, 'cases': rows}
            records.append(record);log.write(json.dumps(record)+'\n');log.flush()
            if (i+1) % 10 == 0:
                print(json.dumps({'policy_seed': policy_seed, 'enumerated': i+1, 'seconds': time.monotonic()-start}), flush=True)
    best = min(records, key=lambda row: rank(row['metrics']))
    result = {'policy_seed': policy_seed, 'selected': best, 'count': len(records), 'checkpoint_hashes': hashes,
              'selection_split': 'm3_development', 'test_used_for_selection': False,
              'seconds': time.monotonic()-start, 'job_id': os.environ.get('SLURM_JOB_ID')}
    (output/'selection-frozen.json').write_text(json.dumps(result, indent=2))


def build_prompt(parent, score, plan, diagnostics, history, generation):
    # No JSON example with an actual task value or diagnostic procedure. All
    # groups see the same grammar and field semantics, differing only by plan.
    return copy.deepcopy({
        'generation': generation, 'current_skill': asdict(parent), 'development_metrics': score,
        'diagnostic_plan': asdict(plan),
        'diagnostic_results': [d[plan.tool] | {'layout_seed': d['layout_seed']} for d in diagnostics[:2]],
        'prior_task_feedback': history[-3:],
        'field_domains': {'estimator_samples': [1, 2, 4], 'estimator_refresh_interval': [1, 2, 4, 8],
                          'coordination_order': ['q_priority', 'cyclic'], 'revisit_trigger': [0, 12, 16, 24, 32]},
        'semantics': {'revisit_trigger': '0 disables human recovery; otherwise raw age threshold activates latched shortest-path recovery',
                      'estimator_refresh_interval': 'larger means less frequent refresh', 'estimator_samples': 'prior samples per refresh'},
        'acceptance': 'Lexicographically minimize coverage loss, target-time age>24 fraction, weighted idleness, total prior samples. Higher coverage is better. All other metrics lower are better except intervention fraction, which is a reported cost.',
        'budget': '96 patrol steps; estimator_samples * ceil(96/estimator_refresh_interval) <= 192',
        'output_grammar': [
            'Return one JSON object with diagnosis (string), patch (object), expected_effect (string), and next_diagnostic (object).',
            'patch keys must be actual keys from field_domains, never placeholders; values must be allowed integers or the coordination string. Include changed fields only.',
            'next_diagnostic has exactly tool and instruction: tool is one of coverage/revisit/oscillation; instruction is a reusable procedure string of 1..600 characters.',
            'This experiment freezes the diagnostic plan. next_diagnostic is recorded but never adopted here. Propose a task patch using the active diagnostic results.']})


def run_trial(output, model_directory, checkpoints, source_directory, enumeration_directory, trial, smoke=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    output.mkdir(parents=True, exist_ok=False)
    c = contract();policy_seed = trial//3;proposal_seed = c['proposal_seeds'][trial%3]
    estimator, policy, device, hashes = load_models(checkpoints, policy_seed)
    src_path = source_directory/f'trial-{trial}/summary.json'
    transferred = source_plan(json.loads(src_path.read_text()))
    (output/'contract.json').write_text(json.dumps(c, indent=2))
    (output/'transferred-protocol.json').write_text(json.dumps({'plan': asdict(transferred), 'source_trial': trial,
             'source_summary_sha256': hashlib.sha256(src_path.read_bytes()).hexdigest(),
             'transferred_fields': ['tool', 'instruction'], 'task_config_reset': asdict(PatrolSkill()), 'history_reset': []}, indent=2))
    plans = {'fixed_coverage': DiagnosticPlan(),
             'fixed_revisit': DiagnosticPlan('revisit', 'Use raw-age diagnostics to propose one testable configuration change.'),
             'transferred_protocol': transferred}
    initial = PatrolSkill()
    initial_score, _, initial_diagnostics = evaluate(estimator, policy, initial, c['development'], device)
    tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True, trust_remote_code=False)
    lm = AutoModelForCausalLM.from_pretrained(model_directory, torch_dtype=torch.float16, attn_implementation='eager',
                                             local_files_only=True, trust_remote_code=False).to(device).eval()
    slots = 2 if smoke else c['candidate_slots'];all_records = [];retained = {}
    for method in METHODS:
        parent, score, diagnostics = initial, initial_score, initial_diagnostics
        plan = plans.get(method, DiagnosticPlan());history = []
        rng = np.random.default_rng(proposal_seed)
        for generation in range(slots):
            r = {'method': method, 'generation': generation, 'parent': asdict(parent), 'plan': asdict(plan),
                 'accepted': False, 'input_tokens': 0, 'output_tokens': 0, 'generation_seconds': 0.0}
            if method == 'random_search':
                candidate = config_space()[int(rng.integers(len(config_space())))];patch = asdict(candidate)
                r['origin'] = 'search_baseline'
            else:
                user = build_prompt(parent, score, plan, diagnostics, history, generation)
                system = 'You improve a frozen VAE + multiagent DDQN through bounded skill/tool configurations. Use provided development evidence. Output JSON only. Do not modify evaluation, weights, data or budgets.'
                text = tokenizer.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(user)}],
                                                      tokenize=False, add_generation_prompt=True, enable_thinking=False)
                inputs = tokenizer(text, return_tensors='pt').to(device)
                if inputs['input_ids'].numel() > c['input_token_cap']:
                    raise RuntimeError('Prompt cap exceeded')
                torch.manual_seed(proposal_seed*100+generation);start = time.monotonic()
                with torch.no_grad():
                    ids = lm.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=.7, top_p=.8, top_k=20)
                answer = tokenizer.decode(ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                r.update({'origin': 'language_model', 'prompt': user, 'rendered_prompt': text, 'raw_response': answer,
                          'input_tokens': int(inputs['input_ids'].numel()), 'output_tokens': int(ids.shape[1]-inputs['input_ids'].shape[1]),
                          'generation_seconds': time.monotonic()-start})
                try:
                    candidate, ignored_plan, obj = parse(answer, parent);patch = obj['patch']
                    r['generated_diagnostic_not_adopted'] = asdict(ignored_plan)
                except (ValueError, TypeError) as error:
                    r['error'] = str(error);history.append(feedback(r));all_records.append(r)
                    (output/f'{method}-{generation}.json').write_text(json.dumps(r))
                    print(json.dumps({'trial': trial, 'method': method, 'generation': generation, 'error': str(error)}), flush=True)
                    continue
            candidate_score, rows, candidate_diagnostics = evaluate(estimator, policy, candidate, c['development'], device)
            accepted = rank(candidate_score) < rank(score)
            r.update({'patch': patch, 'candidate': asdict(candidate), 'metrics': candidate_score, 'cases': rows, 'accepted': accepted})
            if accepted:
                parent, score, diagnostics = candidate, candidate_score, candidate_diagnostics
            history.append(feedback(r));all_records.append(r)
            (output/f'{method}-{generation}.json').write_text(json.dumps(r))
            print(json.dumps({'trial': trial, 'method': method, 'generation': generation, 'accepted': accepted}), flush=True)
        retained[method] = {'skill': asdict(parent), 'development': score, 'plan': asdict(plan)}
    selections = {'optimizer_selections': retained}
    if not smoke:
        enum = json.loads((enumeration_directory/f'policy-{policy_seed}/selection-frozen.json').read_text())
        assert enum['count'] == 110 and not enum['test_used_for_selection'] and enum['checkpoint_hashes'] == hashes
        selections['enumeration_selected'] = enum['selected']
    (output/'selection-frozen.json').write_text(json.dumps(selections, indent=2))
    tests = {}
    if not smoke:
        candidates = [('human_default', initial), ('fixed_trigger12', PatrolSkill(revisit_trigger=12)),
                      ('enumeration_selected', PatrolSkill(**selections['enumeration_selected']['skill']))]
        candidates += [(name, PatrolSkill(**x['skill'])) for name, x in retained.items()]
        for method, skill in candidates:
            score, rows, _ = evaluate(estimator, policy, skill, c['test'], device, trace=True)
            tests[method] = {'metrics': score, 'cases': rows}
    summary = {'trial': trial, 'policy_seed': policy_seed, 'proposal_seed': proposal_seed, 'slots_per_method': slots,
               'initial_development': initial_score, 'retained': retained, 'test': tests, 'smoke': smoke,
               'test_used_for_selection': False, 'checkpoint_hashes': hashes,
               'job_id': os.environ.get('SLURM_JOB_ID'), 'transferred_protocol_only': True,
               'model_revision': json.loads((model_directory/'verified_manifest.json').read_text())['revision'],
               'actual_costs': {name: {key: sum(r[key] for r in all_records if r['method'] == name)
                                       for key in ('input_tokens', 'output_tokens', 'generation_seconds')}
                                for name in METHODS}}
    (output/'summary.json').write_text(json.dumps(summary))


def main():
    p = argparse.ArgumentParser();sub = p.add_subparsers(dest='stage', required=True)
    enum = sub.add_parser('enumerate');enum.add_argument('--policy-seed', type=int, required=True)
    run = sub.add_parser('transfer');run.add_argument('--model', type=Path, required=True)
    run.add_argument('--source', type=Path, required=True);run.add_argument('--enumeration', type=Path, required=True)
    run.add_argument('--trial', type=int, required=True);run.add_argument('--smoke', action='store_true')
    for command in (enum, run):
        command.add_argument('--checkpoints', type=Path, required=True);command.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.stage == 'enumerate':enumerate_development(a.output, a.checkpoints, a.policy_seed)
    else:run_trial(a.output, a.model, a.checkpoints, a.source, a.enumeration, a.trial, a.smoke)


if __name__ == '__main__':main()
