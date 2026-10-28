"""Retrospective improver transfer and tool-meaning review; no new selection."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--evidence', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
aggregate = json.loads((args.evidence / 'report/aggregate.json').read_text())
trials, accepted, examples = [], [], []
for trial in range(3):
    folder = args.evidence / f'trial-{trial}'
    selections = json.loads((folder / 'selections-frozen.json').read_text())
    outcomes = aggregate['trials'][str(trial)]
    trials.append({'trial': trial, 'fixed3_workflow': selections['direct3']['workflow'],
        'meta3_workflow': selections['model_meta3']['workflow'],
        'same_selected_program': selections['direct3']['version_id'] == selections['model_meta3']['version_id'],
        'test_missed_delta_meta_minus_fixed': outcomes['model_meta3']['missed_segments'] - outcomes['direct3']['missed_segments'],
        'test_weighted_wait_delta_s_meta_minus_fixed': outcomes['model_meta3']['weighted_censored_delay_s'] - outcomes['direct3']['weighted_censored_delay_s']})
    for path in sorted((folder / 'meta').glob('model_meta-round*.json')):
        row = json.loads(path.read_text())
        before, after = row['incumbent'], row['candidate']
        changed = [k for k in before if before[k] != after[k]]
        record = {'trial': trial, 'round': row['round'], 'valid': row['valid'], 'accepted': row['accepted'],
            'changed_protocol_fields': changed, 'old_tools': before['diagnostics'], 'candidate_tools': after['diagnostics'],
            'development_incumbent': row['arms']['incumbent']['selected']['validator'],
            'development_candidate': row['arms']['candidate']['selected']['validator']}
        if row['accepted']:
            accepted.append(record)
        if 'novelty' in after['diagnostics']:
            proposal = row.get('proposal', {})
            examples.append({'trial': trial, 'round': row['round'],
                'model_diagnosis_unverified': proposal.get('diagnosis'),
                'model_expected_effect_unverified': proposal.get('expected_effect'),
                'actual_tool_output': ['prior_candidate_count', 'duplicates', 'seen_versions'],
                'review': 'This tool describes proposal repetition, not plant anomalies. The quoted model explanation does not prove a sensing capability.'})
result = {'scope': 'Retrospective metadata review of M12-B; old results unchanged and no workflow reselected',
    'aggregate_sha256': hashlib.sha256((args.evidence / 'report/aggregate.json').read_bytes()).hexdigest(),
    'accepted_model_protocol_updates': accepted,
    'accepted_updates_only_changed_tool_selection': all(r['changed_protocol_fields'] == ['diagnostics'] for r in accepted),
    'fresh_task_outcomes_by_trial': trials, 'tool_semantics_examples': examples,
    'findings': ['Accepted protocol changes did not improve the new-task fixed3comparison in any of3trials.',
        'The accepted4updates changed diagnostic tool selection; memory and instruction stayed the same.',
        'Protocol proposal prompt listed tool names without detailed meanings. Some model explanations confused proposal novelty with anomaly detection.',
        'The old task prompt provided aggregate episode statistics rather than per-device measurement sequences.'],
    'causal_limits': 'This metadata does not prove that coarse diagnostics or tool-name confusion caused transfer failure. A fresh matched comparison is required.'}
with args.output.open('x', encoding='utf-8', newline='\n') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'accepted_updates': len(accepted), 'only_tool_selection_changed': result['accepted_updates_only_changed_tool_selection'],
    'test_wait_deltas': [r['test_weighted_wait_delta_s_meta_minus_fixed'] for r in trials]}, indent=2))
