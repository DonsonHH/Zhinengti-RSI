"""Inspect M12-A selection/objective alignment without selecting a new program."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--evidence', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
compact_path = args.evidence / 'report/compact-metrics.json.gz'
rows = json.loads(gzip.decompress(compact_path.read_bytes()))
by_version = defaultdict(list)
for row in rows:
    if row['specification']['phase'] == 'development':
        by_version[row['version_id']].append(row)
programs = []
for version, cases in by_version.items():
    assert len(cases) == 8
    weight = sum(r['event_weight_sum'] for r in cases)
    public = [r['public_selection'] for r in cases]
    programs.append({'version_id': version, 'workflow': cases[0]['workflow'],
        'anomaly_segments': sum(r['metrics']['anomaly_segments'] for r in cases),
        'missed_segments': sum(r['metrics']['missed_segments'] for r in cases),
        'weighted_censored_delay_s': sum(r['weighted_delay_sum'] for r in cases) / weight if weight else None,
        'mean_public_first_alert_task_weight': sum(r['first_alert_task_weight'] for r in public) / 8,
        'mean_public_weighted_age': sum(r['weighted_age_ratio'] for r in public) / 8,
        'mean_travel_m': sum(r['travel_m'] for r in public) / 8,
        'violating_tasks': sum(r['violating_tasks'] for r in public),
        'mean_coverage': sum(r['coverage'] for r in public) / 8})
programs.sort(key=lambda r: (r['violating_tasks'], -r['mean_coverage'],
    -r['mean_public_first_alert_task_weight'], r['mean_public_weighted_age'], r['mean_travel_m']))
selections = json.loads((args.evidence / 'selections-frozen.json').read_text())
proposals = [json.loads(p.read_text()) for p in sorted((args.evidence / 'proposals').glob('*.json'))]
initial = selections[0]['initial']['workflow']
baseline = next(r for r in programs if r['workflow'] == initial)
model_versions = {r['version_id'] for r in selections if r['method'].startswith('llm')}
assert len(model_versions) == 1
model = next(r for r in programs if r['version_id'] in model_versions)
result = {'stage': 'M12-A retrospective review; no new selection on old test data',
    'compact_source_sha256': hashlib.sha256(compact_path.read_bytes()).hexdigest(),
    'development_unique_programs': len(programs), 'development_case_records': sum(map(len, by_version.values())),
    'all_first_alert_task_weights_equal': len({r['mean_public_first_alert_task_weight'] for r in programs}) == 1,
    'all_coverage_and_exact_violation_terms_equal': len({(r['violating_tasks'], r['mean_coverage']) for r in programs}) == 1,
    'selection_rank_ordered_programs': programs,
    'initial_program': baseline, 'selected_model_program': model,
    'model_versus_initial_development': {'public_age_delta': model['mean_public_weighted_age'] - baseline['mean_public_weighted_age'],
        'missed_segment_delta': model['missed_segments'] - baseline['missed_segments'],
        'censored_delay_delta_s': model['weighted_censored_delay_s'] - baseline['weighted_censored_delay_s']},
    'proposal_outcomes': {method: dict(Counter(p['reason'] for p in proposals if p['method'] == method))
        for method in ('random_search', 'llm_no_memory', 'llm_memory')},
    'all_llm_selected_programs_identical': len(model_versions) == 1,
    'interpretation': 'Public first-alert count saturates on these development cases; selection then favors age, which is not the hidden confirmation-delay objective. Numeric M12-A results remain unchanged.',
    'hypotheses_not_proven': ['Coarse aggregate diagnostics, restricted grammar or proposal instructions may contribute to repeats; repetition alone does not identify the cause'],
    'review_scope': 'Current source/archive/statistic verification plus complete retrospective metadata; previous full312case replay retained, not rerun again here'}
with args.output.open('x', encoding='utf-8', newline='\n') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({k: result[k] for k in ('development_unique_programs', 'all_first_alert_task_weights_equal',
    'all_coverage_and_exact_violation_terms_equal', 'model_versus_initial_development')}, indent=2))
