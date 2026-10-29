"""CPU-only read-only diagnosis of the already saved B2 development branches."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import heapq
import json
import math
from pathlib import Path
import time

from patrol_rsi.aisles import AisleGraph, build_agents
from patrol_rsi.coverage_plan import ReturnPlan
from patrol_rsi.facility import FacilityConfig
from patrol_rsi.queue_diagnostics import ProxyConfig, collateral_proxy, diagnose_queue
from patrol_rsi.queue_history import history_state
from patrol_rsi.queue_oracle import offline_oracle


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n')


def distance(graph, source, target):
    """Independent Dijkstra; does not use AisleGraph.path or ReturnPlan costs."""
    queue, best = [(0., source)], {source: 0.}
    while queue:
        value, node = heapq.heappop(queue)
        if value != best[node]:
            continue
        if node == target:
            return value
        for nxt, edge in graph.adjacency[node].items():
            candidate = value + edge
            if candidate < best.get(nxt, math.inf):
                best[nxt] = candidate
                heapq.heappush(queue, (candidate, nxt))
    raise ValueError('Unreachable static task')


def independent_remaining(plan, state, target):
    order = list(plan.tour[state.cursor:] + plan.tour[:state.cursor])
    queue = ([target] if target is not None else []) + order
    position, now, rows = state.position_m, state.time_s, []
    for repeat in range(1000):
        for j in queue if repeat == 0 else order:
            cost = distance(plan.graph, position, plan.tasks[j].position_m) + plan.tasks[j].service_s
            now += 5 * math.ceil(cost / 5 - 1e-10)
            if now > 1800:
                return rows
            rows.append((j, now))
            position = plan.tasks[j].position_m
    raise RuntimeError('Static schedule did not progress')


def confirm_saved_schedules(diagnostic, plan, state, baseline, branch):
    comparisons = 0
    for original, field, target in ((baseline, 'tour_plan', None), (branch, 'detour_plan', diagnostic['target'])):
        predicted = [(r['task'], r['time_s']) for r in diagnostic[field]['mission_completions']]
        independent = independent_remaining(plan, state, target)
        actual = [(r['task'], r['time_s']) for r in original['inspection_events']
            if r['agent'] == state.agent and r['time_s'] > state.time_s]
        if predicted != independent or predicted != actual:
            raise RuntimeError('Public queue forecast differs from Dijkstra or actual recorded completions')
        comparisons += len(actual)
    certificate = next(r['certificate'] for r in branch['selected_certificates']
        if r['agent'] == state.agent and r['clock_s'] == state.time_s and r['task'] == diagnostic['target'])
    if diagnostic['detour_certificate'] != certificate:
        raise RuntimeError('Whole-return certificate differs from original execution')
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    started = time.perf_counter()
    index = {r['case']: r for r in read(args.evidence / 'report/compact-index.json.gz')}
    files = sorted(list((args.evidence / 'failure-examples').glob('*.json.gz'))
        + list((args.evidence / 'demo').glob('*.json.gz')))
    records, sources = {}, {}
    for path in files:
        case = path.name.removesuffix('.json.gz')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != index[case]['sha256']:
            raise RuntimeError('Saved development case differs from the original B2 index')
        records[case] = read(path)
        sources[case] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
    graph = AisleGraph(FacilityConfig(racks=128), 'open')
    agents = build_agents(graph, 'provisioned')
    public, offline, inputs, compared = [], [], [], 0
    for case, branch in records.items():
        extra = [r for r in branch['workflow_decisions'] if r['task'] != r['baseline_next']]
        if not extra:
            continue
        if len(extra) != 1:
            raise RuntimeError('This diagnostic only matches the one-extra-check experiment')
        action = extra[0]
        baseline_case = case.rsplit('--', 1)[0] + '--tour'
        baseline = records[baseline_case]
        now = action['time_s'];agent = agents[action['agent']]
        # Only observations already completed and earlier assignments enter
        # the public adapter. Hidden metadata stays in the offline envelope.
        assignments = [r for r in branch['assignments'] if r['time_s'] < now]
        events = [r for r in branch['inspection_events'] if r['time_s'] <= now]
        if assignments != [r for r in baseline['assignments'] if r['time_s'] < now]:
            raise RuntimeError('Paired branches do not share the same pre-intervention actions')
        if events != [r for r in baseline['inspection_events'] if r['time_s'] <= now]:
            raise RuntimeError('Paired branches do not share the same observed history')
        plan = ReturnPlan(graph, graph.tasks, agent['tour'], branch['workflow_contract']['deadlines_s'])
        state = history_state(plan, now=now, agent=agent, assignments=assignments, events=events,
            weights=branch['workflow_contract']['task_weights'], global_used=False)
        diagnosis = diagnose_queue(plan, state, action['task'])
        if not diagnosis['legal']:
            raise RuntimeError('Previously executed branch is not legal under the frozen plan')
        proxy = collateral_proxy(plan, state, diagnosis)
        checked = confirm_saved_schedules(diagnosis, plan, state, baseline, branch)
        compared += checked
        alias = f'development-{len(public) + 1:02d}'
        public.append({'case_alias': alias, 'queue_diagnostic': diagnosis, 'uncertain_opportunity_proxy': proxy})
        timing = next(r for r in branch['timing_decisions'] if r['time_s'] == now)
        saved_value = timing.get('now_value')
        outcome = offline_oracle(diagnosis, baseline, branch)
        offline.append({'case_alias': alias, 'original_case': case, 'baseline_case': baseline_case,
            'specification': branch['specification'], 'original_action': action,
            'saved_B2_timing_comparison': timing,
            'saved_target_value_minus_new_collateral': saved_value - proxy['collateral_cost'] if saved_value is not None else None,
            'note': 'Post hoc diagnostic arithmetic only, not a deployed revised policy or test performance.',
            'offline_outcome': outcome, 'all_remaining_completions_checked': checked})
        inputs.append({'case_alias': alias, 'public_state': asdict(state),
            'public_static_agent': {'id': agent['id'], 'role': agent['role'], 'tour': agent['tour']},
            'source_case': case, 'source': sources[case], 'baseline_source': sources[baseline_case]})
    write(args.output / 'public-queue-diagnostics.json', {'view': 'public_only',
        'schema_version': 1, 'prototype_coefficients': asdict(ProxyConfig()), 'cases': public})
    write(args.output / 'offline-queue-review.json', {'view': 'offline_evaluator_only',
        'development_only_old_B2_cases': True, 'cases': offline})
    write(args.output / 'diagnostic-inputs.json', {'view': 'offline_provenance_wrapper_with_public_states', 'cases': inputs})
    write(args.output / 'execution-summary.json', {'new_simulation_rollouts': 0, 'new_neural_training': 0, 'LLM_calls': 0,
        'old_complete_source_records': len(records), 'diagnosed_recorded_branches': len(public),
        'distinct_physical_configurations': len({r['original_case'].split('--history--')[0] for r in offline}),
        'independent_Dijkstra_and_actual_completion_checks': compared,
        'seconds': time.perf_counter() - started,
        'scope': 'Local CPU reconstruction, not generalization or an online value-policy comparison'})
    print(json.dumps({'branches': len(public), 'checked_completions': compared, 'new_rollouts': 0}))


if __name__ == '__main__':
    main()
