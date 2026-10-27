"""Locate raw waiting extremes and demonstrate public-state saturation."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from patrol_rsi.m8 import VARIANTS, contract
from patrol_rsi.relations import RelationDispatchEnv
from patrol_rsi.stress import planner_choices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for variant in VARIANTS:
        for seed in range(3):
            path = args.results / f'{variant}-seed-{seed}/summary.json.gz'
            summary = json.loads(gzip.decompress(path.read_bytes()))
            worst = max(summary['test']['learned']['cases'], key=lambda row: row['peak_age'])
            ages = np.asarray(worst['target_ages'])
            step, target = np.unravel_index(ages.argmax(), ages.shape)
            rows.append({'variant': variant, 'seed': seed, 'family': worst['family'],
                         'layout_seed': worst['layout_seed'], 'profile': worst['profile'], 'noise': worst['noise'],
                         'peak_age': int(ages[step, target]), 'peak_step': int(step), 'target_index': int(target),
                         'target_deadline': worst['target_deadlines'][target], 'violation': worst['deadline_violation'],
                         'coverage': worst['target_coverage']})
    left, right = RelationDispatchEnv(contract()['preflight']), RelationDispatchEnv(contract()['preflight'])
    # This is a constructed saturation diagnostic, not a claimed trajectory.
    for wrapper, multiple in ((left, 5), (right, 8)):
        wrapper.env.raw_age[wrapper.env.scenario.targets] = wrapper.deadlines * multiple
        wrapper.env.idleness[wrapper.env.scenario.targets] = 1
    first, second = left.prepare(), right.prepare()
    equal = all(np.array_equal(first[key], second[key]) for key in first)
    if not equal:
        raise RuntimeError('Saturation counterexample inputs differ')
    choice = planner_choices(first, 'slack')
    reward_left, event_left = left.step(choice, first)
    reward_right, event_right = right.step(choice, second)
    if not np.any(left.env.raw_age[left.env.scenario.targets] != right.env.raw_age[right.env.scenario.targets]):
        raise RuntimeError('Counterexample raw ages must still differ')
    if reward_left != reward_right or event_left['deadline_violation'] != event_right['deadline_violation']:
        raise RuntimeError('Saturated rewards differ')
    result = {'worst_case_by_training_seed': rows, 'synthetic_saturation_counterexample': {
        'public_actor_and_critic_inputs_equal': equal, 'raw_ages_differ_before_and_after_step': True,
        'same_choices_public_reward_equal': True, 'age_multiples_of_deadline': [5, 8], 'reward': reward_left,
        'scope': 'Controlled public-state counterexample after saturation; not proof that clipping uniquely caused the observed 92-step case'},
        'interpretation': 'Age/deadline is capped at 4, idleness at 1 and violation indicator is binary; extreme delays can be indistinguishable after saturation; future average-reward results must include raw maximum waiting and long-horizon tests'}
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print('M8_TAIL_DIAGNOSTIC', max(row['peak_age'] for row in rows), 'steps; saturation counterexample reproduced')


if __name__ == '__main__':
    main()
