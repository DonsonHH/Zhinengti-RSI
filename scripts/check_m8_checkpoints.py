"""Read actual selected weights and replay first-case choices on CPU."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from audit_m8 import read
from patrol_rsi.m6 import parameter_hash
from patrol_rsi.m8 import VARIANTS, contract
from patrol_rsi.relation_model import RelationActorCritic, tensors
from patrol_rsi.relations import RelationDispatchEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = []
    for variant in VARIANTS:
        for seed in range(3):
            directory = args.results / f'{variant}-seed-{seed}'
            summary = read(directory)
            path = directory / 'best-policy.pt'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == summary['checkpoint_sha256']
            saved = torch.load(path, map_location='cpu', weights_only=True)
            assert saved['variant'] == variant and saved['seed'] == seed
            assert saved['selected_episode'] == summary['selected_episode'] and saved['validation'] == summary['best_validation']
            assert saved['contract_sha256'] == hashlib.sha256(json.dumps(contract(), sort_keys=True).encode()).hexdigest()
            model = RelationActorCritic(variant.endswith('_relation')).eval()
            assert saved['relational'] == model.relational
            model.load_state_dict(saved['model'])
            assert sum(p.numel() for p in model.parameters()) == 22594
            assert parameter_hash(model) == summary['selected_parameter_sha256']
            case = contract()['test'][0]
            expected = summary['test']['learned']['cases'][0]
            wrapper = RelationDispatchEnv(case)
            for step, event in enumerate(expected['events']):
                estimate = np.full_like(wrapper.env.values, .5)
                estimate[wrapper.env.scenario.targets] = np.asarray(event['target_estimates'], dtype=np.float32)
                wrapper.env.set_estimate(estimate)
                state = wrapper.prepare()
                with torch.no_grad():
                    chosen, _, _, _ = model.joint(**tensors([state], torch.device('cpu')), deterministic=True)
                assert chosen[0].tolist() == event['choices'], (variant, seed, step, 'actual CPU checkpoint choice differs')
                wrapper.step(chosen[0].numpy(), state, True)
                assert wrapper.env.positions.tolist() == expected['positions'][step + 1]
            rows.append({'variant': variant, 'seed': seed, 'parameters': 22594,
                         'checkpoint_sha256': summary['checkpoint_sha256'], 'selected_episode': saved['selected_episode'],
                         'actual_metadata_parameters_and_CPU_first_case_choices_match': True})
    with args.output.open('x') as handle:
        json.dump({'actual_selected_weights_read': 12, 'CPU_first_case_microsteps_replayed': 12 * 96, 'rows': rows}, handle, indent=2)
    print('M8_ACTUAL_WEIGHTS_AND_CPU_CHOICES_VERIFIED', len(rows), flush=True)


if __name__ == '__main__':
    main()
