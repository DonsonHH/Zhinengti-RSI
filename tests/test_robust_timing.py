import copy
import unittest

import numpy as np

from patrol_rsi.improver_task import ImproverTask, PatrolProgram
from patrol_rsi.public_dispatch import snapshot
from patrol_rsi.inspection_timing import public_packet, opportunities
from patrol_rsi.robust_timing import (OBSERVATION_FIELDS, observation_control, projected_values,
    RobustTimingPolicy, blind_time)
from patrol_rsi.timing_run import run as run_old
from patrol_rsi.robust_timing_run import run


def spec():
    return dict(phase='smoke', seed=8399, scenario='aux_feeder_loss+cdu_fouling', duration_s=180,
        fault_start_s=335, noise_c=.2, estimator='nominal', control='feedback', resources='provisioned')


class RobustTimingTests(unittest.TestCase):
    def packet(self):
        env = ImproverTask(spec(), PatrolProgram())
        return env, public_packet(env)

    def test_role_permutation_preserves_observation_distribution_and_legality_fields(self):
        env, packet = self.packet();before = copy.deepcopy(packet)
        for j, r in enumerate(packet['tasks']):
            r['temperature_c'] = 50. + j / 10
            r['velocity_c_s'] = j / 10000
        controlled = observation_control(packet, 'permuted')
        for kind in ('thermal', 'electrical', 'cooling', 'control'):
            ids = [j for j, r in enumerate(packet['tasks']) if r['kind'] == kind]
            old = sorted(tuple(packet['tasks'][j][k] for k in OBSERVATION_FIELDS) for j in ids)
            new = sorted(tuple(controlled['tasks'][j][k] for k in OBSERVATION_FIELDS) for j in ids)
            self.assertEqual(old, new)
        for a, b in zip(packet['tasks'], controlled['tasks']):
            self.assertNotEqual(a['temperature_c'], b['temperature_c'])
            self.assertEqual({k:v for k,v in a.items() if k not in OBSERVATION_FIELDS},
                {k:v for k,v in b.items() if k not in OBSERVATION_FIELDS})
        self.assertEqual(controlled['agents'], packet['agents'])
        self.assertEqual(before['agents'], packet['agents'])

    def test_masked_signals_cannot_generate_positive_utility(self):
        env, packet = self.packet()
        masked = observation_control(packet, 'masked')
        self.assertEqual(opportunities(masked, 120), [])
        groups = snapshot(env, lambda f:f[:,0])
        value = RobustTimingPolicy(dict(method='lookahead',signal_mode='masked')).choose(groups, packet, False)
        self.assertFalse(value[2])

    def test_conservative_future_never_increases_and_current_values_unchanged(self):
        env, packet = self.packet()
        for r in packet['tasks']:
            if r['kind'] == 'thermal':
                r.update(temperature_c=65.,last_temperature_c=64.,temperature_age_s=0.,velocity_c_s=.2)
        original = projected_values(packet);conservative = projected_values(packet,conservative=True)
        key = lambda r:(r['agent'],r['task'],r['dispatch_time_s'])
        lookup = {key(r):r for r in original}
        self.assertTrue(any(r['dispatch_time_s'] > packet['time_s'] for r in original))
        for row in conservative:
            self.assertLessEqual(row['surrogate_value'],lookup[key(row)]['surrogate_value']+1e-12)
        self.assertEqual([r for r in conservative if r['dispatch_time_s']==packet['time_s']],
            [r for r in original if r['dispatch_time_s']==packet['time_s']])

    def test_hidden_metadata_and_observations_cannot_change_blind_choice(self):
        env, packet = self.packet();groups = snapshot(env,lambda f:f[:,0])
        policy = RobustTimingPolicy(dict(method='blind',blind_seed=0))
        now = blind_time(0);packet['time_s'] = now
        changed = copy.deepcopy(groups)
        for g in changed:
            g.update(hidden_fault_start_s=150,scenario='secret')
            for r in g['candidates']:
                r.update(cue_strength=100.,known_measured_alarm=True,actor_score=-999.,future_label=True)
        a = policy.choose(groups,packet,False);b = policy.choose(changed,dict(time_s=now,future_labels=[1]),False)
        self.assertEqual(a[0],b[0]);self.assertTrue(a[2]);self.assertTrue(b[2])
        self.assertEqual(a[3]['selected_blind'],b[3]['selected_blind'])
        self.assertFalse(policy.choose(groups,{'time_s':now-5},False)[2])
        self.assertFalse(policy.choose(groups,{'time_s':now},True)[2])
        with self.assertRaises(ValueError):
            RobustTimingPolicy(dict(method='blind',blind_seed=3))
        with self.assertRaises(ValueError):
            RobustTimingPolicy(dict(method='lookahead',scenario='secret'))

    def test_public_controls_ignore_future_labels_and_do_not_mutate_packet(self):
        env, packet = self.packet();original = copy.deepcopy(packet)
        changed = copy.deepcopy(packet);changed.update(seed=1,scenario='secret',fault_start_s=5,future_labels=[True]*360)
        for mode in ('original','permuted','masked'):
            self.assertEqual(projected_values(observation_control(packet,mode)),projected_values(observation_control(changed,mode)))
        self.assertEqual(packet,original)

    def test_original_controller_is_action_and_event_equivalent_to_b1(self):
        for method in ('evidence','lookahead','cue'):
            old,_ = run_old(spec(),None,lambda f:f[:,0],method)
            new,_ = run(spec(),None,lambda f:f[:,0],method)
            for key in ('action_log','inspection_events','anomaly_intervals'):
                self.assertEqual(old[key],new[key])
            self.assertEqual(new['metrics']['tasks_violating_exact_gap'],0)
            self.assertLessEqual(new['metrics']['selected_detours'],1)

    def test_blind_and_conservative_whole_episode_preserve_coverage_and_budget(self):
        for config in (dict(method='blind',blind_seed=1),dict(method='lookahead',conservative=True)):
            record,_ = run(spec(),None,lambda f:f[:,0],config['method'],config=config)
            self.assertEqual(record['metrics']['tasks_violating_exact_gap'],0)
            self.assertLessEqual(record['metrics']['selected_detours'],1)


if __name__=='__main__':
    unittest.main()
