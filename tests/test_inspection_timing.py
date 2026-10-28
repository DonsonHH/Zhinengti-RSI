import copy
import unittest

import numpy as np

from patrol_rsi.improver_task import ImproverTask, PatrolProgram
from patrol_rsi.public_dispatch import snapshot, choose
from patrol_rsi.inspection_timing import public_packet, opportunities, strength, TimingPolicy
from patrol_rsi.timing_run import run


def spec():
    return dict(phase='smoke', seed=8199, scenario='healthy', duration_s=900, fault_start_s=335,
        noise_c=.2, estimator='nominal', control='feedback', resources='provisioned')


class TimingTests(unittest.TestCase):
    def test_future_tour_projection_uses_only_known_schedule_and_preserves_legality(self):
        env = ImproverTask(spec(), PatrolProgram())
        packet = public_packet(env)
        for row in packet['tasks']:
            row.update(temperature_c=90., last_temperature_c=90., contact_c=90., flow_kg_s=10., sensor_verified=True)
        predicted = opportunities(packet, 120)
        self.assertTrue(predicted)
        for _ in range(25):
            groups = snapshot(env, lambda f: f[:, 0])
            legal = {(g['agent'], r['task']) for g in groups for r in g['candidates'] if r['is_optional']}
            for row in predicted:
                if row['dispatch_time_s'] == env.clock:
                    self.assertIn((row['agent'], row['task']), legal)
                    self.assertGreaterEqual(row['return_slack_s'], 0.)
            env.step(choose(groups, 'tour')[0])

    def test_hidden_fields_do_not_affect_public_planner(self):
        env = ImproverTask(spec(), PatrolProgram());packet = public_packet(env)
        changed = copy.deepcopy(packet)
        changed.update(scenario='secret', future_labels=[1]*360, fault_start_s=5)
        for row in changed['tasks']:
            row['future_temperature_c'] = 1000.
        self.assertEqual(opportunities(packet, 120), opportunities(changed, 120))

    def test_explicit_wait_has_value_and_keeps_global_budget(self):
        env = ImproverTask(spec(), PatrolProgram());groups = snapshot(env, lambda f: f[:, 0])
        packet = public_packet(env)
        policy = TimingPolicy({'method':'lookahead'})
        choices, logs, used, diagnostic = policy.choose(groups, packet, False)
        self.assertFalse(used)
        self.assertIn(diagnostic['decision'], ('wait_no_candidate','wait_value'))
        self.assertEqual(choices, choose(groups,'tour')[0])
        self.assertEqual(policy.choose(groups,packet,True)[3]['decision'],'budget_used')

    def test_now_vs_wait_value_changes_the_stop_decision(self):
        def packet(first_other):
            tasks=[dict(kind='thermal',service_s=10.,deadline_s=300.,weight_scaled=1.,last_check_s=-1.,
                already_alarm=False,temperature_c=90.,last_temperature_c=90.,temperature_age_s=5.,velocity_c_s=0.) for _ in range(2)]
            edge={str(j):{str(k):10. for k in range(2)} for j in range(2)}
            agent=dict(agent=0,tour=[0,1],cursor=0,busy_target=None,busy_finish_s=None,detours_used=0,
                baseline_completed=0,first_costs_s={'0':10.,'1':first_other},costs_s=edge,period_s=20.)
            return dict(time_s=5.,tasks=tasks,agents=[agent])
        rows=[dict(task=j,actor_score=0.,cue_strength=1.,known_measured_alarm=False,asset_weight_scaled=1.,
            eta_over_deadline=.1,is_optional=j!=0) for j in (0,1)]
        groups=[dict(agent=0,time_s=5.,baseline_next=0,detours_used=0,baseline_completed=0,candidates=rows)]
        policy=TimingPolicy({'method':'lookahead'})
        pending=packet(30.)
        pending['tasks'].append(copy.deepcopy(pending['tasks'][0]))
        pending['agents'][0].update(tour=[0,1,2],busy_target=0,busy_finish_s=15.,
            first_costs_s={'0':10.,'1':30.,'2':30.},
            costs_s={str(j):{str(k):5. if (j,k)==(0,2) else 10. for k in range(3)} for j in range(3)})
        waiting=policy.choose([],pending,False)
        self.assertFalse(waiting[2]);self.assertGreater(waiting[3]['wait_value'],waiting[3]['now_value'])
        checking=policy.choose(groups,packet(5.),False)
        self.assertTrue(checking[2]);self.assertGreater(checking[3]['now_value'],checking[3]['wait_value'])

    def test_gate_waits_without_idling_and_off_never_spends(self):
        env = ImproverTask(spec(), PatrolProgram());groups = snapshot(env, lambda f: f[:,0]);packet=public_packet(env)
        for gate in (60,180,300,'off'):
            value=TimingPolicy({'method':'cue','not_before_s':gate}).choose(groups,packet,False)
            self.assertFalse(value[2]);self.assertEqual(value[0],choose(groups,'tour')[0])
        with self.assertRaises(ValueError):
            TimingPolicy({'method':'cue','not_before_s':150})

    def test_current_positive_surrogate_has_legal_action_and_forecasts_bounded(self):
        env = ImproverTask(spec(),PatrolProgram());groups=snapshot(env,lambda f:f[:,0]);packet=public_packet(env)
        for row in packet['tasks']:
            row.update(temperature_c=90.,last_temperature_c=90.,contact_c=90.,flow_kg_s=10.,sensor_verified=True)
        choices,logs,used,decision=TimingPolicy({'method':'evidence'}).choose(groups,packet,False)
        self.assertTrue(used);self.assertEqual(sum(v!=g['baseline_next'] for g in groups for i,v in choices.items() if i==g['agent']),1)
        row=dict(kind='thermal',last_temperature_c=90.,velocity_c_s=.2,temperature_age_s=500.,temperature_c=50.)
        self.assertEqual(strength(row,500.),(110.-65)/20)
        for row in packet['tasks']:
            row['already_alarm']=True
        self.assertFalse(opportunities(packet,120))

    def test_whole_episode_wait_and_lookahead_preserve_exact_revisit(self):
        for method in ('evidence','lookahead'):
            record,_=run(spec(),None,lambda f:f[:,0],method)
            self.assertLessEqual(record['metrics']['selected_detours'],1)
            self.assertEqual(record['metrics']['tasks_violating_exact_gap'],0)
            self.assertEqual(len(record['timing_decisions']),359)


if __name__=='__main__':
    unittest.main()
