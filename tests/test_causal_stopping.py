import copy
from dataclasses import replace
import unittest

from patrol_rsi.improver_task import ImproverTask, PatrolProgram
from patrol_rsi.causal_observation import capture_public, validate_public


def specification(seed=8300):
    return dict(phase='development', seed=seed, scenario='aux_feeder_loss+cdu_fouling',
        duration_s=180, fault_start_s=335, noise_c=.2, estimator='nominal',
        control='feedback', resources='provisioned')


class CausalStoppingTests(unittest.TestCase):
    def test_latest_remote_flow_is_not_replaced_by_old_qualified_flow(self):
        env=ImproverTask(specification(),PatrolProgram())
        bundle=capture_public(env)
        self.assertEqual(bundle['schema'],'causal-public-v1')
        task=bundle['tasks'][156]
        self.assertEqual(task['pod'],12)
        self.assertEqual(task['flow']['source'],'remote_flow')
        self.assertEqual(task['flow']['time_s'],5.)
        self.assertEqual(task['qualified']['available'],False)
        self.assertIsNone(task['qualified']['value'])
        self.assertIsNone(task['cvae_raw']['value'])
        self.assertEqual(validate_public(bundle),bundle)

    def test_wait_retains_budget_and_compares_against_same_terminal_check_policy(self):
        from patrol_rsi.public_stopping import stopping_decision
        candidates=[{'agent':0,'task':1,'dispatch_time_s':5.,'net_score':.1},
            {'agent':1,'task':2,'dispatch_time_s':20.,'net_score':.3}]
        decision=stopping_decision(candidates,now=5.,used=False)
        self.assertEqual(decision['action'],'WAIT')
        self.assertEqual(decision['q_check'],.1)
        self.assertEqual(decision['q_wait'],.3)
        self.assertEqual(decision['CHECK']['budget_after'],0)
        self.assertEqual(decision['WAIT']['budget_after'],1)
        self.assertEqual(decision['CHECK']['end_s'],1800.)
        self.assertEqual(decision['WAIT']['end_s'],1800.)
        future=stopping_decision(candidates,now=20.,used=False)
        self.assertEqual(future['action'],'CHECK')
        self.assertEqual(stopping_decision(candidates,now=20.,used=True)['action'],'TOUR')

    def test_future_or_hidden_fields_are_rejected_in_nested_public_sources(self):
        bundle=capture_public(ImproverTask(specification(),PatrolProgram()))
        for mutate in (
            lambda b:b['tasks'][156]['flow'].update(time_s=10.,age_s=-5.),
            lambda b:b['telemetry'].update(future_labels=[1]),
            lambda b:b['agents'][0].update(scenario='hidden'),
            lambda b:b['tasks'][0]['nominal'].update(future_temperature=90.)):
            changed=copy.deepcopy(bundle);mutate(changed)
            with self.assertRaises(ValueError):validate_public(changed)

    def test_pure_prior_and_assimilated_measurement_have_separate_timestamps(self):
        import numpy as np
        from patrol_rsi.causal_observation import ObservedTask
        env=ObservedTask(dict(specification(),estimator='history'),PatrolProgram(),lambda context:np.full(128,47.))
        from patrol_rsi.public_dispatch import snapshot,choose
        while env.clock<35:
            groups=snapshot(env,lambda f:f[:,0]);env.step(choose(groups,'tour')[0])
        row=capture_public(env)['tasks'][0]
        self.assertEqual(row['cvae_raw']['value'],47.)
        self.assertEqual(row['cvae_raw']['time_s'],30.)
        self.assertEqual(row['history_assimilated']['time_s'],35.)
        self.assertNotEqual(row['history_assimilated']['value'],row['cvae_raw']['value'])

    def test_missing_flow_does_not_return_zero_risk(self):
        from patrol_rsi.causal_observation import task_signal,reading
        bundle=capture_public(ImproverTask(specification(),PatrolProgram()))
        row=copy.deepcopy(bundle['tasks'][156])
        row['flow']=reading(source='not_observed',unit='kg/s',now=5.)
        self.assertEqual(task_signal(row),(None,'flow_unavailable'))

    def test_saved_8300_current_telemetry_restores_positive_target_benefit(self):
        from patrol_rsi.public_stopping import public_frontier
        from patrol_rsi.public_dispatch import snapshot,choose
        env=ImproverTask(specification(),PatrolProgram())
        while env.clock<370:
            groups=snapshot(env,lambda f:f[:,0]);env.step(choose(groups,'tour')[0])
        b=capture_public(env);flow=b['tasks'][156]['flow'];qualified=b['tasks'][156]['qualified']
        self.assertEqual(flow['time_s'],370.)
        self.assertEqual(qualified['time_s'],130.)
        self.assertLess(flow['value'],33.6)
        self.assertGreater(qualified['value'],33.6)
        row=next(r for r in public_frontier(b) if r['task']==156 and r['dispatch_time_s']==370)
        self.assertGreater(row['target_benefit'],0.)
        self.assertEqual(row['target_source'],'remote_flow')
        self.assertTrue(all(k in row for k in ('first_round_collateral_cost','full_delay_exposure_s','future_risk_correction','net_score')))

    def test_revised_full_episode_keeps_exact_returns_and_one_global_budget(self):
        from patrol_rsi.causal_run import run
        record,_=run(specification(8499),None,lambda f:f[:,0],'revised')
        self.assertEqual(len(record['action_log']),359)
        self.assertEqual(record['metrics']['tasks_violating_exact_gap'],0)
        self.assertLessEqual(record['metrics']['selected_detours'],1)

    def test_stopping_value_agrees_with_explicit_finite_recursion(self):
        from patrol_rsi.public_stopping import stopping_decision
        rows=[dict(agent=0,task=1,dispatch_time_s=t,net_score=v) for t,v in ((5.,.2),(10.,.1),(15.,.4),(20.,.3))]
        future=0.
        expected={}
        for row in reversed(rows):
            expected[row['dispatch_time_s']]=(row['net_score'],future)
            future=max(row['net_score'],future)
        for t,(check,wait) in expected.items():
            result=stopping_decision(rows,now=t,used=False)
            self.assertEqual((result['q_check'],result['q_wait']),(check,wait))

    def test_frozen_validation_rejects_bad_worst_seed_and_never_reads_test(self):
        from patrol_rsi.frozen_selection import select_validation
        row=lambda missed,delay:dict(missed=missed,weighted_delay_sum=delay,travel_m=100.,violations=0)
        tour={'8600':row(5,100.),'8601':row(5,100.)}
        bad={'8600':row(2,80.),'8601':row(6,80.)}
        good={'8600':row(4,100.),'8601':row(5,99.)}
        self.assertEqual(select_validation({'C0':bad},tour),'TOUR')
        self.assertEqual(select_validation({'C0':bad,'C1':good},tour),'C1')
        with self.assertRaises(ValueError):select_validation({}, {'8700':row(1,1.)})


if __name__=='__main__':
    unittest.main()
