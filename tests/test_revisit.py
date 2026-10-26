import unittest
import json
from dataclasses import asdict
import numpy as np
from patrol_rsi.environment import PatrolEnv
from patrol_rsi.experiment import room
from patrol_rsi.revisit import PatrolSkill, RevisitSkill, trajectory_diagnostics


class RevisitTests(unittest.TestCase):
    def test_protected_fields_and_resource_budget_rejected(self):
        with self.assertRaises(ValueError):PatrolSkill().proposed({'evaluator': 'replace'})
        with self.assertRaises(ValueError):PatrolSkill(estimator_samples=4,estimator_refresh_interval=1)
        with self.assertRaises(ValueError):PatrolSkill(revisit_trigger=True)

    def test_latched_skill_reaches_all_targets_with_unhelpful_policy(self):
        env=PatrolEnv(room(6000),seed=700000);skill=RevisitSkill(env,6000)
        scores=np.zeros((4,8));maximum_age=0
        for _ in range(96):
            env.step(env.safe_actions(skill.scores(env,scores,12)))
            maximum_age=max(maximum_age,int(env.raw_age.max()))
        self.assertEqual(env.evaluation()['target_coverage'],1)
        self.assertLess(maximum_age,48)
        self.assertGreater(skill.executor_steps,0)

    def test_skill_uses_public_age_and_geometry_only(self):
        s=room(6001)
        a=PatrolEnv(s,truth=np.zeros((24,24)));b=PatrolEnv(s,truth=np.ones((24,24)))
        a.raw_age[s.targets]=30;b.raw_age[s.targets]=30
        score=np.zeros((4,8))
        np.testing.assert_array_equal(RevisitSkill(a,6001).scores(a,score,16),RevisitSkill(b,6001).scores(b,score,16))

    def test_disabled_skill_preserves_scores(self):
        env=PatrolEnv(room(6002));scores=np.random.default_rng(1).random((4,8))
        np.testing.assert_array_equal(RevisitSkill(env,6002).scores(env,scores,0),scores)

    def test_diagnostic_tools_expose_different_trace_views(self):
        env=PatrolEnv(room(6003));goals=env.scenario.targets
        ages=[env.raw_age[goals].copy()];positions=[env.positions.copy()]
        for _ in range(3):
            env.step(np.full(4,-1));ages.append(env.raw_age[goals].copy());positions.append(env.positions.copy())
        coverage=trajectory_diagnostics(env,ages,positions,'coverage')
        revisit=trajectory_diagnostics(env,ages,positions,'revisit')
        oscillation=trajectory_diagnostics(env,ages,positions,'oscillation')
        self.assertEqual(len(coverage['never_observed_positions']),8)
        self.assertTrue(all(r['step'] is None for r in coverage['first_observation_step']))
        self.assertIn('longest_unobserved',revisit)
        self.assertIn('two_step_returns_per_executor',oscillation)
        json.dumps([coverage,revisit,oscillation])
