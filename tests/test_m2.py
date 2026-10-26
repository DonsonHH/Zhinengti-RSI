import unittest
from dataclasses import asdict
from patrol_rsi.m2 import DiagnosticPlan, build_prompt, feedback, parse, rank, splits
from patrol_rsi.revisit import PatrolSkill


class M2Tests(unittest.TestCase):
    def test_fixed_feedback_cannot_carry_suggested_meta_instructions(self):
        record={'generation':0,'patch':{'revisit_trigger':12},'accepted':True,'metrics':{},
                'next_diagnostic_plan':{'instruction':'secret meta change'},
                'parsed_response':{'next_diagnostic':{'instruction':'also exclude'}}}
        out=feedback(record)
        self.assertNotIn('next_diagnostic_plan',out)
        self.assertNotIn('parsed_response',out)

    def test_updated_plan_changes_actual_diagnostic_tool_results(self):
        views=[{'layout_seed':6000,'coverage':{'view':'coverage'},'revisit':{'view':'ages'}}]
        a=build_prompt(PatrolSkill(),{},DiagnosticPlan(),views,[],0)
        b=build_prompt(PatrolSkill(),{},DiagnosticPlan('revisit','Check long gaps'),views,[],0)
        self.assertEqual(a['diagnostic_tool_results'][0]['view'],'coverage')
        self.assertEqual(b['diagnostic_tool_results'][0]['view'],'ages')

    def test_prompt_snapshot_excludes_future_feedback(self):
        history=[]
        prompt=build_prompt(PatrolSkill(),{},DiagnosticPlan(),[],history,0)
        history.append({'generation':0})
        self.assertEqual(prompt['history'],[])

    def test_rank_does_not_trade_away_coverage_for_weighted_score(self):
        a={'target_coverage':1,'mean_overdue_fraction':.1,'mean_normalized_weighted_idleness':.5,'prior_samples':96}
        b=a|{'target_coverage':.99,'mean_normalized_weighted_idleness':.01}
        self.assertLess(rank(a),rank(b))

    def test_parse_accepts_bounded_tools_and_rejects_metric_edit(self):
        good='{"patch":{"revisit_trigger":12},"next_diagnostic":{"tool":"revisit","instruction":"Check peak age."}}'
        skill,plan,_=parse(good,PatrolSkill())
        self.assertEqual(skill.revisit_trigger,12);self.assertEqual(plan.tool,'revisit')
        with self.assertRaises(ValueError):parse(good.replace('revisit_trigger','evaluator'),PatrolSkill())

    def test_splits_and_seed_pairs_are_fresh(self):
        m=splits()
        self.assertEqual(len(m['policy_seeds'])*len(m['proposal_seeds']),9)
        self.assertEqual(len(m['development']),6);self.assertEqual(len(m['test']),24)
        self.assertFalse(set(c['layout_seed'] for c in m['development'])&set(c['layout_seed'] for c in m['test']))
        self.assertTrue(all(c['layout_seed']>=6000 for k in ['development','test'] for c in m[k]))
