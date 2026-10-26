import json
from dataclasses import asdict
import unittest
from patrol_rsi.m2 import DiagnosticPlan
from patrol_rsi.m3 import build_prompt, config_space, contract, source_plan
from patrol_rsi.revisit import PatrolSkill


class M3Tests(unittest.TestCase):
    def test_enumeration_covers_every_unique_admissible_configuration(self):
        space=config_space()
        self.assertEqual(len(space),110)
        self.assertEqual(len({s.version_id for s in space}),110)
        self.assertIn(PatrolSkill(revisit_trigger=0),space)
        self.assertIn(PatrolSkill(revisit_trigger=12),space)

    def test_transfer_whitelist_excludes_source_task_and_test_fields(self):
        summary={'test':'SECRET_SOURCE_TEST','retained':{'updated_improver':{
            'skill':{'revisit_trigger':12},'development':'SOURCE_SCORES',
            'final_diagnostic_plan':{'tool':'revisit','instruction':'Check long gaps.'}}}}
        plan=source_plan(summary)
        self.assertEqual(plan.tool,'revisit')
        self.assertNotIn('SECRET',json.dumps(asdict(plan)))
        self.assertNotIn('revisit_trigger',json.dumps(asdict(plan)))

    def test_prompt_has_no_answer_example_and_no_history_alias(self):
        history=[]
        view=[{'layout_seed':8000,'coverage':{'tool':'coverage'}}]
        prompt=build_prompt(PatrolSkill(),{},DiagnosticPlan(),view,history,0)
        history.append({'generation':0})
        self.assertEqual(prompt['prior_task_feedback'],[])
        self.assertNotIn('JSON_shape_example_only',prompt)
        self.assertNotIn('patch',prompt)  # Grammar is described; no filled answer.

    def test_noise_is_paired_on_same_case_not_confounded_with_layout(self):
        tests=contract()['test']
        for a,b in zip(tests[:12],tests[12:]):
            self.assertEqual(a['layout_seed'],b['layout_seed'])
            self.assertEqual(a['field_seed'],b['field_seed'])
            self.assertNotEqual(a['noise'],b['noise'])

    def test_fresh_development_and_test_sets_disjoint(self):
        c=contract()
        self.assertTrue(all(x['layout_seed']>=8000 for x in c['development']))
        self.assertFalse({x['layout_seed'] for x in c['development']}&{x['layout_seed'] for x in c['test']})
