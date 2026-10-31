import unittest

from patrol_rsi.timing_skill import TimingSkill


class TimingSkillTests(unittest.TestCase):
    def test_editable_skill_can_acquire_missing_professional_measurement(self):
        public={'kind':'electrical','professional_available':False,'professional_signal':None,'already_alarm':False,
            'direct_available':False,'direct_age_s':None,'inspection_age_ratio':.75,'lead_s':120.,'shift_s':20.,'importance':.75}
        acquisition=TimingSkill.from_dict({'schema_version':'timing-skill-v1','rules':[{'all':[
            {'field':'direct_age_s','op':'is_missing','value':None},
            {'field':'inspection_age_ratio','op':'ge','value':.5},
            {'field':'already_alarm','op':'eq','value':False},
            {'field':'lead_s','op':'ge','value':60},
            {'field':'shift_s','op':'le','value':30}]}],'default_action':'CONTINUE_TOUR'})
        self.assertEqual(TimingSkill.initial_e1().action(public),'CONTINUE_TOUR')
        self.assertEqual(acquisition.action(public),'CHECK')

    def test_missing_direct_reading_is_not_a_known_abnormal_or_normal_reading(self):
        rule={'schema_version':'timing-skill-v1','rules':[{'all':[{'field':'direct_reading_normal','op':'eq','value':False}]}],'default_action':'CONTINUE_TOUR'}
        skill=TimingSkill.from_dict(rule)
        self.assertEqual(skill.action({'direct_reading_normal':None}),'CONTINUE_TOUR')
        self.assertEqual(skill.action({'direct_available':True,'direct_age_s':0.,'direct_reading_normal':True}),'CONTINUE_TOUR')
        self.assertEqual(skill.action({'direct_available':True,'direct_age_s':0.,'direct_reading_normal':False}),'CHECK')

    def test_professional_reference_threshold_requires_a_known_profession(self):
        public={'professional_available':True,'professional_signal':.1,'already_alarm':False,'lead_s':120.,'shift_s':10.,'importance':.75}
        self.assertEqual(TimingSkill.initial_e1().action(public),'CONTINUE_TOUR')

    def test_direct_constructor_cannot_create_an_empty_true_rule(self):
        with self.assertRaises(ValueError):TimingSkill(((),))

    def test_malformed_field_and_huge_threshold_are_typed_rejections(self):
        for atom in ({'field':[],'op':'ge','value':0},{'field':'lead_s','op':'ge','value':10**500}):
            value={'schema_version':'timing-skill-v1','rules':[{'all':[atom]}],'default_action':'CONTINUE_TOUR'}
            with self.subTest(atom=atom),self.assertRaises(ValueError):TimingSkill.from_dict(value)

    def test_unavailable_direct_measurement_cannot_be_a_known_normal_or_abnormal(self):
        skill=TimingSkill.from_dict({'schema_version':'timing-skill-v1','rules':[{'all':[{'field':'direct_reading_normal','op':'eq','value':False}]}],'default_action':'CONTINUE_TOUR'})
        with self.assertRaises(ValueError):skill.action({'direct_available':False,'direct_reading_normal':False})

    def test_initial_skill_reports_the_actual_matching_rule(self):
        public={'kind':'thermal','professional_available':True,'professional_signal':.1,'already_alarm':False,'lead_s':120.,'shift_s':20.,'importance':.75}
        self.assertEqual(TimingSkill.initial_e1().decide(public),{'action':'CHECK','matched_rule':0})


if __name__=='__main__':unittest.main()
