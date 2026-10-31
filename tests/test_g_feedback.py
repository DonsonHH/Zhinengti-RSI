import json
import unittest
from patrol_rsi.timing_skill import TimingSkill
from patrol_rsi.g_feedback import difference_card,retrieve_difference_cards


class DifferenceFeedbackTests(unittest.TestCase):
    def test_shadow_decision_does_not_claim_an_unobserved_counterfactual_measurement(self):
        parent=TimingSkill.initial_e1();value=parent.to_dict();value['rules'].append({'all':[{'field':'direct_age_s','op':'is_missing','value':None}]});child=TimingSkill.from_dict(value)
        raw={'decision':{'time_s':180.,'task':0,'task_public_id':'thermal:0','agent':0,
            'features':{'kind':'thermal','professional_available':False,'professional_signal':None,'direct_available':False,'direct_age_s':None},
            'skill_action':'CHECK','executed_action':'CHECK_THIS','team_action':'CHECK','candidate_finish_s':220.,'Tour_target_finish_s':300.},
            'measurement':{'time_s':255.,'kind':'thermal','value':70.,'field':'temperature_c','unit':'degC','qualified_role':'thermal','calibrated':True,'observed_alarm':True,'transition':'new_measured_alert'},
            'elapsed_to_actual_measurement_s':75.}
        shadow={'parent_team_choice':None,'child_team_choice':{'agent':0,'task_public_id':'thermal:0'}}
        card=difference_card(parent,child,raw,shadow,'a'*64)
        self.assertEqual(card['same_input_parent_admission'],'CONTINUE_TOUR');self.assertEqual(card['same_input_child_admission'],'CHECK')
        self.assertEqual(card['actual_follow_up']['time_s'],255.)
        self.assertEqual(card['counterfactual_measurement'],'unobserved')
        self.assertEqual(len(retrieve_difference_cards([card])),1)
        self.assertNotIn('fault_start_s',json.dumps(card))
        raw['measurement']['time_s']=180.;raw['elapsed_to_actual_measurement_s']=0.
        with self.assertRaises(ValueError):difference_card(parent,child,raw,shadow,'a'*64)
        raw['measurement']['time_s']=255.;raw['elapsed_to_actual_measurement_s']=75.
        raw['decision']['features']['fault_start_s']=200
        with self.assertRaises(ValueError):difference_card(parent,child,raw,shadow,'a'*64)


if __name__=='__main__':unittest.main()
