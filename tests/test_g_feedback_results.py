import unittest
from patrol_rsi.g_feedback_results import OptionalResultIndex


class OptionalResultTests(unittest.TestCase):
    def test_later_positive_tour_reading_is_not_the_earlier_optional_result(self):
        events=[{'task_id':'thermal:0','task':0,'agent':0,'time_s':220.,'kind':'thermal','role':'thermal','calibrated':True,'alarm':False,'measurement_transition':'normal_check','measurement':{'temperature_c':50.}},
            {'task_id':'thermal:0','task':0,'agent':0,'time_s':255.,'kind':'thermal','role':'thermal','calibrated':True,'alarm':True,'measurement_transition':'new_measured_alert','measurement':{'temperature_c':70.}}]
        certs=[{'clock_s':180.,'agent':0,'task':0,'baseline_next':1,'certificate':{'candidate_finish_s':220.}}]
        card={'decision':{'time_s':180.,'agent':0,'task_public_id':'thermal:0','executed_action':'CHECK_THIS'},
            'public_effect':'selected_optional','actual_follow_up':{'time_s':255.,'kind':'thermal','observed_alarm':True}}
        value=OptionalResultIndex(certs,events).annotate(card)
        self.assertEqual(value['direct_optional_measurement']['time_s'],220.)
        self.assertFalse(value['direct_optional_measurement']['observed_alarm'])
        self.assertFalse(value['follow_up_is_optional_completion'])
        self.assertFalse(value['direct_optional_positive'])
        self.assertEqual(value['actual_follow_up']['time_s'],255.)


if __name__=='__main__':unittest.main()
