import unittest
import json
import numpy as np
from patrol_rsi.patrol_timing import gate_candidate,gated_choices,count_later_roots
from patrol_rsi.aligned_patrol import common_one_check


class PatrolTimingTests(unittest.TestCase):
    def candidate(self,task=2,**kw):
        v={'agent':0,'task':task,'baseline_next':1,'kind':'thermal','signal_name':'thermal_excess','signal':.1,'threshold':.05,'available':True,'measurement_age_s':5.,
            'signal_source':'history_assimilated_temperature','observed_value':67.,'observed_unit':'degC','estimate_age_s':5.,
            'weight_scaled':1.,'age_ratio':.3,'eta_ratio':.1,'candidate_finish_s':100.,'tour_target_finish_s':200.,'lead_s':100.,'shift_s':20.,'already_alarm':False}
        return dict(v,**kw)
    def group(self,**kw):
        a=self.candidate(1,lead_s=0.);b=self.candidate(2);c=self.candidate(3,age_ratio=.8)
        return dict({'agent':0,'ids':np.array([1,2,3]),'baseline_next':1,'clock':5.,'scores':np.array([100.,50.,1.]),
            'features':np.array([[.1,.1],[.2,.1],[.8,.1]]),'public':[a,b,c],'detours_used':0,'baseline_completed':0},**kw)
    def test_missing_evidence_preserves_quota(self):self.assertFalse(gate_candidate(self.candidate(available=False)))
    def test_positive_gate_includes_threshold_boundary(self):self.assertTrue(gate_candidate(self.candidate(signal=.05)))
    def test_low_signal_and_excessive_shift_rejected(self):
        self.assertFalse(gate_candidate(self.candidate(signal=.0499)));self.assertFalse(gate_candidate(self.candidate(shift_s=1000.)))
    def test_alarm_or_no_lead_not_an_extra(self):
        self.assertFalse(gate_candidate(self.candidate(already_alarm=True)));self.assertFalse(gate_candidate(self.candidate(lead_s=0.)))
    def test_closed_public_interface_rejects_hidden_label(self):
        v=self.candidate();v['fault_start_s']=335
        with self.assertRaises(ValueError):gate_candidate(v)
    def test_absolute_time_is_not_a_gate_threshold(self):
        a=self.candidate();b=self.candidate(candidate_finish_s=800.,tour_target_finish_s=900.)
        self.assertEqual(gate_candidate(a),gate_candidate(b))
    def test_rankers_share_gate_but_can_choose_different_target(self):
        p,_=gated_choices([self.group()],'evidence_gate_PPO',False);u,_=gated_choices([self.group()],'evidence_gate_urgency',False)
        self.assertEqual(p[0],2);self.assertEqual(u[0],3)
    def test_both_gates_continue_Tour_without_evidence(self):
        g=self.group();g['public']=[dict(v,available=False) for v in g['public']]
        for method in ('evidence_gate_PPO','evidence_gate_urgency'):
            choices,logs=gated_choices([g],method,False);self.assertEqual(choices[0],1);self.assertEqual(logs[0]['quota_decision'],'PRESERVE_CONTINUE_TOUR')
    def test_spent_quota_forces_Tour(self):
        choices,logs=gated_choices([self.group()],'evidence_gate_PPO',True);self.assertEqual(choices[0],1);self.assertEqual(logs[0]['quota_decision'],'BUDGET_SPENT_TOUR')
    def test_shared_gate_accepts_only_one_team_check(self):
        first=self.group();second=self.group(agent=1);choices,logs=gated_choices([first,second],'evidence_gate_PPO',False)
        actual,receipt,used=common_one_check(choices,logs,False)
        self.assertTrue(used);self.assertEqual(actual,{0:2,1:1});self.assertEqual(receipt[1]['tool'],'common_budget_return_tour')
    def test_useful_root_counts_deduplicate_and_JSON_roundtrip(self):
        value=count_later_roots(np.array([5.,5.,10.,15.]),5.)
        self.assertEqual(value,2);self.assertEqual(json.loads(json.dumps({'roots':value})),{'roots':2})


if __name__=='__main__':unittest.main()
