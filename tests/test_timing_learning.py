"""Public proposal, feedback and selection boundaries for timing-skill learning."""
import json
import unittest

from patrol_rsi.timing_skill import TimingSkill
from patrol_rsi.timing_learning import parse_skill_reply,skill_context,project_card,sample_skill,aggregate,strict_gain,parse_meta_reply,retrieve_cards,exact_policy_result
from patrol_rsi.rsi_controlled import ControlledImprover


class TimingLearningTests(unittest.TestCase):
    def test_proposal_accepts_only_typed_skill_without_hidden_trigger_fields(self):
        skill=TimingSkill.initial_e1().to_dict()
        reply={'diagnosis':'Use available professional evidence.','skill':skill,'expected_effect':'Earlier qualified check.'}
        self.assertEqual(parse_skill_reply(json.dumps(reply)).to_dict(),skill)
        reply['skill']['rules'][0]['all'][0]={'field':'fault_start_s','op':'ge','value':335}
        with self.assertRaises(ValueError):parse_skill_reply(json.dumps(reply))

    def test_prior_context_cannot_receive_acceptance_or_selected_parent(self):
        initial=TimingSkill.initial_e1()
        history=[{'skill':initial.to_dict(),'accepted':True,'score':{'missed_segments':0},'cards':[{'seed':33000}]}]
        value=skill_context(initial,history,[],prior_only=True)
        self.assertEqual(value['initial_skill'],initial.to_dict())
        self.assertEqual(value['proposal_history'],[{'skill':initial.to_dict()}])
        text=json.dumps(value)
        self.assertNotIn('accepted',text);self.assertNotIn('missed_segments',text);self.assertNotIn('33000',text)

    def test_experience_uses_only_later_actual_qualified_measurements(self):
        card={'decision':{'time_s':100.,'task_public_id':'electrical:0','agent':0,
            'features':{'kind':'electrical','direct_available':False,'direct_age_s':None},
            'skill_action':'CONTINUE_TOUR','executed_action':'CONTINUE_TOUR','candidate_finish_s':120.,'Tour_target_finish_s':150.},
            'measurement':{'time_s':140.,'kind':'electrical','value':70.,'field':'contact_temperature_c','unit':'degC',
                'qualified_role':'electrical','calibrated':True,'observed_alarm':True,'transition':'new_measured_alert'},
            'elapsed_to_actual_measurement_s':40.}
        self.assertEqual(project_card(card)['measurement']['value'],70.)
        value=skill_context(TimingSkill.initial_e1(),[],[card])
        self.assertEqual(len(value['experience_cards']),1)
        card['measurement']['time_s']=90.
        with self.assertRaises(ValueError):project_card(card)

    def test_simple_search_is_seeded_and_stays_in_the_same_skill_language(self):
        skills=[sample_skill(seed).to_dict() for seed in range(20)]
        self.assertEqual(skills,[sample_skill(seed).to_dict() for seed in range(20)])
        for value in skills:self.assertEqual(TimingSkill.from_dict(value).to_dict(),value)
        self.assertGreater(len({json.dumps(v,sort_keys=True) for v in skills}),10)

    def test_macro_selection_weights_processes_not_event_or_noise_counts(self):
        rows=[]
        for seed in (1,2):
            for family,miss,wait in [('healthy',0,None),('fault_a',2,100.),('fault_b',0,20.)]:
                for noise in (.2,1.):
                    rows.append({'specification':{'seed':seed,'scenario':family,'duration_s':180,'fault_start_s':95,'noise_c':noise},
                        'process_fingerprint':f'{seed}-{family}','event_eligibility_SHA256':family,
                        'metrics':{'anomaly_segments':0 if family=='healthy' else 10,'missed_segments':miss,'consequence_weighted_censored_delay_s':wait,'selected_detours':0,'travel_m':100.}})
        expected=[v['specification'] for v in rows]
        base=aggregate(rows,expected)
        self.assertAlmostEqual(base['macro_missed_count'],2/3)
        self.assertEqual(base['macro_wait_s'],60.)
        better=json.loads(json.dumps(rows))
        for row in better:
            if row['metrics']['consequence_weighted_censored_delay_s'] is not None:
                row['metrics']['consequence_weighted_censored_delay_s']*=.9
        self.assertTrue(strict_gain(aggregate(better,expected),base))
        better[2]['metrics']['missed_segments']=4
        self.assertFalse(strict_gain(aggregate(better,expected),base))

    def test_aggregation_rejects_missing_paired_noise(self):
        rows=[{'specification':{'seed':1,'scenario':'healthy','duration_s':180,'fault_start_s':95,'noise_c':.2},
            'metrics':{'missed_segments':0,'consequence_weighted_censored_delay_s':None,'selected_detours':0,'travel_m':100.}}]
        expected=[rows[0]['specification'],dict(rows[0]['specification'],noise_c=1.)]
        with self.assertRaises(ValueError):aggregate(rows,expected)

    def test_meta_proposal_cannot_relax_acceptance_or_failure_contract(self):
        reply={'diagnosis':'Change retrieval.','protocol':ControlledImprover().to_dict(),'expected_effect':'More useful proposals.'}
        self.assertEqual(parse_meta_reply(json.dumps(reply)),ControlledImprover())
        reply['protocol']['acceptance']='keep_parent'
        with self.assertRaises(ValueError):parse_meta_reply(json.dumps(reply))

    def test_raw_proposals_reject_duplicate_JSON_keys(self):
        skill=json.dumps(TimingSkill.initial_e1().to_dict())
        raw='{"diagnosis":"first","diagnosis":"second","skill":'+skill+',"expected_effect":"bounded"}'
        with self.assertRaises(ValueError):parse_skill_reply(raw)

    def test_selection_rejects_nonfinite_metrics_and_duplicate_numerical_processes(self):
        rows=[{'specification':{'seed':1,'scenario':'healthy','duration_s':180,'fault_start_s':95,'noise_c':noise},
            'process_fingerprint':'same','event_eligibility_SHA256':'empty',
            'metrics':{'anomaly_segments':0,'missed_segments':0,'consequence_weighted_censored_delay_s':None,'selected_detours':0,'travel_m':100.}} for noise in (.2,1.)]
        expected=[v['specification'] for v in rows]
        rows[0]['metrics']['travel_m']=float('nan')
        with self.assertRaises(ValueError):aggregate(rows,expected)
        rows[0]['metrics']['travel_m']=100.
        duplicate=json.loads(json.dumps(rows))
        for row in duplicate:row['specification']['scenario']='another_label_same_physics'
        with self.assertRaises(ValueError):aggregate(rows+duplicate,expected+[v['specification'] for v in duplicate])

    def test_identical_rejected_proposals_keep_their_actual_history_positions(self):
        history=[{'skill':TimingSkill.initial_e1().to_dict(),'accepted':False} for _ in range(3)]
        context=skill_context(TimingSkill.initial_e1(),history,[])
        self.assertEqual([v['previous_proposal_offset'] for v in context['acceptance_feedback']],[2,1])

    def test_truncated_card_prefix_preserves_one_actual_measurement_per_profession(self):
        fields={'thermal':('temperature_c','degC'),'electrical':('contact_temperature_c','degC'),
            'cooling':('pod_flow_kg_s','kg/s'),'control':('controller_sensor_error_c','degC')}
        cards=[]
        for kind,(field,unit) in fields.items():
            for alarm in (True,False):
                measured={'thermal':(70.,50.),'electrical':(70.,38.),'cooling':(20.,40.),'control':(3.,0.)}[kind][0 if alarm else 1]
                cards.append({'decision':{'time_s':100.,'task_public_id':kind+':0','agent':0,'features':{'kind':kind},
                    'skill_action':'CONTINUE_TOUR','executed_action':'CONTINUE_TOUR','candidate_finish_s':120.,'Tour_target_finish_s':150.},
                    'measurement':{'time_s':140.,'kind':kind,'value':measured,'field':field,'unit':unit,'qualified_role':kind,'calibrated':True,
                        'observed_alarm':alarm,'transition':'new_measured_alert' if alarm else 'normal_check'},'elapsed_to_actual_measurement_s':40.})
        prefix=retrieve_cards(cards,('professional_evidence',))[:4]
        self.assertEqual({v['measurement']['kind'] for v in prefix},set(fields))

    def test_final_lookup_preserves_distinct_legal_numeric_JSON_representations(self):
        integer={'executor':'timing-skill-v1','ranker':'urgency','skill':TimingSkill.initial_e1().to_dict()}
        floating=json.loads(json.dumps(integer));floating['skill']['rules'][0]['all'][3]['value']=0.0
        TimingSkill.from_dict(floating['skill'])
        self.assertEqual(integer,floating)
        results=[{'policy':integer,'key':'integer-run'},{'policy':floating,'key':'float-run'}]
        self.assertEqual(exact_policy_result(results,integer)['key'],'integer-run')
        self.assertEqual(exact_policy_result(results,floating)['key'],'float-run')


if __name__=='__main__':unittest.main()
