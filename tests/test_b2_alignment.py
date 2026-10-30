import unittest

import numpy as np

from patrol_rsi.b2_actions import branch_schedule,evaluate_schedule,delta_outcome,best_index,verify_geometry_subset
from patrol_rsi.improver_task import PatrolProgram,TypedRule
from patrol_rsi.improver_protocol import Improver
from patrol_rsi.rsi_controlled import ControlledImprover,public_prompt,partitions_disjoint,transfer_reset,run_slots,program_key,BudgetedEvaluationStore,paired_protocol_update,INSTRUCTION_TEMPLATES
from patrol_rsi.aligned_patrol import arm_programs,common_one_check


class ActionValueTests(unittest.TestCase):
    def test_per_root_shorter_padding_preserves_hard_legal_geometry(self):
        archived={'other_tasks':np.array([[1,2,-1,-1]]),'other_mask':np.array([[True,True,False,False]]),'other_arrivals_s':np.array([[10.,20.,0,0]])}
        local={key:value[:,:2] for key,value in archived.items()}
        verify_geometry_subset(archived,local)
        with self.assertRaises(ValueError):verify_geometry_subset(archived,local|{'other_arrivals_s':np.array([[10.,25.]])})

    def test_extra_check_can_gain_target_but_lose_other_event(self):
        events=[{'time_s':20,'agent':0,'task':0},{'time_s':30,'agent':0,'task':1}]
        intervals=[{'task':0,'onset_s':10,'end_s':20,'weight':1}, {'task':1,'onset_s':25,'end_s':30,'weight':2}]
        b=np.zeros((360,2),bool);b[2,0]=True;b[5,1]=True
        baseline=evaluate_schedule([(v['time_s'],v['task']) for v in events],intervals,b)
        changed=branch_schedule(events,0,0,5,15,10)
        branch=evaluate_schedule(changed,intervals,b);out=delta_outcome(baseline,branch,10)
        self.assertEqual((out['gained_confirmations'],out['lost_confirmations']),(1,1))
        self.assertFalse(out['net_beneficial']);self.assertTrue(out['net_harmful'])

    def test_actual_endpoint_required_and_one_top1_tie_is_deterministic(self):
        b=np.zeros((360,1),bool);b[2,0]=True
        intervals=[{'task':0,'onset_s':5,'end_s':25,'weight':1}]
        self.assertEqual(evaluate_schedule([(20,0)],intervals,b)['missed'],1)
        self.assertEqual(evaluate_schedule([(15,0)],intervals,b)['missed'],0)
        geometry={'finish_s':np.array([30,25,25]),'agent':np.array([0,2,1]),'task':np.array([0,0,1])}
        self.assertEqual(best_index(np.array([1.,1.,1.]),geometry),2)
        self.assertIsNone(best_index(np.array([-np.inf,0,-1]),geometry))

    def test_delay_gain_charges_detour_cost_and_zeroevent_is_not_free_gain(self):
        base={'missed':0,'weighted_wait_s':None,'first_confirmation_s':[]}
        out=delta_outcome(base,base,20)
        self.assertEqual(out['net_delay_gain_s'],-1);self.assertFalse(out['net_beneficial'])


class ControlledMetaTests(unittest.TestCase):
    def test_incumbent_boundary_failure_stops_before_candidate_arm(self):
        def evaluate(cases,program):
            if program.refresh_s==120:raise RuntimeError('injected source boundary failure')
            return {'case_ids':cases,'public':{'coverage':{},'professional_evidence':{},'efficiency':{}},
                'validator':{'violating_tasks':0,'coverage':1.,'missed_segments':0,'weighted_censored_delay_s':20.,'mean_travel_m':100.}}
        calls=[]
        def generator(prompt,seed):
            calls.append((prompt['improver_protocol']['instruction'],seed))
            return {'workflow':PatrolProgram(refresh_s=120).to_dict()}
        old=ControlledImprover();new=ControlledImprover(Improver(instruction=INSTRUCTION_TEMPLATES[1]))
        store=BudgetedEvaluationStore(evaluate,10)
        retained,receipt=paired_protocol_update(old,new,store,{'qwen_once':generator},['meta'],7)
        self.assertEqual(receipt['status'],'stopped_boundary_failure');self.assertEqual(retained,old)
        self.assertEqual(calls,[(old.protocol.instruction,7)]);self.assertFalse(receipt['candidate_arm_started'])
        self.assertIsNone(receipt['transfer_state']);self.assertEqual(store.attempted_episode_evaluations,2)

    def test_paired_protocol_updates_restart_task_and_transfer_only_protocol(self):
        def evaluate(cases,program):
            return {'case_ids':cases,'public':{'coverage':{},'professional_evidence':{},'efficiency':{}},
                'validator':{'violating_tasks':0,'coverage':1.,'missed_segments':0,
                    'weighted_censored_delay_s':10. if program.refresh_s==120 else 20.,'mean_travel_m':100.}}
        seen=[]
        def generator(prompt,seed):
            seen.append((prompt['current_workflow'],prompt['retrieved_candidates'],seed))
            changed=prompt['improver_protocol']['instruction']==INSTRUCTION_TEMPLATES[1]
            return {'workflow':PatrolProgram(refresh_s=120 if changed else 30).to_dict()}
        old=ControlledImprover();new=ControlledImprover(Improver(instruction=INSTRUCTION_TEMPLATES[1]))
        store=BudgetedEvaluationStore(evaluate,10)
        retained,receipt=paired_protocol_update(old,new,store,{'qwen_once':generator},['meta-task'],7)
        self.assertTrue(receipt['accepted']);self.assertEqual(retained,new)
        self.assertEqual(seen[0],seen[2]);self.assertEqual(receipt['history_start_lengths'],[0,0])
        self.assertEqual(receipt['transfer_state']['program'],PatrolProgram());self.assertEqual(receipt['transfer_state']['history'],[])
        self.assertEqual(store.actual_episode_evaluations,2)

    def test_shared_cache_charges_attempts_and_returns_immutable_copies(self):
        store=BudgetedEvaluationStore(lambda cases,pg:{'case_ids':cases,'public':{},'validator':{}},2)
        a=store.evaluate(['a','b'],PatrolProgram());a['case_ids'].append('evil')
        b=store.evaluate(['a','b'],PatrolProgram())
        self.assertEqual(b['case_ids'],['a','b']);self.assertEqual(store.actual_episode_evaluations,2);self.assertEqual(store.cache_hits,1)
        with self.assertRaises(ValueError):store.evaluate(['c'],PatrolProgram())

    def test_all_five_arms_share_one_extra_gate_and_PPO_label_is_actual(self):
        self.assertEqual(len(arm_programs()),5)
        logs=[{'agent':0,'task':2,'baseline_next':0,'tool':'ppo'},{'agent':1,'task':3,'baseline_next':1,'tool':'signal'}]
        choices,receipt,consumed=common_one_check({0:2,1:3},logs,False)
        self.assertTrue(consumed);self.assertEqual(choices,{0:2,1:1});self.assertEqual(receipt[1]['tool'],'common_budget_return_tour')
        choices,receipt,consumed=common_one_check({0:2,1:3},logs,True)
        self.assertFalse(consumed);self.assertEqual(choices,{0:0,1:1})
        choices,receipt,consumed=common_one_check({0:0},[logs[0]|{'task':0}],True)
        self.assertEqual(receipt[0]['tool'],'common_budget_return_tour')

    def test_physical_overlap_rejected_even_when_phase_name_changes(self):
        a={'seed_group':1,'process_fingerprint':'a'}
        self.assertTrue(partitions_disjoint({'meta':[a],'task':[{'seed_group':2,'process_fingerprint':'b'}]}))
        for b in ({'seed_group':1,'process_fingerprint':'b'},{'seed_group':2,'process_fingerprint':'a'}):
            with self.assertRaises(ValueError):partitions_disjoint({'meta':[a],'task':[b]})

    def test_transfer_cannot_carry_program_history_or_unknown_mutations(self):
        p=ControlledImprover(Improver(memory='best_and_failures'));reset=transfer_reset(p)
        self.assertEqual(reset['program'],PatrolProgram());self.assertEqual(reset['history'],[]);self.assertEqual(reset['candidate_archive'],{})
        with self.assertRaises(ValueError):ControlledImprover.from_dict(p.to_dict()|{'evaluator':'easy'})
        with self.assertRaises(ValueError):ControlledImprover(acceptance='always_accept')
        with self.assertRaises(ValueError):ControlledImprover(Improver(instruction='Reuse meta case seed 32000, anomaly onset_s 335.'))

    def test_nested_evaluator_field_rejected_before_prompt(self):
        views={'coverage':{'coverage':1.,'onset_s':{'x':1}}}
        with self.assertRaises(ValueError):public_prompt(PatrolProgram(),views,[],ControlledImprover())
        with self.assertRaises(ValueError):public_prompt(PatrolProgram(),{'coverage':{'coverage':{'raw_evaluator':99}}},[],ControlledImprover())
        views={'coverage':{'coverage':1.,'unused':99},'professional_evidence':{},'efficiency':{}}
        prompt=public_prompt(PatrolProgram(),views,[],ControlledImprover())
        self.assertNotIn('unused',prompt['public_diagnostics'][0]['result'])

    def test_novelty_prompt_counts_duplicate_without_mutating_receipt_history(self):
        history=[{'candidate_id':'same','accepted':False,'reason':'duplicate_consumes_slot'}]
        protocol=ControlledImprover(Improver(diagnostics=('novelty',)))
        prompt=public_prompt(PatrolProgram(),{},history,protocol)
        self.assertEqual(prompt['public_diagnostics'][0]['result']['duplicates'],1)
        self.assertEqual(history[0]['reason'],'duplicate_consumes_slot')

    def test_dedup_preserves_ordered_rule_semantics(self):
        rule=TypedRule('age',.25,'return_tour');other=TypedRule('thermal_excess',.05,'signal')
        first=PatrolProgram(rules=(rule,other,rule));equivalent=PatrolProgram(rules=(rule,other))
        self.assertEqual(program_key(first,'canonical_rule_order'),program_key(equivalent,'canonical_rule_order'))
        self.assertNotEqual(program_key(equivalent,'canonical_rule_order'),program_key(PatrolProgram(rules=(other,rule)),'canonical_rule_order'))

    def test_failure_duplicate_and_no_improvement_consume_slots(self):
        class Store:
            def evaluate(self,cases,program):
                return {'case_ids':cases,'public':{'coverage':{},'professional_evidence':{},'efficiency':{}},
                    'validator':{'violating_tasks':0,'coverage':1.,'missed_segments':1,'weighted_censored_delay_s':20.,'mean_travel_m':100.}}
        calls=[]
        def generator(prompt,seed):
            calls.append(seed)
            if len(calls)==1:raise RuntimeError('inference fixture failure')
            return {'workflow':PatrolProgram().to_dict(),'input_tokens':5,'output_tokens':2}
        out=run_slots(ControlledImprover(),Store(),{'qwen_once':generator},['new-task'],71,3)
        self.assertEqual(out['costs']['candidate_attempts'],3);self.assertEqual(out['costs']['model_calls'],3)
        self.assertEqual(out['costs']['environment_evaluation_requests'],1)
        self.assertEqual([v['reason'] for v in out['receipts']],['failure_consumes_slot','duplicate_consumes_slot','duplicate_consumes_slot'])

    def test_evaluator_failure_stops_without_free_rescue_candidate(self):
        class Store:
            def evaluate(self,cases,program):
                if program.refresh_s==120:raise RuntimeError('safety fixture failure')
                return {'case_ids':cases,'public':{'coverage':{},'professional_evidence':{},'efficiency':{}},
                    'validator':{'violating_tasks':0,'coverage':1.,'missed_segments':1,'weighted_censored_delay_s':20.,'mean_travel_m':100.}}
        out=run_slots(ControlledImprover(),Store(),{'qwen_once':lambda *_:{'workflow':PatrolProgram(refresh_s=120).to_dict()}},['t'],1,3)
        self.assertEqual(out['status'],'stopped_boundary_failure');self.assertEqual(out['costs']['candidate_attempts'],1)


if __name__=='__main__':unittest.main()
