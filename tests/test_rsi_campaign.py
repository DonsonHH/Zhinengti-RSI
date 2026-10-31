import unittest
from pathlib import Path
import json

from patrol_rsi.improver_task import PatrolProgram
from patrol_rsi.paired_campaign import protocols
from patrol_rsi.rsi_campaign import cases,prefix_program,retain_on_validation,validate_mapping,GROUPS,FatalCampaignBoundary

class RsiCampaignTests(unittest.TestCase):
    def test_fatal_model_budget_boundary_is_not_a_failed_proposal_retry(self):
        from patrol_rsi.rsi_controlled import ControlledImprover,run_slots
        class Store:
            def evaluate(self,cases,program):
                return {'case_ids':cases,'public':{'coverage':{},'professional_evidence':{},'efficiency':{}},
                    'validator':{'violating_tasks':0,'coverage':1.,'missed_segments':0,'weighted_censored_delay_s':1.,'mean_travel_m':1.}}
        calls=[]
        def generate(*_):
            calls.append(1);raise FatalCampaignBoundary('weight or budget changed')
        with self.assertRaises(FatalCampaignBoundary):run_slots(ControlledImprover(),Store(),{'qwen_once':generate},['fixture'],1,3)
        self.assertEqual(len(calls),1)
    def test_mapping_seal_recomputes_fallback_and_rejects_missing_groups(self):
        import copy
        stat={'seeds':[{'seed':32120,'macro_missed_count':1.},{'seed':32121,'macro_missed_count':1.}], 'macro_missed_count':1.,'macro_wait_s':100.}
        nominal=PatrolProgram(refresh_s=120).to_dict();tour=PatrolProgram().to_dict()
        programs={g:{'nominal_program':tour if g=='Tour' else nominal,'actual_program':tour,
            'validation_gate':{'retain':True} if g=='Tour' else retain_on_validation(stat,stat)} for g in GROUPS}
        receipt={'complete':True,'final_source_access':False,'execution_identity_SHA256':'a','programs':programs,'validation':{g:stat for g in GROUPS},'case_index':[]}
        mapping={'execution_identity_SHA256':'a','RSI_protocol_SHA256':'p','programs':programs}
        self.assertTrue(validate_mapping(receipt,mapping,'a','p'))
        broken=copy.deepcopy(mapping);broken['programs']['controlled_meta']['actual_program']=nominal
        bad=copy.deepcopy(receipt);bad['programs']=broken['programs']
        with self.assertRaises(ValueError):validate_mapping(bad,broken,'a','p')
        broken=copy.deepcopy(mapping);del broken['programs']['fixed3']
        with self.assertRaises(ValueError):validate_mapping(receipt,broken,'a','p')
    def test_separate_task_partitions_and_fixed_noise_process_counts(self):
        s=Path(__file__).resolve().parents[1]/'research/2026-10-30-m12c1b4b2-alignment'
        p,r=protocols(s/'MultiAgent-Patrol-Paired-v1.json',s/'RSI-Controlled-Meta-v1.json')
        seen=set()
        for phase in r['partitions']:
            values=cases(phase,p,r);seeds={v['seed'] for v in values}
            self.assertFalse(seeds&seen);seen|=seeds
            self.assertEqual(len(values),18*len(seeds))
            self.assertEqual({v['noise_c'] for v in values},{.2,1.})
        self.assertEqual({v['fault_start_s'] for v in cases('fresh_task_final',p,r)},{150,520})

    def test_fixed3_reuses_original_prefix_without_free_calls(self):
        a=PatrolProgram(refresh_s=60);b=PatrolProgram(refresh_s=120)
        result={'history':[{'accepted':True,'workflow':a.to_dict()},{'accepted':False,'workflow':b.to_dict()},
            {'accepted':False,'workflow':b.to_dict()},{'accepted':True,'workflow':b.to_dict()}]}
        self.assertEqual(prefix_program(result),a)
        self.assertEqual(prefix_program(result,4),b)

    def test_validation_fallback_cannot_hide_per_seed_regression(self):
        tour={'seeds':[{'seed':1,'macro_missed_count':1.},{'seed':2,'macro_missed_count':1.}], 'macro_missed_count':1.,'macro_wait_s':100.}
        same=dict(tour)
        self.assertFalse(retain_on_validation(same,tour)['retain'])
        better=tour|{'macro_wait_s':99.}
        self.assertTrue(retain_on_validation(better,tour)['retain'])
        bad=better|{'seeds':[{'seed':1,'macro_missed_count':2.},{'seed':2,'macro_missed_count':0.}]}
        self.assertFalse(retain_on_validation(bad,tour)['retain'])


if __name__=='__main__':unittest.main()
