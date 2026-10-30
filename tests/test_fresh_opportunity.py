import copy
from pathlib import Path
import unittest

from patrol_rsi.fresh_opportunity import coverage_gate, fingerprint_conflicts, frozen_inputs, ROOTS


class FreshOpportunityTests(unittest.TestCase):
    def test_exact_frozen_inventory_and_disjoint_splits(self):
        base=Path(__file__).resolve().parents[1]/'research/2026-10-30-m12c1b4b0-data'
        protocol, rows=frozen_inputs(base/'B4-B1冻结协议.json',base/'B4-B1新数据清单.json')
        self.assertEqual(len(rows),1280)
        self.assertEqual(len({r['physical_key'] for r in rows}),640)
        self.assertEqual(len({r['group_seed'] for r in rows}),64)
        self.assertEqual(len(ROOTS),59)
        for seed in {r['group_seed'] for r in rows}:
            self.assertEqual(len({r['split'] for r in rows if r['group_seed']==seed}),1)
        self.assertEqual(protocol['candidate_budget']['optional_small_temporal']['training_budget'],0)

    def test_repeated_positive_rows_do_not_satisfy_event_or_seed_gate(self):
        minimum={'positive_seed_groups':8,'positive_numeric_processes':12,
            'distinct_retained_positive_events':20,'valid_negative_seed_groups':12}
        row={'B_positive_seed_groups':[20000],'B_positive_numeric_processes':1,
            'distinct_B_positive_retained_events':1,'B_negative_seed_groups':list(range(12)),
            'B_action_positive':1000000}
        value=coverage_gate(row,minimum)
        self.assertFalse(value['passed'])
        self.assertEqual(value['deficits'],{'positive_seed_groups':7,'positive_numeric_processes':11,
            'distinct_retained_positive_events':19,'valid_negative_seed_groups':0})

    def test_covered_subset_does_not_turn_missing_profession_into_GO(self):
        minimum={'positive_seed_groups':4,'positive_numeric_processes':6,
            'distinct_retained_positive_events':8,'valid_negative_seed_groups':6}
        good={'B_positive_seed_groups':list(range(4)),'B_positive_numeric_processes':6,
            'distinct_B_positive_retained_events':8,'B_negative_seed_groups':list(range(6))}
        bad=copy.deepcopy(good);bad['B_positive_seed_groups']=[]
        self.assertTrue(coverage_gate(good,minimum)['passed'])
        self.assertFalse(coverage_gate(bad,minimum)['passed'])

    def test_numeric_conflict_is_across_splits_not_names_or_noise(self):
        rows=[{'id':'phase335-low','split':'development','numeric_process_fingerprint':'abc'},
            {'id':'phase705-high','split':'development','numeric_process_fingerprint':'abc'}]
        self.assertEqual(fingerprint_conflicts(rows),[])
        rows.append({'id':'newname','split':'final_test','numeric_process_fingerprint':'abc'})
        value=fingerprint_conflicts(rows)
        self.assertEqual(len(value),1)
        self.assertEqual(set(value[0]['splits']),{'development','final_test'})


if __name__=='__main__':unittest.main()
