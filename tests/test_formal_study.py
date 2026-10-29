import unittest

from patrol_rsi.formal_study import conditions,aggregate,development_selection,method_definitions,validation_selection


class FormalStudyTests(unittest.TestCase):
    def test_conditions_exactly_match_frozen_phase_seed_noise_budget(self):
        from pathlib import Path
        import json
        root=Path(__file__).resolve().parents[1]
        p=json.loads((root/'research/2026-10-29-m12c1b31-interface/正式实验冻结方案.json').read_bytes())
        rows=conditions(p,'development',0)
        self.assertEqual(len(rows),16)
        self.assertEqual({r['seed'] for r in rows},{8500})
        self.assertEqual({r['fault_start_s'] for r in rows},{335,705})
        self.assertEqual({r['noise_c'] for r in rows},{.2,1.})
        self.assertTrue(all(r['estimator']=='history' for r in rows))
        with self.assertRaises(ValueError):conditions(p,'test',4)

    def test_development_ranking_uses_misses_then_pooled_wait_then_travel_then_id(self):
        rows=[]
        for identity,missed,delay,travel in [('C0',1,10.,5.),('C1',0,20.,10.),('C2',0,10.,10.),('C3',0,10.,10.)]:
            rows.extend(self.row(identity,8500,missed,delay,travel) for _ in range(32))
        selected=development_selection(rows)
        self.assertEqual(selected['ranked'],['C2','C3','C1','C0'])
        self.assertEqual(selected['retained'],['C2','C3'])

    def row(self,alias,seed,missed=1,delay=10.,travel=10.):
        return dict(alias=alias,specification={'seed':seed},event_weight_sum=1.,weighted_delay_sum=delay,wall_s=1.,
            metrics=dict(anomaly_segments=2,missed_segments=missed,travel_m=travel,service_s=1.,selected_detours=0,tasks_violating_exact_gap=0))

    def test_worst_seed_regression_causes_tour_fallback_despite_good_average(self):
        rows=[]
        for seed in (8600,8601):
            rows.extend(self.row('tour',seed) for _ in range(16))
            rows.extend(self.row('C0',seed,0 if seed==8600 else 2,5.) for _ in range(16))
            rows.extend(self.row('C1',seed) for _ in range(16))
        selected=validation_selection(rows,['C0','C1'])
        self.assertEqual(selected['selected'],'TOUR')
        self.assertFalse(selected['rationale']['C0']['eligible'])
        self.assertFalse(selected['rationale']['C1']['at_least_one_strict_improvement'])

    def test_final_method_is_actual_tour_when_fallback_frozen(self):
        p={'revised_candidates':[{'id':'C0','config':{}}]}
        with self.assertRaises(ValueError):method_definitions(p,'test')
        methods=method_definitions(p,'test',{'phase':'validation','selected':'TOUR'})
        self.assertEqual(len(methods),5)
        self.assertEqual(methods['revised'],{'method':'tour','config':None})


if __name__=='__main__':unittest.main()
