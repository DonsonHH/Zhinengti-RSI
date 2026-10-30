from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from patrol_rsi.limited_prediction import COL,HeadData,expanded_features,scalar_score,view_indices,sha,write
from patrol_rsi.limited_metrics import evaluate,fpr_threshold,probabilities,weighted_ap


class LimitedPredictionTests(unittest.TestCase):
    def public(self):
        x=np.zeros((4,98),np.float32);x[:,COL['root_time_s']]=[30,60,90,120];x[:,COL['pod']]=[0,1,2,3]
        return x

    def test_time_view_cannot_observe_state_or_CVAE_channels(self):
        x=self.public();first=expanded_features(x,'time_lr','thermal')
        x[:,COL['temperature_measurement.value']]=100;x[:,COL['cvae_raw.value']]=999
        x[:,COL['already_measured_alarm']]=1;x[:,COL['public_primary_supply_c']]=300
        self.assertTrue(np.array_equal(first,expanded_features(x,'time_lr','thermal')))
        self.assertFalse(np.array_equal(expanded_features(x,'public_lr','thermal'),expanded_features(self.public(),'public_lr','thermal')))

    def test_CVAE_views_are_exactly_separate_and_thermal_only(self):
        base=view_indices('public_lr','thermal');raw=view_indices('public_raw_lr','thermal')
        cache=view_indices('public_assimilated_lr','thermal');both=view_indices('public_all_lr','thermal')
        self.assertEqual((len(raw)-len(base),len(cache)-len(base),len(both)-len(base)),(6,6,12))
        self.assertNotIn(COL['history_assimilated.value'],raw)
        self.assertNotIn(COL['cvae_raw.value'],cache)
        with self.assertRaises(ValueError):view_indices('public_all_lr','electrical')
        with self.assertRaises(ValueError):scalar_score(self.public(),'frozen_CVAE_raw','control')

    def test_missing_measurement_is_not_a_real_normal_observation(self):
        x=self.public();x[:,COL['contact.value']]=90
        self.assertTrue((scalar_score(x,'professional_hold','electrical')==0).all())
        x[:,COL['contact.available']]=1
        self.assertTrue((scalar_score(x,'professional_hold','electrical')==1).all())

    def test_tied_constant_AUPRC_equals_weighted_prevalence(self):
        y=np.array([1,0,0,1]);w=np.array([.1,.2,.3,.4]);p=np.full(4,.2)
        self.assertAlmostEqual(weighted_ap(y,p,w),.5)
        result=probabilities(y,p,w)
        self.assertAlmostEqual(result['Brier'],.34)
        self.assertAlmostEqual(sum(v['weight'] for v in result['reliability_bins']),1.)

    def test_1percent_threshold_conservatively_rejects_ties(self):
        y=np.zeros(100,int);p=np.r_[np.ones(3),np.zeros(97)]
        threshold=fpr_threshold(y,p)
        self.assertEqual(threshold,1.)
        self.assertEqual(int((p>threshold).sum()),0)

    def test_invalid_B_minus_one_cannot_enter_probability_metrics(self):
        with self.assertRaises(ValueError):probabilities(np.array([-1,0,1]),np.array([.1,.1,.8]),np.ones(3))

    def cohort(self):
        y=np.array([0,0,0,0,1,1,1,1],np.int8)
        m=np.zeros((8,8),np.int32);m[:,2]=np.repeat(range(4),2);m[:,3]=20000;m[:,4]=[0,0,0,0,1,1,1,1]
        m[:,5]=[0,0,0,0,11,11,11,11];m[:,1]=1;m[:,7]=[60,90]*4
        return SimpleNamespace(n=8,y=np.column_stack((y,y)),w=np.full(8,1/8),meta=m)

    def test_noise_event_and_zero_event_denominators(self):
        data=self.cohort();p=np.array([.1,.1,.1,.1,.9,.1,.9,.1])
        result=evaluate(data,1,p,.5)
        self.assertEqual(result['summary']['unique_eligible_events'],1)
        self.assertEqual(result['summary']['seed_macro_event_recall'],1.)
        self.assertEqual(result['seeds'][0]['event_eligible_processes'],1)
        self.assertEqual(result['seeds'][0]['numeric_processes'],2)
        self.assertEqual(result['summary']['samples'],8)

    def test_noise_eligibility_mismatch_stops_instead_of_dropping_noise(self):
        data=self.cohort();data.y[6:,1]=0
        with self.assertRaisesRegex(ValueError,'Paired noise eligibility'):
            evaluate(data,1,np.full(8,.1),.5)

    def test_A_head_uses_fixed_B_confirmation_event_denominator(self):
        ya=np.array([1,1,0,1,1,0],np.int8);yb=np.array([1,0,0,1,0,0],np.int8)
        m=np.zeros((6,8),np.int32);m[:,2]=[0,0,0,1,1,1];m[:,3]=20000;m[:,4]=1
        m[:,5]=[1,2,0,1,2,0];m[:,1]=[0,1,2,0,1,2];m[:,7]=[60,90,120]*2
        data=SimpleNamespace(n=6,y=np.column_stack((ya,yb)),w=np.full(6,1/6),meta=m)
        p=np.array([.9,.1,.1,.9,.1,.1]);a=evaluate(data,0,p,.5);b=evaluate(data,1,p,.5)
        self.assertEqual(a['summary']['unique_eligible_events'],1)
        self.assertEqual(a['summary']['seed_macro_event_recall'],b['summary']['seed_macro_event_recall'])
        self.assertEqual((a['summary']['sample_recall'],b['summary']['sample_recall']),(.5,1.))

    def cache(self,folder):
        np.save(folder/'features.npy',self.public()[:2]);np.save(folder/'weights.npy',np.full(2,.5))
        np.save(folder/'labels.npy',np.array([[0,0],[1,1]],np.int8));np.save(folder/'evaluator-meta.npy',np.zeros((2,8),np.int32))
        write(folder/'complete.json',{'phase':'validation_selection','cell_index':0,'kind':'thermal','horizon_s':30,
            'protocol_SHA256':'test-protocol','rows':2,'capacity':2,
            'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in folder.iterdir() if p.is_file()}})

    def test_cache_must_match_expected_phase_protocol_and_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.cache(folder)
            HeadData(folder,expected_phase='validation_selection',expected_cell=0,expected_protocol='test-protocol')
            for phase,cell,protocol in (('final_test',0,'test-protocol'),('validation_selection',6,'test-protocol'),('validation_selection',0,'wrong')):
                with self.subTest(phase=phase,cell=cell,protocol=protocol),self.assertRaises(ValueError):
                    HeadData(folder,expected_phase=phase,expected_cell=cell,expected_protocol=protocol)

    def test_changed_cache_array_cannot_bypass_registered_SHA(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.cache(folder);np.save(folder/'weights.npy',np.array([.25,.75]))
            with self.assertRaisesRegex(ValueError,'Cache array byte/SHA'):
                HeadData(folder,expected_phase='validation_selection',expected_cell=0,expected_protocol='test-protocol')


if __name__=='__main__':unittest.main()
