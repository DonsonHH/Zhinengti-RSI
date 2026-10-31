import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from patrol_rsi.aligned_patrol import arm_programs
from patrol_rsi.limited_prediction import sha
from patrol_rsi.paired_campaign import protocols,manifest,aggregate,validation_gate,require_final_seal,PAIRED_SHA,RSI_SHA,exclude_history,verify_identity

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'research/2026-10-30-m12c1b4b2-alignment'


def scores():
    return {arm:{'macro_missed_count':1.,'macro_wait_s':100.,'safety_ok':True,'all_replays_and_independent_audits_passed':True,
        'seeds':[{'seed':31020,'macro_missed_count':1.},{'seed':31021,'macro_missed_count':1.}]} for arm in arm_programs()}


class PairedCampaignTests(unittest.TestCase):
    def test_incomplete_source_identity_cannot_validate_empty_artifacts(self):
        with self.assertRaises((ValueError,KeyError)):verify_identity({'artifacts':[]})
        with self.assertRaises(ValueError):verify_identity({'bound_inputs':{},'paired_SHA256':PAIRED_SHA,'RSI_SHA256':RSI_SHA,'artifacts':[]})
    def test_old_seed_or_physical_fingerprint_cannot_enter_new_phase(self):
        registry={'seed_groups':[20000],'numeric_fingerprints':['old-fingerprint']}
        for seed,fp in ((20000,'new'),(31000,'old-fingerprint')):
            with self.assertRaises(ValueError):exclude_history(seed,fp,registry)
        exclude_history(31000,'new',registry)
    def test_exact_protocol_hash_and_five_arm_identity(self):
        p,r=protocols(STAGE/'MultiAgent-Patrol-Paired-v1.json',STAGE/'RSI-Controlled-Meta-v1.json')
        self.assertEqual(sha(STAGE/'MultiAgent-Patrol-Paired-v1.json'),PAIRED_SHA)
        self.assertEqual(sha(STAGE/'RSI-Controlled-Meta-v1.json'),RSI_SHA)
        self.assertEqual(len(p['arms']),5)

    def test_manifest_is_72_processes_144_noise_conditions_no_split_overlap(self):
        p,_=protocols(STAGE/'MultiAgent-Patrol-Paired-v1.json',STAGE/'RSI-Controlled-Meta-v1.json')
        m=manifest(p);self.assertEqual(len(m['rows']),144);self.assertEqual(len({v['physical_key'] for v in m['rows']}),72)
        for phase in ('development','validation','final_test'):
            for row in m['rows']:
                if row['phase']==phase:self.assertIn(row['seed'],p['partition'][phase])

    def test_equal_scores_fail_no_Tour_replacement(self):
        value=validation_gate(scores(),sources_ok=True,budget_ok=True)
        self.assertFalse(value['passed']);self.assertEqual(value['status'],'NO-GO_stop_final_sealed')

    def test_wait_5percent_requires_every_seed_miss_nonincrease(self):
        s=scores();s['revised_PPO_evidence']['macro_wait_s']=95.
        self.assertTrue(validation_gate(s,sources_ok=True,budget_ok=True)['passed'])
        s['revised_PPO_evidence']['seeds'][0]['macro_missed_count']=2.
        self.assertFalse(validation_gate(s,sources_ok=True,budget_ok=True)['passed'])

    def test_any_safety_source_or_resource_failure_keeps_final_closed(self):
        s=scores();s['revised_PPO_evidence']['macro_missed_count']=.5
        for source,budget in ((False,True),(True,False)):
            self.assertFalse(validation_gate(s,sources_ok=source,budget_ok=budget)['passed'])
        s['Tour']['safety_ok']=False
        self.assertFalse(validation_gate(s,sources_ok=True,budget_ok=True)['passed'])

    def test_final_seal_cannot_be_missing_or_failed(self):
        with TemporaryDirectory() as directory:
            p=Path(directory);(p/'manifest.json').write_text('{}')
            with self.assertRaises(FileNotFoundError):require_final_seal(p/'none',p/'manifest.json')
            (p/'seal.json').write_text(json.dumps({'gate':{'passed':False},'protocol_SHA256':PAIRED_SHA,'manifest_SHA256':sha(p/'manifest.json')}))
            with self.assertRaises(ValueError):require_final_seal(p/'seal.json',p/'manifest.json')

    def test_group_aggregation_requires_paired_noise_not_random_rows(self):
        with self.assertRaises(ValueError):aggregate([])
        rows=[{'arm':arm,'seed':31020,'physical_key':'only_one','noise_c':.2} for arm in arm_programs()]
        with self.assertRaises(ValueError):aggregate(rows)


if __name__=='__main__':unittest.main()
