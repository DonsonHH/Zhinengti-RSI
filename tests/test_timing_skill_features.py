import copy
import gzip
import json
from pathlib import Path
import unittest

from patrol_rsi.timing_skill_features import PublicTimingView


class TimingFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=Path(__file__).resolve().parents[1]/'research/2026-10-29-m12c1b32-formal/evidence/explanatory-cases/development-positive.json.gz'
        cls.bundle=json.loads(gzip.decompress(p.read_bytes()))['selected_public_snapshots'][0]

    def geometry(self,bundle,task):return {'task':task,'finish_s':bundle['time_s']+20,'target_lead_s':100.,'shift_s':20.}

    def test_observer_contact_default_is_unknown_without_a_qualified_measurement(self):
        b=copy.deepcopy(self.bundle);j=next(i for i,r in enumerate(b['tasks']) if r['kind']=='electrical');r=b['tasks'][j]
        r.update(contact_c=38.,qualified_available=False,already_alarm=False,last_check_s=-1.)
        for name in ('contact','qualified'):
            r[name].update(available=False,value=None,time_s=None,age_s=None,context_cutoff_s=None,reason='not_observed')
        f=PublicTimingView(b).features(self.geometry(b,j))
        self.assertFalse(f['professional_available']);self.assertIsNone(f['professional_signal'])
        self.assertFalse(f['direct_available']);self.assertIsNone(f['direct_reading_normal']);self.assertIsNone(f['direct_age_s'])

    def test_retained_remote_flow_is_not_relabelled_as_a_qualified_reference(self):
        b=copy.deepcopy(self.bundle);now=b['time_s'];j=next(i for i,r in enumerate(b['tasks']) if r['kind']=='cooling');r=b['tasks'][j]
        r.update(qualified_available=True,last_check_s=now-30,flow_kg_s=36.)
        r['qualified'].update(available=True,value=36.,time_s=now-30,age_s=30.,context_cutoff_s=now-30,reason=None)
        r['flow'].update(available=True,value=36.,time_s=now-5,age_s=5.,context_cutoff_s=now-5,reason=None,source='qualified_flow')
        f=PublicTimingView(b).features(self.geometry(b,j))
        self.assertTrue(f['direct_available']);self.assertFalse(f['direct_is_qualified']);self.assertEqual(f['direct_age_s'],5.)


if __name__=='__main__':unittest.main()
