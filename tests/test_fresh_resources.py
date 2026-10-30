import importlib.util
from pathlib import Path
import unittest


class FreshResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[1]/'scripts/review_fresh_resources.py'
        spec=importlib.util.spec_from_file_location('fresh_resources_for_test',path)
        cls.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.module)

    def test_missing_allocation_timestamps_cannot_prove_concurrency(self):
        for missing in ('','None','Unknown',None):
            with self.subTest(missing=missing),self.assertRaises(ValueError):
                self.module.verified_interval({'Start':missing,'End':'2026-10-30T12:01:00','ElapsedRaw':'60'})

    def test_reversed_or_inconsistent_times_are_rejected(self):
        with self.assertRaises(ValueError):
            self.module.verified_interval({'Start':'2026-10-30T12:02:00','End':'2026-10-30T12:01:00','ElapsedRaw':'60'})
        with self.assertRaises(ValueError):
            self.module.verified_interval({'Start':'2026-10-30T12:00:00','End':'2026-10-30T12:01:00','ElapsedRaw':'600'})

    def test_actual_interval_matches_requested_charge_seconds(self):
        start,end=self.module.verified_interval({'Start':'2026-10-30T12:00:00','End':'2026-10-30T12:01:00','ElapsedRaw':'60'})
        self.assertEqual((end-start).total_seconds(),60)


if __name__=='__main__':unittest.main()
