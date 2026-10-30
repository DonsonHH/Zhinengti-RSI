import contextlib
import gzip
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch


class FreshExportSealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script=Path(__file__).resolve().parents[1]/'scripts/package_fresh_evidence.py'
        spec=importlib.util.spec_from_file_location('fresh_export_for_test',cls.script)
        cls.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.module)

    def fixture(self,root):
        """Export-boundary fixture only; no simulated or research samples."""
        for name in ('report','final/gate','jobs','data/development/seed-20000/processes'):
            (root/name).mkdir(parents=True)
        names=('coverage.json','coverage-dimensions.json','independent-audits.json','raw-storage.json',
            'electrical-nonfinal-diagnostics.json','sealed-diagnostic-receipt.json',
            'development-clock-diagnostic.json','覆盖表.md','cross-split-conflicts.json','counts-independent.json','source-replay.json')
        for name in names:(root/'report'/name).write_text('{}',encoding='utf-8')
        coverage={'coverage_cells':[{'split':'final_test','kind':'thermal','horizon_s':30}],
            'sealed_full_coverage':{'sha256':'fixture-only-not-research'}}
        (root/'report/coverage.json').write_text(json.dumps(coverage),encoding='utf-8')
        (root/'report/coverage-dimensions.json').write_text(json.dumps({'split':{'final_test/thermal/30':{}}}),encoding='utf-8')
        (root/'report/condition-counts-nonfinal.json.gz').write_bytes(gzip.compress(json.dumps([{'split':'development'}]*800).encode()))
        (root/'jobs/final-v5-1_40.out').write_text('PRIVATE_FINAL_SEED_EVENTS=12',encoding='utf-8')
        (root/'jobs/final-v5-1_40.err').write_bytes(b'')
        (root/'jobs/package-final-log-seal-correction.txt').write_text('public cancellation explanation',encoding='utf-8')
        (root/'jobs/final-audit-version-correction.txt').write_text('public audit version explanation',encoding='utf-8')
        (root/'jobs/package-99.out').write_text('live packaging output',encoding='utf-8')
        (root/'final/gate/full-tests.json').write_text('{}',encoding='utf-8')
        for name in ('cluster-start.txt','environment-start.txt'):(root/name).write_text('fixture',encoding='utf-8')
        group=root/'data/development/seed-20000'
        (group/'development--20000--dc_contact_hotspot--180--335--0.2').mkdir()
        (group/'development--20000--dc_contact_hotspot--180--335--0.2/example.json').write_text('{}',encoding='utf-8')
        for name in ('geometry.npz','processes/20000--dc_contact_hotspot--180--335.npz','processes/20000--dc_contact_hotspot--180--335.json'):
            (group/name).write_text('fixture-not-a-data-array',encoding='utf-8')

    def run_package(self,root,out):
        with patch.object(sys,'argv',[str(self.script),'--run',str(root),'--output',str(out)]),contextlib.redirect_stdout(io.StringIO()):
            self.module.main()

    def test_final_seed_event_log_is_sealed_but_cancellation_reason_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'fixture';out=Path(tmp)/'export';self.fixture(root);self.run_package(root,out)
            receipt=json.loads((out/'delivery-manifest.json').read_bytes())
            self.assertEqual(len(receipt['sealed_final_log_receipts']),2)
            with tarfile.open(out/'delivery.tar.gz','r:gz') as tar:
                names=tar.getnames()
                self.assertIn('jobs/package-final-log-seal-correction.txt',names)
                self.assertIn('jobs/final-audit-version-correction.txt',names)
                self.assertFalse(any(name.startswith('jobs/final-') and Path(name).suffix in ('.out','.err') for name in names))
                self.assertNotIn('jobs/package-99.out',names)
                self.assertFalse(any(b'PRIVATE_FINAL_SEED_EVENTS' in tar.extractfile(v).read() for v in tar.getmembers() if v.isfile()))

    def test_unapproved_final_evaluator_output_cannot_enter_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'fixture';out=Path(tmp)/'export';self.fixture(root)
            (root/'report/final-labels.json').write_text('PRIVATE_FINAL_LABELS',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Exact released report files'):
                self.run_package(root,out)


if __name__=='__main__':unittest.main()
