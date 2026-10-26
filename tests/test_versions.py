import tempfile
import unittest
from pathlib import Path

from patrol_rsi.versions import AgentVersion, record_proposal


class VersionTests(unittest.TestCase):
    def test_protected_metric_or_model_cannot_be_edited(self):
        for patch in ({"evaluator": "always accept"}, {"model_weights": "different"},
                      {"truth": "read test labels"}, {"steps": 999}):
            with self.assertRaises(ValueError):
                AgentVersion().proposed(patch)

    def test_test_set_cannot_select_a_proposal(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                record_proposal(Path(folder), AgentVersion(), {"coordination_order": "cyclic"},
                                {"split": "test"}, "fixture", True, "human_fixture")

    def test_record_retains_origin_parent_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            args = (Path(folder), AgentVersion(), {"estimator_samples": 2},
                    {"split": "development", "score": 1.0}, "fixture, not LLM output", False, "human_fixture")
            path = record_proposal(*args)
            self.assertIn('"origin": "human_fixture"', path.read_text())
            with self.assertRaises(FileExistsError):
                record_proposal(*args)


if __name__ == "__main__":
    unittest.main()
