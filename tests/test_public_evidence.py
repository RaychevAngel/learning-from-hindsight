"""The release's dependency-free checks are also discoverable by unittest."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "public_evidence", ROOT / "scripts/reproduce_evidence.py"
)
EVIDENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVIDENCE)


class PublicEvidenceTest(unittest.TestCase):
    def test_reanalysis_regrades_all_completed_records_and_keeps_limits(self):
        report = EVIDENCE.analyze(bootstrap_samples=100)
        self.assertEqual(report["regraded_records"], 2848)
        self.assertEqual(report["new_model_calls"], 0)
        self.assertEqual(report["independently_checked_synthetic_pairs"], 8720)
        main = report["experiments"]["main_teacher"]["counts"]
        self.assertEqual(main["baseline/Q"]["correct"], 27)
        self.assertEqual(main["adapted_bidirectional/Q"]["correct"], 30)
        earlier = report["experiments"]["earlier_transfer"]["counts"]
        self.assertEqual(earlier["baseline/Q"]["correct"], 23)
        self.assertEqual(earlier["adapted_pi/Q"]["correct"], 36)
        self.assertEqual(report["proposer_training"]["optimizer_updates"], 14)
        self.assertGreater(len(report["limits"]), 3)

    def test_duplicate_or_changed_grade_is_rejected(self):
        _, index = EVIDENCE.task_index(ROOT / "data/balanced-transfer-v1")
        row = EVIDENCE.rows(ROOT / "results/hindsight-teacher-qwen35-023/observations.jsonl")[0]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            EVIDENCE.regrade([row, row], index)
        changed = {**row, "success": not row["success"]}
        with self.assertRaisesRegex(ValueError, "grade differs"):
            EVIDENCE.regrade([changed], index)

    def test_incomplete_expected_matrix_is_rejected(self):
        _, index = EVIDENCE.task_index(ROOT / "data/balanced-transfer-v1")
        observations = EVIDENCE.rows(
            ROOT / "results/hindsight-teacher-qwen35-023/observations.jsonl"
        )
        expected = EVIDENCE.rows(ROOT / "results/hindsight-teacher-qwen35-023/expected.jsonl")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            EVIDENCE.regrade(observations[:-1], index, expected)

    def test_source_equal_and_position_pooled_means_remain_distinct(self):
        data = EVIDENCE.read(ROOT / "paper/derived/source-fidelity025-independent.json")
        report = EVIDENCE.fidelity_from_positions(data)
        self.assertAlmostEqual(report["source_equal_means"]["initial_kl_mean"], 0.2466060666)
        self.assertNotEqual(
            report["source_equal_means"]["initial_kl_mean"],
            report["pooled_means"]["initial_kl_mean"],
        )
        self.assertAlmostEqual(report["initial_KL_fraction_in_rule_text"], 0.9853573178520135)

    def test_changed_native_metric_or_duplicate_position_is_rejected(self):
        data = EVIDENCE.read(ROOT / "paper/derived/source-fidelity025-independent.json")
        changed = copy.deepcopy(data)
        changed["positions"][0]["initial_kl"] += 1
        with self.assertRaisesRegex(ValueError, "metric mismatch"):
            EVIDENCE.fidelity_from_positions(changed)
        data["positions"][1] = data["positions"][0]
        with self.assertRaisesRegex(ValueError, "Duplicate native"):
            EVIDENCE.fidelity_from_positions(data)

    def test_manifest_rejects_changed_inputs(self):
        import json

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "provenance").mkdir()
            (root / "input.json").write_text("{}\n")
            manifest = {
                "schema": "hindsight-public-evidence-v1",
                "files": [
                    {"path": "input.json", "bytes": 3, "sha256": EVIDENCE.sha(root / "input.json")}
                ],
            }
            (root / "provenance/evidence-manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(EVIDENCE.verify_manifest(root), 1)
            (root / "input.json").write_text("[]\n")
            with self.assertRaisesRegex(ValueError, "Changed evidence"):
                EVIDENCE.verify_manifest(root)


if __name__ == "__main__":
    unittest.main()
