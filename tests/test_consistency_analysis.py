"""
test_consistency_analysis.py
----------------------------
Tests for the grading logic in evaluation/run_consistency_check.py: the
expected-vs-actual comparison, the false positive / false negative
detection, and the paraphrase consistency rule.
These tests use hand-built records so the grading logic itself is checked
in isolation, without running the agents.
"""
import json
import unittest
from tests.support import OfflineTestCase
from evaluation.run_consistency_check import (
    CONSISTENCY_TOLERANCE,
    DATASET_PATH,
    SCORE_TOLERANCE,
    analyze_case,
    check_paraphrase_consistency,
    dataset_coverage,
)
def build_record(record_id, relevance, accuracy, status, expected, paraphrase_of=None):
    return {
        "id": record_id,
        "category": "test",
        "evidence_mode": "rag",
        "paraphrase_of": paraphrase_of,
        "expected_behavior": expected,
        "actual_output": {
            "relevance": {"score": relevance},
            "accuracy": {"score": accuracy},
            "hallucination": {"hallucination_status": status, "flagged_claims": []},
        },
    }
ALL_CORRECT = {
    "relevance_score": 5,
    "accuracy_score": 5,
    "hallucination_status": ["No hallucination"],
}
class TestExpectedVsActual(unittest.TestCase):
    def test_matching_result_passes(self):
        analysis = analyze_case(build_record("TC01", 5, 5, "No hallucination", ALL_CORRECT))
        self.assertTrue(analysis["passed"])
        self.assertFalse(analysis["relevance_mismatch"])
        self.assertFalse(analysis["hallucination_false_positive"])
    def test_one_point_of_difference_is_tolerated(self):
        analysis = analyze_case(build_record("TC01", 4, 5, "No hallucination", ALL_CORRECT))
        self.assertFalse(analysis["relevance_mismatch"])
        self.assertEqual(SCORE_TOLERANCE, 1)
    def test_a_large_score_gap_is_a_mismatch(self):
        analysis = analyze_case(build_record("TC01", 2, 5, "No hallucination", ALL_CORRECT))
        self.assertTrue(analysis["relevance_mismatch"])
        self.assertFalse(analysis["passed"])
    def test_false_positive_is_detected(self):
        """The system cried hallucination when the evidence supported the response."""
        analysis = analyze_case(build_record("TC01", 5, 5, "Hallucinated", ALL_CORRECT))
        self.assertTrue(analysis["hallucination_false_positive"])
        self.assertFalse(analysis["hallucination_false_negative"])
        self.assertFalse(analysis["passed"])
    def test_false_negative_is_detected(self):
        """The system stayed silent about a claim that really was unsupported."""
        expected = {
            "relevance_score": 5,
            "accuracy_score": 1,
            "hallucination_status": ["Hallucinated"],
        }
        analysis = analyze_case(build_record("TC03", 5, 1, "No hallucination", expected))
        self.assertTrue(analysis["hallucination_false_negative"])
        self.assertFalse(analysis["hallucination_false_positive"])
    def test_a_null_expectation_is_skipped_not_guessed(self):
        expected = {
            "relevance_score": 1,
            "accuracy_score": None,
            "hallucination_status": ["Hallucinated"],
        }
        analysis = analyze_case(build_record("TC05", 1, 2, "Hallucinated", expected))
        self.assertIsNone(analysis["expected_accuracy_score"])
        self.assertFalse(analysis["accuracy_mismatch"])
        self.assertTrue(analysis["passed"])
    def test_several_acceptable_statuses_are_all_allowed(self):
        expected = {
            "relevance_score": 3,
            "accuracy_score": 2,
            "hallucination_status": ["Hallucinated", "Partially hallucinated"],
        }
        analysis = analyze_case(build_record("TC09", 3, 2, "Partially hallucinated", expected))
        self.assertFalse(analysis["hallucination_mismatch"])
    def test_a_forbidden_claim_status_fails_the_case(self):
        """TC09 must never report Contradicted, because the evidence is merely silent."""
        expected = {
            "relevance_score": 3,
            "accuracy_score": 2,
            "hallucination_status": ["Hallucinated"],
            "expected_claim_statuses_exclude": ["Contradicted"],
        }
        record = build_record("TC09", 3, 2, "Hallucinated", expected)
        record["actual_output"]["hallucination"]["flagged_claims"] = [
            {"claim_status": "Contradicted"}
        ]
        analysis = analyze_case(record)
        self.assertEqual(analysis["forbidden_claim_statuses_found"], ["Contradicted"])
        self.assertFalse(analysis["passed"])
class TestParaphraseConsistency(unittest.TestCase):
    def test_identical_scores_are_consistent(self):
        records = {
            "TC01": build_record("TC01", 5, 5, "No hallucination", ALL_CORRECT),
            "TC02": build_record("TC02", 5, 5, "No hallucination", ALL_CORRECT, paraphrase_of="TC01"),
        }
        checks = check_paraphrase_consistency(records)
        self.assertEqual(len(checks), 1)
        self.assertTrue(checks[0]["consistent"])
        self.assertEqual(checks[0]["relevance_gap"], 0)
    def test_a_gap_larger_than_the_tolerance_is_inconsistent(self):
        records = {
            "TC01": build_record("TC01", 5, 5, "No hallucination", ALL_CORRECT),
            "TC02": build_record("TC02", 5, 2, "No hallucination", ALL_CORRECT, paraphrase_of="TC01"),
        }
        checks = check_paraphrase_consistency(records)
        self.assertFalse(checks[0]["consistent"])
        self.assertEqual(checks[0]["accuracy_gap"], 3)
        self.assertEqual(CONSISTENCY_TOLERANCE, 1)
    def test_a_missing_partner_is_reported_rather_than_ignored(self):
        records = {
            "TC02": build_record("TC02", 5, 5, "No hallucination", ALL_CORRECT, paraphrase_of="TC01"),
        }
        checks = check_paraphrase_consistency(records)
        self.assertFalse(checks[0]["checked"])
class TestBenchmarkDataset(unittest.TestCase):
    """The dataset itself must cover everything Milestone 2 asks for."""
    @classmethod
    def setUpClass(cls):
        with open(DATASET_PATH, "r", encoding="utf-8") as handle:
            cls.dataset = json.load(handle)
    def test_dataset_size_is_in_the_required_range(self):
        self.assertGreaterEqual(len(self.dataset), 10)
        self.assertLessEqual(len(self.dataset), 15)
    def test_every_case_has_the_required_fields(self):
        for case in self.dataset:
            for key in ("id", "category", "evidence_mode", "question", "ai_response", "expected_behavior"):
                self.assertIn(key, case, f"{case.get('id')} is missing {key}")
    def test_case_ids_are_unique(self):
        ids = [case["id"] for case in self.dataset]
        self.assertEqual(len(ids), len(set(ids)))
    def test_both_accuracy_evidence_modes_are_covered(self):
        coverage = dataset_coverage(self.dataset)
        self.assertGreater(coverage["reference_answer_cases"], 0)
        self.assertGreater(coverage["rag_evidence_cases"], 0)
    def test_reference_answer_cases_actually_supply_a_reference(self):
        for case in self.dataset:
            if case["evidence_mode"] == "reference_answer":
                self.assertTrue(case.get("reference_answer"), f"{case['id']} has no reference answer")
    def test_rag_cases_supply_no_reference_answer(self):
        for case in self.dataset:
            if case["evidence_mode"] == "rag":
                self.assertIsNone(case.get("reference_answer"), f"{case['id']} should not have a reference answer")
    def test_all_required_response_types_are_present(self):
        categories = " ".join(case["category"] for case in self.dataset)
        for required in (
            "correct",
            "incorrect",
            "partially_correct",
            "irrelevant",
            "incomplete",
            "unsupported",
            "contradicted",
            "paraphrase",
        ):
            self.assertIn(required, categories, f"no test case covers '{required}'")
    def test_paraphrase_pairs_exist_in_both_evidence_modes(self):
        modes = {case["evidence_mode"] for case in self.dataset if case.get("paraphrase_of")}
        self.assertIn("rag", modes)
        self.assertIn("reference_answer", modes)
if __name__ == "__main__":
    unittest.main()
