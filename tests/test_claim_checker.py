"""
test_claim_checker.py
---------------------
The claim checker decides Supported / Unsupported / Contradicted for a
single claim. Both the Accuracy Judge and the Hallucination Agent depend
on it, so its rules are tested directly here.
"""
import unittest
from tests.support import OfflineTestCase
from agents.claim_checker import check_all_claims, check_claim
GROTTO_EVIDENCE = [
    "Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. "
    "It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared "
    "to Saint Bernadette Soubirous in 1858."
]
class TestClaimChecker(OfflineTestCase):
    def test_claim_confirmed_by_evidence_is_supported(self):
        result = check_claim(
            "The Grotto is a Marian place of prayer and reflection.", GROTTO_EVIDENCE
        )
        self.assertEqual(result.claim_status, "Supported")
        self.assertIsNotNone(result.evidence)
    def test_claim_absent_from_evidence_is_unsupported_not_contradicted(self):
        """The core Milestone 2 rule: unsupported does not mean false."""
        result = check_claim(
            "The Eiffel Tower receives exactly 7 million visitors every year.",
            ["The Eiffel Tower is located in Paris and was completed in 1889."],
        )
        self.assertEqual(result.claim_status, "Unsupported")
        self.assertIn("not", result.reasoning.lower())
    def test_claim_with_a_swapped_subject_is_contradicted(self):
        result = check_claim(
            "Berlin is the capital of France.",
            ["Paris is the capital and most populous city of France."],
        )
        self.assertEqual(result.claim_status, "Contradicted")
    def test_claim_with_a_different_number_is_contradicted(self):
        result = check_claim(
            "The Moon is about 1,000,000 kilometers from Earth.",
            ["The Moon is about 384,400 kilometers from Earth on average."],
        )
        self.assertEqual(result.claim_status, "Contradicted")
    def test_claim_refuted_by_the_evidence_is_contradicted(self):
        result = check_claim(
            "Eating carrots dramatically improves your night vision.",
            [
                "Carrots contain beta-carotene. However, the popular claim that eating carrots "
                "gives you dramatically improved night vision is exaggerated."
            ],
        )
        self.assertEqual(result.claim_status, "Contradicted")
    def test_topical_overlap_alone_does_not_prove_support(self):
        """Same subject, completely different content: must never be Supported."""
        result = check_claim(
            "The Grotto at Notre Dame is a football stadium used for sporting events.",
            GROTTO_EVIDENCE,
        )
        self.assertIn(result.claim_status, ("Unsupported", "Contradicted"))
        self.assertNotEqual(result.claim_status, "Supported")
    def test_no_evidence_means_unsupported_with_no_evidence_quoted(self):
        result = check_claim("Any factual claim at all.", [])
        self.assertEqual(result.claim_status, "Unsupported")
        self.assertIsNone(result.evidence)
    def test_evidence_quoted_is_always_taken_from_the_input(self):
        """The agent must never invent evidence."""
        result = check_claim("The Grotto is a Marian place of prayer.", GROTTO_EVIDENCE)
        self.assertIn(result.evidence, GROTTO_EVIDENCE[0])
    def test_check_all_claims_returns_one_result_per_claim(self):
        results = check_all_claims(
            ["Paris is the capital of France.", "Paris has exactly 12 million residents."],
            ["Paris is the capital and most populous city of France."],
        )
        self.assertEqual(len(results), 2)
if __name__ == "__main__":
    unittest.main()
class TestSwappedAnswerContradiction(unittest.TestCase):
    def test_london_vs_paris_is_contradicted_and_not_perfect(self):
        from agents.orchestrator import evaluate_response
        r = evaluate_response(
            "What is the capital of France?",
            "The capital of France is London.",
            "Paris is the capital of France.",
        )
        self.assertLess(r.accuracy.score, 5)
        self.assertEqual(r.hallucination.flagged_claims[0].claim_status, "Contradicted")
        self.assertNotEqual(r.verdict.verdict, "Pass")
    def test_matching_answer_stays_supported(self):
        from agents.orchestrator import evaluate_response
        r = evaluate_response(
            "What is the capital of France?",
            "The capital of France is Paris.",
            "Paris is the capital of France.",
        )
        self.assertEqual(r.accuracy.score, 5)
        self.assertEqual(r.hallucination.flagged_claims[0].claim_status, "Supported")
