"""
test_completeness_judge.py
---------------------------
Tests for the M3.1 Completeness Judge. Every test runs offline with
deterministic embeddings and no API key, so the heuristic evaluator is
exercised (same pattern as tests/test_agents.py for the Milestone 2
judges).
"""
import unittest
from tests.support import OfflineTestCase
from agents import completeness_judge
from agents.schemas import CompletenessInput
from agents.utils import extract_requirements
JUPITER_REFERENCE = "Jupiter is the largest planet in the Solar System. It has 95 known moons."
JUPITER_EVIDENCE = ["Jupiter is the largest planet in the Solar System. It has 95 known moons."]
class TestExtractRequirements(unittest.TestCase):
    def test_single_part_question_is_one_requirement(self):
        self.assertEqual(
            extract_requirements("What is the capital of France?"),
            ["What is the capital of France?"],
        )
    def test_two_part_question_joined_by_and_splits_in_two(self):
        reqs = extract_requirements("What is the capital of France, and what language is spoken there?")
        self.assertEqual(len(reqs), 2)
    def test_two_sentence_question_splits_in_two(self):
        reqs = extract_requirements("What is the largest planet? How many moons does it have?")
        self.assertEqual(len(reqs), 2)
    def test_short_conjunction_phrase_is_not_split(self):
        """'bread and butter' has too little content on each side to be two asks."""
        reqs = extract_requirements("Explain bread and butter.")
        self.assertEqual(len(reqs), 1)
    def test_empty_question_returns_empty_list(self):
        self.assertEqual(extract_requirements(""), [])
class TestCompletenessJudge(OfflineTestCase):
    def _judge(self, question, response, reference_answer=None, evidence=None):
        return completeness_judge.evaluate_completeness(
            CompletenessInput(
                question=question,
                response=response,
                reference_answer=reference_answer,
                evidence=evidence or [],
            )
        )
    def test_single_part_fully_addressed_scores_five(self):
        result = self._judge("What is the capital of France?", "Paris is the capital of France.")
        self.assertEqual(result.mode, "heuristic")
        self.assertEqual(result.score, 5)
        self.assertEqual(result.label, "Completely complete")
        self.assertEqual(result.missing_aspects, [])
        self.assertEqual(len(result.addressed_aspects), 1)
    def test_multi_part_fully_addressed_scores_five(self):
        result = self._judge(
            "What is the capital of France, and what language is spoken there?",
            "Paris is the capital of France, and French is the language spoken there.",
        )
        self.assertEqual(result.score, 5)
        self.assertEqual(result.missing_aspects, [])
        self.assertEqual(len(result.requirement_assessments), 2)
    def test_multi_part_half_addressed_is_partial(self):
        result = self._judge(
            "What is the capital of France, and what language is spoken there?",
            "Paris is the capital of France.",
        )
        self.assertIn(result.score, (2, 3, 4))
        self.assertEqual(len(result.addressed_aspects), 1)
        self.assertEqual(len(result.missing_aspects), 1)
    def test_completely_off_topic_response_is_missing(self):
        result = self._judge(
            "What is the capital of France?",
            "Bananas are an excellent source of potassium and vitamin B6.",
        )
        self.assertEqual(result.score, 1)
        self.assertEqual(result.label, "Incomplete")
        self.assertEqual(len(result.missing_aspects), 1)
        self.assertEqual(result.addressed_aspects, [])
    def test_multi_part_three_requirements_mixed_coverage(self):
        result = self._judge(
            "What is the capital of France, what language is spoken there, and what currency is used?",
            "Paris is the capital of France, and French is the language spoken there.",
        )
        self.assertEqual(len(result.requirement_assessments), 3)
        self.assertGreaterEqual(len(result.missing_aspects), 1)
        self.assertGreaterEqual(len(result.addressed_aspects), 1)
    def test_reference_answer_case_reports_reference_answer_evidence_mode(self):
        result = self._judge(
            "What is the largest planet, and how many moons does it have?",
            "Jupiter is the largest planet in the Solar System.",
            reference_answer=JUPITER_REFERENCE,
        )
        self.assertEqual(result.evidence_mode, "reference_answer")
        self.assertEqual(len(result.addressed_aspects), 1)
        self.assertEqual(len(result.missing_aspects), 1)
    def test_reference_answer_fully_covered_scores_five(self):
        result = self._judge(
            "What is the largest planet, and how many moons does it have?",
            "Jupiter is the largest planet in the Solar System, with 95 known moons.",
            reference_answer=JUPITER_REFERENCE,
        )
        self.assertEqual(result.score, 5)
        self.assertEqual(result.evidence_mode, "reference_answer")
    def test_rag_evidence_case_reports_rag_evidence_mode(self):
        result = self._judge(
            "What is the largest planet, and how many moons does it have?",
            "Jupiter is the largest planet in the Solar System.",
            evidence=JUPITER_EVIDENCE,
        )
        self.assertEqual(result.evidence_mode, "rag_evidence")
        self.assertEqual(len(result.missing_aspects), 1)
    def test_no_evidence_and_no_reference_falls_back_to_question_wording(self):
        result = self._judge("What is the capital of France?", "Paris is the capital of France.")
        self.assertEqual(result.evidence_mode, "none")
    def test_every_requirement_has_reasoning(self):
        result = self._judge(
            "What is the capital of France, and what language is spoken there?",
            "Paris is the capital of France.",
        )
        for assessment in result.requirement_assessments:
            self.assertTrue(assessment.requirement)
            self.assertTrue(assessment.reasoning)
            self.assertIn(assessment.status, ("Addressed", "Partially Addressed", "Missing"))
    def test_score_is_always_within_the_scale(self):
        result = self._judge("A question?", "An answer.")
        self.assertIn(result.score, (1, 2, 3, 4, 5))
if __name__ == "__main__":
    unittest.main()
