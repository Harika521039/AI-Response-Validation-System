
import unittest
from unittest import mock
from tests.support import OfflineTestCase
from agents import completeness_judge, hallucination_judge, verdict_agent
from agents.completeness_judge import CompletenessInput
from agents.hallucination_judge import FlaggedClaim, HallucinationInput, aggregate_status
from agents.utils import extract_requirements
from backend import retrieval
def _judge(question, response, reference_answer="", evidence=None):
    return completeness_judge.evaluate_completeness(
        CompletenessInput(
            question=question,
            response=response,
            reference_answer=reference_answer,
            evidence=evidence or [],
        )
    )
class TestCompletenessAccuracy(OfflineTestCase):
    def test_correct_paraphrase_is_addressed_not_missing(self):
        """
        "People there speak French." answers "what language is spoken there?"
        without reusing the word "language". This used to be scored Missing.
        """
        result = _judge(
            "What is the capital of France, and what language is spoken there?",
            "Paris is the capital of France. People there speak French.",
        )
        statuses = [a.status for a in result.requirement_assessments]
        self.assertEqual(statuses, ["Addressed", "Addressed"])
        self.assertEqual(result.score, 5)
        self.assertEqual(result.missing_aspects, [])
    def test_terse_but_correct_answer_is_addressed(self):
        """A bare "Paris." is complete: the question asked for one thing."""
        result = _judge("What is the capital of France?", "Paris.")
        self.assertEqual(result.requirement_assessments[0].status, "Addressed")
        self.assertEqual(result.score, 5)
    def test_question_echo_is_not_treated_as_a_complete_answer(self):
        """
        Repeating the question's wording adds no information, so it must not
        earn a full score - the accuracy fix must not become a loophole.
        """
        result = _judge(
            "What is photosynthesis, explain how it occurs and mention its products?",
            "Photosynthesis is photosynthesis and it occurs as it occurs.",
        )
        self.assertLessEqual(result.score, 3)
        self.assertNotIn("Addressed", [a.status for a in result.requirement_assessments])
    def test_partial_answer_still_reports_the_missing_part(self):
        result = _judge(
            "What is photosynthesis, explain how it occurs and mention its products?",
            "Photosynthesis is the process plants use to make food.",
            reference_answer=(
                "Photosynthesis is how green plants make food from light. "
                "It happens in the chloroplasts using chlorophyll. "
                "It produces glucose and releases oxygen."
            ),
        )
        self.assertGreaterEqual(len(result.missing_aspects), 1)
        self.assertLessEqual(result.score, 3)
    def test_unrelated_evidence_cannot_lower_a_good_answer(self):
        """
        Evidence with nothing to do with the question is ignored rather than
        used as the yardstick for what the answer should have said.
        """
        with_junk = _judge(
            "What is the capital of France?",
            "Paris is the capital of France.",
            evidence=["The axolotl is a salamander that lives in Mexican lakes."],
        )
        self.assertEqual(with_junk.score, 5)
        self.assertEqual(with_junk.missing_aspects, [])
def _claim(status):
    return FlaggedClaim(claim="c", claim_status=status, evidence=None, reasoning="r")
class TestUnsupportedIsNotHallucinated(OfflineTestCase):
    def test_all_unsupported_claims_are_only_partially_hallucinated(self):
        self.assertEqual(
            aggregate_status([_claim("Unsupported"), _claim("Unsupported")]),
            "Partially hallucinated",
        )
    def test_all_contradicted_claims_are_hallucinated(self):
        self.assertEqual(
            aggregate_status([_claim("Contradicted"), _claim("Contradicted")]),
            "Hallucinated",
        )
    def test_a_mix_of_supported_and_contradicted_is_partial(self):
        self.assertEqual(
            aggregate_status([_claim("Supported"), _claim("Contradicted")]),
            "Partially hallucinated",
        )
    def test_unsupported_claims_score_half_credit_not_zero(self):
        output = hallucination_judge.HallucinationOutput(
            hallucination_status="Partially hallucinated",
            flagged_claims=[_claim("Unsupported"), _claim("Unsupported")],
            reasoning="r",
            mode="heuristic",
        )
        self.assertEqual(verdict_agent._normalize_hallucination(output), 50.0)
    def test_unsupported_claims_do_not_force_a_fail_verdict(self):
        """
        A thin knowledge base must not condemn a correct answer. Only claims
        the evidence actually contradicts trigger the critical rule.
        """
        result = hallucination_judge.detect_hallucination(
            HallucinationInput(
                response="The axolotl is a salamander that keeps its gills for life.",
                evidence=["Paris is the capital and most populous city of France."],
            )
        )
        self.assertTrue(all(c.claim_status == "Unsupported" for c in result.flagged_claims))
        self.assertEqual(result.hallucination_status, "Partially hallucinated")
        verdict, issues = verdict_agent._apply_critical_rules(
            "Pass",
            mock.Mock(score=4),
            result,
        )
        self.assertEqual(verdict, "Pass")
        self.assertEqual(issues, [])
    def test_contradicted_claims_still_force_a_fail_verdict(self):
        contradicted = hallucination_judge.HallucinationOutput(
            hallucination_status="Hallucinated",
            flagged_claims=[_claim("Contradicted")],
            reasoning="r",
            mode="heuristic",
        )
        verdict, issues = verdict_agent._apply_critical_rules(
            "Pass", mock.Mock(score=4), contradicted
        )
        self.assertEqual(verdict, "Fail")
        self.assertTrue(issues)
class TestEvidenceRelevanceCutoff(OfflineTestCase):
    def _search(self, results):
        return mock.patch.object(
            retrieval.vector_store, "search", return_value=results
        ), mock.patch.object(retrieval.embeddings, "embed_single", return_value=[0.0, 1.0])
    def test_unrelated_chunks_are_dropped(self):
        results = [
            {
                "text": "The axolotl is a salamander found in Mexican lakes.",
                "source": "demo",
                "question": "What is an axolotl?",
                "distance": 0.93,
            }
        ]
        search_patch, embed_patch = self._search(results)
        with search_patch, embed_patch:
            evidence = retrieval.retrieve_evidence("What is the capital of France?")
        self.assertEqual(evidence, [], "an unrelated chunk was used as evidence")
    def test_relevant_chunks_are_kept(self):
        results = [
            {
                "text": "Paris is the capital and most populous city of France.",
                "source": "demo",
                "question": "What is the capital of France?",
                "distance": 0.08,
            }
        ]
        search_patch, embed_patch = self._search(results)
        with search_patch, embed_patch:
            evidence = retrieval.retrieve_evidence("What is the capital of France?")
        self.assertEqual(len(evidence), 1)
        self.assertIn("Paris", evidence[0]["text"])
    def test_only_the_relevant_chunk_survives_a_mixed_result_set(self):
        results = [
            {
                "text": "Paris is the capital of France.",
                "source": "demo",
                "question": "What is the capital of France?",
                "distance": 0.10,
            },
            {
                "text": "Photosynthesis converts light into chemical energy.",
                "source": "demo",
                "question": "What is photosynthesis?",
                "distance": 0.88,
            },
        ]
        search_patch, embed_patch = self._search(results)
        with search_patch, embed_patch:
            evidence = retrieval.retrieve_evidence("What is the capital of France?")
        self.assertEqual([item["text"] for item in evidence], ["Paris is the capital of France."])
    def test_an_empty_result_set_is_returned_unchanged(self):
        search_patch, embed_patch = self._search([])
        with search_patch, embed_patch:
            self.assertEqual(retrieval.retrieve_evidence("Anything at all?"), [])
class TestRequirementExtraction(OfflineTestCase):
    def test_compare_x_and_y_stays_one_requirement(self):
        self.assertEqual(
            extract_requirements("Compare photosynthesis and cellular respiration."),
            ["Compare photosynthesis and cellular respiration"],
        )
    def test_genuine_multi_part_questions_still_split(self):
        self.assertEqual(
            len(
                extract_requirements(
                    "What is the capital of France, and what language is spoken there?"
                )
            ),
            2,
        )
    def test_requirements_have_no_trailing_punctuation_artifacts(self):
        for requirement in extract_requirements(
            "What is photosynthesis, explain how it occurs and mention its products?"
        ):
            self.assertFalse(requirement.endswith(".."), requirement)
            self.assertEqual(requirement, requirement.strip())
if __name__ == "__main__":
    unittest.main()
