
import unittest
from unittest import mock
from tests.support import OfflineTestCase
from agents import accuracy_judge, hallucination_judge, orchestrator, relevance_judge
from agents.schemas import AccuracyInput, HallucinationInput, RelevanceInput
PARIS_EVIDENCE = ["Paris is the capital and most populous city of France."]
JUPITER_EVIDENCE = ["Jupiter is the largest planet in the Solar System. It is a gas giant."]
class TestRelevanceJudge(OfflineTestCase):
    def _judge(self, question, response):
        return relevance_judge.evaluate_relevance(RelevanceInput(question=question, response=response))
    def test_fully_relevant_response_scores_high(self):
        result = self._judge("What is the capital of France?", "Paris is the capital of France.")
        self.assertEqual(result.mode, "heuristic")
        self.assertGreaterEqual(result.score, 4)
        self.assertTrue(result.label)
        self.assertTrue(result.reasoning)
    def test_partially_relevant_response_scores_in_the_middle(self):
        result = self._judge(
            "What is the capital of France and roughly how large is its population?",
            "Paris is the capital of France.",
        )
        self.assertIn(result.score, (2, 3, 4))
    def test_off_topic_response_scores_low(self):
        result = self._judge(
            "What is the capital of France?",
            "Bananas are an excellent source of potassium and vitamin B6.",
        )
        self.assertLessEqual(result.score, 2)
    def test_relevance_ignores_factual_correctness(self):
        """A wrong but on-topic answer is still fully relevant."""
        correct = self._judge("What is the capital of France?", "Paris is the capital of France.")
        wrong = self._judge("What is the capital of France?", "Berlin is the capital of France.")
        self.assertGreaterEqual(wrong.score, 4)
        self.assertEqual(correct.score, wrong.score)
    def test_direct_yes_no_answer_is_treated_as_relevant(self):
        result = self._judge(
            "Do goldfish have a memory span of only 3 seconds?",
            "No, that's a myth. Goldfish can remember things for several months.",
        )
        self.assertGreaterEqual(result.score, 4)
    def test_paraphrases_receive_the_same_score(self):
        first = self._judge("What is the capital of France?", "Paris is the capital of France.")
        second = self._judge("What is the capital of France?", "France's capital city is Paris.")
        self.assertLessEqual(abs(first.score - second.score), 1)
    def test_score_is_always_within_the_scale(self):
        result = self._judge("A question?", "An answer.")
        self.assertIn(result.score, (1, 2, 3, 4, 5))
class TestAccuracyJudge(OfflineTestCase):
    def test_case_one_uses_the_reference_answer(self):
        result = accuracy_judge.evaluate_accuracy(
            AccuracyInput(
                question="What is the capital of France?",
                response="Paris is the capital of France.",
                reference_answer="Paris is the capital of France.",
            )
        )
        self.assertTrue(result.used_reference_answer)
        self.assertEqual(result.evidence_mode, "reference_answer")
        self.assertEqual(result.score, 5)
    def test_case_two_uses_rag_evidence_when_no_reference_is_given(self):
        result = accuracy_judge.evaluate_accuracy(
            AccuracyInput(
                question="What is the largest planet in our solar system?",
                response="Jupiter is the largest planet in the Solar System.",
                reference_answer=None,
                evidence=JUPITER_EVIDENCE,
            )
        )
        self.assertFalse(result.used_reference_answer)
        self.assertEqual(result.evidence_mode, "rag_evidence")
        self.assertGreaterEqual(result.score, 4)
    def test_correct_response_scores_five(self):
        result = accuracy_judge.evaluate_accuracy(
            AccuracyInput(
                question="What is the capital of France?",
                response="Paris is the capital of France.",
                evidence=PARIS_EVIDENCE,
            )
        )
        self.assertEqual(result.score, 5)
    def test_partially_correct_response_scores_in_the_middle(self):
        result = accuracy_judge.evaluate_accuracy(
            AccuracyInput(
                question="What is the largest planet in our solar system?",
                response="Jupiter is the largest planet in the Solar System. It has exactly 100 confirmed moons.",
                evidence=JUPITER_EVIDENCE,
            )
        )
        self.assertIn(result.score, (3, 4))
    def test_incorrect_response_scores_low(self):
        result = accuracy_judge.evaluate_accuracy(
            AccuracyInput(
                question="What is the Grotto at Notre Dame?",
                response="The Grotto at Notre Dame is a football stadium used for sporting events.",
                evidence=["The Grotto is a Marian place of prayer and reflection."],
            )
        )
        self.assertLessEqual(result.score, 2)
    def test_contradictory_response_scores_one(self):
        result = accuracy_judge.evaluate_accuracy(
            AccuracyInput(
                question="What is the capital of France?",
                response="Berlin is the capital of France.",
                reference_answer="Paris is the capital of France.",
            )
        )
        self.assertEqual(result.score, 1)
    def test_numeric_contradiction_scores_one(self):
        result = accuracy_judge.evaluate_accuracy(
            AccuracyInput(
                question="How far is the Moon from Earth?",
                response="The Moon is about 1,000,000 kilometers from Earth.",
                reference_answer="The Moon is about 384,400 kilometers from Earth on average.",
            )
        )
        self.assertEqual(result.score, 1)
    def test_supporting_evidence_is_never_invented(self):
        result = accuracy_judge.evaluate_accuracy(
            AccuracyInput(
                question="What is the capital of France?",
                response="Paris is the capital of France.",
                evidence=PARIS_EVIDENCE,
            )
        )
        for snippet in result.supporting_evidence:
            self.assertIn(snippet, PARIS_EVIDENCE[0])
    def test_no_evidence_gives_a_neutral_score_and_says_so(self):
        result = accuracy_judge.evaluate_accuracy(
            AccuracyInput(question="Some question?", response="Some response.", evidence=[])
        )
        self.assertEqual(result.score, 3)
        self.assertEqual(result.evidence_mode, "none")
        self.assertEqual(result.supporting_evidence, [])
class TestHallucinationAgent(OfflineTestCase):
    def test_fully_supported_response_has_no_hallucination(self):
        result = hallucination_judge.detect_hallucination(
            HallucinationInput(
                response="Jupiter is the largest planet in the Solar System.",
                evidence=JUPITER_EVIDENCE,
            )
        )
        self.assertEqual(result.hallucination_status, "No hallucination")
        self.assertTrue(all(c.claim_status == "Supported" for c in result.flagged_claims))
    def test_one_bad_claim_gives_partially_hallucinated_not_hallucinated(self):
        result = hallucination_judge.detect_hallucination(
            HallucinationInput(
                response="Jupiter is the largest planet in the Solar System. It has exactly 100 confirmed moons.",
                evidence=JUPITER_EVIDENCE,
            )
        )
        self.assertEqual(result.hallucination_status, "Partially hallucinated")
        self.assertEqual(len(result.flagged_claims), 2)
        statuses = {c.claim_status for c in result.flagged_claims}
        self.assertIn("Supported", statuses)
    def test_fully_unsupported_response_is_partially_not_fully_hallucinated(self):
        """
        Evidence that says nothing about a claim does not make the claim
        false. A response whose claims are all merely Unsupported must still
        be flagged, but "Hallucinated" is reserved for claims the evidence
        actually contradicts.
        """
        result = hallucination_judge.detect_hallucination(
            HallucinationInput(
                response="Bananas are an excellent source of potassium.",
                evidence=PARIS_EVIDENCE,
            )
        )
        self.assertTrue(all(c.claim_status == "Unsupported" for c in result.flagged_claims))
        self.assertEqual(result.hallucination_status, "Partially hallucinated")
        self.assertEqual(len(result.problem_claims), len(result.flagged_claims))
        self.assertGreaterEqual(len(result.problem_claims), 1)
    def test_contradicted_claim_is_reported_as_contradicted(self):
        result = hallucination_judge.detect_hallucination(
            HallucinationInput(response="Berlin is the capital of France.", evidence=PARIS_EVIDENCE)
        )
        self.assertEqual(result.flagged_claims[0].claim_status, "Contradicted")
        self.assertEqual(result.hallucination_status, "Hallucinated")
    def test_missing_evidence_gives_unsupported_never_contradicted(self):
        result = hallucination_judge.detect_hallucination(
            HallucinationInput(response="Bananas are a great source of potassium.", evidence=[])
        )
        self.assertEqual(result.flagged_claims[0].claim_status, "Unsupported")
    def test_every_claim_is_reported_with_evidence_and_reasoning(self):
        result = hallucination_judge.detect_hallucination(
            HallucinationInput(
                response="Jupiter is the largest planet in the Solar System. It has exactly 100 confirmed moons.",
                evidence=JUPITER_EVIDENCE,
            )
        )
        for claim in result.flagged_claims:
            self.assertTrue(claim.claim)
            self.assertTrue(claim.reasoning)
            self.assertIn(claim.claim_status, ("Supported", "Unsupported", "Contradicted"))
class TestOrchestrator(OfflineTestCase):
    def _patch_retrieval(self, evidence_text=PARIS_EVIDENCE[0]):
        from backend import retrieval
        patcher = mock.patch.object(
            retrieval,
            "retrieve_evidence",
            side_effect=lambda question, top_k=3: [
                {"text": evidence_text, "source": "demo", "question": question, "distance": 0.05}
            ],
        )
        patcher.start()
        self.addCleanup(patcher.stop)
    def test_orchestrator_calls_all_three_agents(self):
        self._patch_retrieval()
        result = orchestrator.evaluate_response(
            question="What is the capital of France?",
            ai_response="Paris is the capital of France.",
        )
        self.assertIsNotNone(result.relevance)
        self.assertIsNotNone(result.accuracy)
        self.assertIsNotNone(result.hallucination)
        self.assertEqual(len(result.retrieved_evidence), 1)
        self.assertEqual(result.hallucination.hallucination_status, "No hallucination")
    def test_orchestrator_passes_the_reference_answer_to_the_accuracy_judge(self):
        self._patch_retrieval()
        result = orchestrator.evaluate_response(
            question="How far is the Moon from Earth?",
            ai_response="The Moon is about 384,400 kilometers from Earth on average.",
            reference_answer="The Moon is about 384,400 kilometers from Earth on average.",
        )
        self.assertTrue(result.accuracy.used_reference_answer)
        self.assertEqual(result.accuracy.evidence_mode, "reference_answer")
    def test_orchestrator_survives_a_retrieval_failure(self):
        from backend import retrieval
        patcher = mock.patch.object(
            retrieval, "retrieve_evidence", side_effect=RuntimeError("vector store down")
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        result = orchestrator.evaluate_response(
            question="What is the capital of France?",
            ai_response="Paris is the capital of France.",
        )
        self.assertEqual(result.retrieved_evidence, [])
        self.assertEqual(result.accuracy.evidence_mode, "none")
    def test_orchestrator_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            orchestrator.evaluate_response(question="", ai_response="something")
        with self.assertRaises(ValueError):
            orchestrator.evaluate_response(question="something?", ai_response="   ")
    def test_summary_reports_all_three_judges(self):
        self._patch_retrieval()
        result = orchestrator.evaluate_response(
            question="What is the capital of France?",
            ai_response="Paris is the capital of France.",
        )
        summary = result.summary()
        self.assertIn("relevance", summary)
        self.assertIn("accuracy", summary)
        self.assertIn("hallucination", summary)
if __name__ == "__main__":
    unittest.main()
