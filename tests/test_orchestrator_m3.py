"""
test_orchestrator_m3.py
------------------------
Confirms the Evaluation Orchestrator runs all FOUR judges (Relevance,
Accuracy, Hallucination, Completeness) plus the Verdict Agent, and that
the Milestone 2 orchestrator behaviour (evidence retrieval, reference
answer handling, empty-input validation) still works unchanged.
"""
import unittest
from unittest import mock
from tests.support import OfflineTestCase
from agents import orchestrator
PARIS_EVIDENCE = "Paris is the capital and most populous city of France."
class TestOrchestratorWithFourJudges(OfflineTestCase):
    def _patch_retrieval(self, evidence_text=PARIS_EVIDENCE):
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
    def test_orchestrator_calls_all_four_judges(self):
        self._patch_retrieval()
        result = orchestrator.evaluate_response(
            question="What is the capital of France?",
            ai_response="Paris is the capital of France.",
        )
        self.assertIsNotNone(result.relevance)
        self.assertIsNotNone(result.accuracy)
        self.assertIsNotNone(result.hallucination)
        self.assertIsNotNone(result.completeness)
        self.assertIsNotNone(result.verdict)
    def test_completeness_judge_receives_the_same_reference_answer_as_accuracy(self):
        self._patch_retrieval()
        result = orchestrator.evaluate_response(
            question="What is the capital of France?",
            ai_response="Paris is the capital of France.",
            reference_answer="Paris is the capital of France.",
        )
        self.assertEqual(result.completeness.evidence_mode, "reference_answer")
        self.assertEqual(result.accuracy.evidence_mode, "reference_answer")
    def test_completeness_judge_falls_back_to_rag_evidence(self):
        self._patch_retrieval()
        result = orchestrator.evaluate_response(
            question="What is the capital of France?",
            ai_response="Paris is the capital of France.",
        )
        self.assertEqual(result.completeness.evidence_mode, "rag_evidence")
    def test_verdict_agent_combines_all_four_judges(self):
        self._patch_retrieval()
        result = orchestrator.evaluate_response(
            question="What is the capital of France?",
            ai_response="Paris is the capital of France.",
        )
        self.assertEqual(result.verdict.relevance_score, result.relevance.score)
        self.assertEqual(result.verdict.accuracy_score, result.accuracy.score)
        self.assertEqual(result.verdict.completeness_score, result.completeness.score)
        self.assertEqual(result.verdict.hallucination_status, result.hallucination.hallucination_status)
        self.assertIn(result.verdict.verdict, ("Pass", "Needs Improvement", "Fail"))
    def test_good_response_passes_end_to_end(self):
        self._patch_retrieval()
        result = orchestrator.evaluate_response(
            question="What is the capital of France?",
            ai_response="Paris is the capital of France.",
            reference_answer="Paris is the capital of France.",
        )
        self.assertEqual(result.verdict.verdict, "Pass")
    def test_contradictory_response_fails_end_to_end(self):
        self._patch_retrieval()
        result = orchestrator.evaluate_response(
            question="What is the capital of France?",
            ai_response="Berlin is the capital of France.",
            reference_answer="Paris is the capital of France.",
        )
        self.assertEqual(result.verdict.verdict, "Fail")
    def test_multi_part_question_flags_incomplete_coverage(self):
        self._patch_retrieval()
        result = orchestrator.evaluate_response(
            question="What is the capital of France, and what language is spoken there?",
            ai_response="Paris is the capital of France.",
        )
        self.assertEqual(len(result.completeness.requirement_assessments), 2)
        self.assertLess(len(result.completeness.addressed_aspects), 2)
        self.assertLess(result.completeness.score, 5)
    def test_summary_includes_completeness_and_verdict(self):
        self._patch_retrieval()
        result = orchestrator.evaluate_response(
            question="What is the capital of France?",
            ai_response="Paris is the capital of France.",
        )
        summary = result.summary()
        self.assertIn("completeness", summary)
        self.assertIn("verdict", summary)
        self.assertIn("weighted_overall_score", summary)
    def test_orchestrator_still_survives_a_retrieval_failure(self):
        """Milestone 2 behaviour must be unchanged: no crash on retrieval failure."""
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
        self.assertEqual(result.completeness.evidence_mode, "none")
        self.assertIsNotNone(result.verdict)
    def test_orchestrator_still_rejects_empty_input(self):
        """Milestone 2 behaviour must be unchanged."""
        with self.assertRaises(ValueError):
            orchestrator.evaluate_response(question="", ai_response="something")
        with self.assertRaises(ValueError):
            orchestrator.evaluate_response(question="something?", ai_response="   ")
if __name__ == "__main__":
    unittest.main()
