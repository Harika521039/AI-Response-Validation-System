"""
test_regression_grotto.py
-------------------------
Regression tests for two real bugs found while running the Milestone 2
benchmark. Both involved the same evidence, so they are kept together.
Bug 1 - FALSE NEGATIVE
    Response: "The Grotto at Notre Dame is a football stadium used for
               sporting events."
    The claim shares its subject ("the Grotto") with the evidence, which
    pushed embedding similarity high enough that the claim was marked
    Supported. Fixed by requiring the claim's own descriptive words to
    appear in the evidence before calling anything Supported.
Bug 2 - FALSE POSITIVE
    Response: "The Grotto at Notre Dame is a Marian place of prayer and
               reflection."
    This is correct and fully backed by the evidence, but the subject-swap
    rule fired on the evidence sentence "Immediately behind the basilica is
    the Grotto ...", treating the adverbial opening as a competing subject.
    Fixed by skipping the subject-swap rule whenever the claim's subject is
    mentioned in the evidence.
Both directions are locked in here, so a future change cannot fix one by
breaking the other.
"""
import unittest
from unittest import mock
from tests.support import OfflineTestCase
from agents import accuracy_judge, hallucination_judge, orchestrator
from agents.schemas import AccuracyInput, HallucinationInput
QUESTION = "What is the Grotto at Notre Dame?"
GROTTO_EVIDENCE = [
    "Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. "
    "It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared "
    "to Saint Bernadette Soubirous in 1858."
]
class TestGrottoRegressions(OfflineTestCase):
    def _hallucination(self, response, evidence=None):
        return hallucination_judge.detect_hallucination(
            HallucinationInput(
                question=QUESTION, response=response, evidence=evidence or GROTTO_EVIDENCE
            )
        )
    def _accuracy(self, response, evidence=None):
        return accuracy_judge.evaluate_accuracy(
            AccuracyInput(question=QUESTION, response=response, evidence=evidence or GROTTO_EVIDENCE)
        )
    def test_correct_response_is_fully_supported(self):
        response = (
            "The Grotto at Notre Dame is a Marian place of prayer and reflection. "
            "It is a replica of the grotto at Lourdes, France."
        )
        result = self._hallucination(response)
        self.assertEqual(result.hallucination_status, "No hallucination")
        self.assertTrue(all(c.claim_status == "Supported" for c in result.flagged_claims))
        self.assertGreaterEqual(self._accuracy(response).score, 4)
    def test_football_stadium_response_is_never_supported(self):
        response = "The Grotto at Notre Dame is a football stadium used for sporting events."
        result = self._hallucination(response)
        self.assertNotEqual(result.hallucination_status, "No hallucination")
        self.assertNotEqual(result.flagged_claims[0].claim_status, "Supported")
    def test_football_stadium_response_scores_low_on_accuracy(self):
        response = "The Grotto at Notre Dame is a football stadium used for sporting events."
        self.assertLessEqual(self._accuracy(response).score, 2)
    def test_football_stadium_response_through_the_orchestrator(self):
        from backend import retrieval
        patcher = mock.patch.object(
            retrieval,
            "retrieve_evidence",
            side_effect=lambda question, top_k=3: [
                {"text": GROTTO_EVIDENCE[0], "source": "demo", "question": QUESTION, "distance": 0.1}
            ],
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        result = orchestrator.evaluate_response(
            question=QUESTION,
            ai_response="The Grotto at Notre Dame is a football stadium used for sporting events.",
        )
        self.assertNotEqual(result.hallucination.hallucination_status, "No hallucination")
        self.assertLessEqual(result.accuracy.score, 2)
    def test_an_invented_claim_is_still_caught(self):
        """
        The invented claim must still be flagged. The evidence never mentions
        Mars or NASA, so the honest verdict is Unsupported rather than
        Contradicted - but it is reported as a problem claim either way, and
        the response is never waved through as clean.
        """
        result = self._hallucination(
            "The Grotto at Notre Dame is located on Mars and was built by NASA in 2050."
        )
        self.assertNotEqual(result.hallucination_status, "No hallucination")
        self.assertGreaterEqual(len(result.problem_claims), 1)
        self.assertTrue(
            all(c.claim_status != "Supported" for c in result.flagged_claims),
            "No part of the invented claim may be reported as supported.",
        )
    def test_no_false_positive_on_an_unrelated_correct_claim(self):
        response = "Berlin is the capital of Germany."
        evidence = ["Berlin is the capital and largest city of Germany."]
        self.assertEqual(
            self._hallucination(response, evidence).hallucination_status, "No hallucination"
        )
        self.assertGreaterEqual(self._accuracy(response, evidence).score, 4)
    def test_no_false_positive_on_a_paraphrase(self):
        evidence = ["Paris is the capital of France."]
        first = self._accuracy("Paris is the capital of France.", evidence)
        second = self._accuracy("France's capital is Paris.", evidence)
        self.assertLessEqual(abs(first.score - second.score), 1)
    def test_no_false_positive_on_a_correctly_stated_myth(self):
        """Evidence full of 'myth' and 'not' must not contradict a correct answer."""
        evidence = [
            "The claim that humans only use 10% of their brains is a myth. Brain imaging studies "
            "show that humans use virtually all of their brain, although not all regions are "
            "active at the exact same moment."
        ]
        result = hallucination_judge.detect_hallucination(
            HallucinationInput(
                question="Is it true that we only use 10% of our brains?",
                response="Humans use virtually all of their brain.",
                evidence=evidence,
            )
        )
        self.assertEqual(result.hallucination_status, "No hallucination")
if __name__ == "__main__":
    unittest.main()
