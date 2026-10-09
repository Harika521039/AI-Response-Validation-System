
import unittest
from tests.support import OfflineTestCase
from agents.schemas import (
    AccuracyOutput,
    EvaluationResult,
    FlaggedClaim,
    HallucinationOutput,
    RelevanceInput,
    RelevanceOutput,
    SchemaValidationError,
)
class TestSchemaValidation(unittest.TestCase):
    def test_relevance_output_accepts_valid_score(self):
        result = RelevanceOutput(score=4, label="Mostly relevant", reasoning="Because.")
        self.assertEqual(result.score, 4)
        self.assertEqual(result.mode, "heuristic")
    def test_relevance_output_rejects_out_of_range_score(self):
        with self.assertRaises(SchemaValidationError):
            RelevanceOutput(score=7, label="x", reasoning="y")
        with self.assertRaises(SchemaValidationError):
            RelevanceOutput(score=0, label="x", reasoning="y")
    def test_relevance_input_rejects_empty_text(self):
        with self.assertRaises(SchemaValidationError):
            RelevanceInput(question="   ", response="something")
    def test_flagged_claim_rejects_unknown_status(self):
        with self.assertRaises(SchemaValidationError):
            FlaggedClaim(claim="a claim", claim_status="Probably true", reasoning="why")
    def test_hallucination_output_rejects_unknown_status(self):
        with self.assertRaises(SchemaValidationError):
            HallucinationOutput(hallucination_status="Very hallucinated", reasoning="why")
    def test_model_dump_is_json_safe(self):
        output = HallucinationOutput(
            hallucination_status="Partially hallucinated",
            reasoning="mixed",
            flagged_claims=[FlaggedClaim(claim="c", claim_status="Supported", reasoning="r")],
        )
        dumped = output.model_dump()
        self.assertEqual(dumped["flagged_claims"][0]["claim_status"], "Supported")
    def test_problem_claims_only_returns_non_supported(self):
        output = HallucinationOutput(
            hallucination_status="Partially hallucinated",
            reasoning="mixed",
            flagged_claims=[
                FlaggedClaim(claim="ok", claim_status="Supported", reasoning="r"),
                FlaggedClaim(claim="bad", claim_status="Unsupported", reasoning="r"),
            ],
        )
        self.assertEqual(len(output.problem_claims), 1)
        self.assertEqual(output.problem_claims[0].claim, "bad")
    def test_evaluation_result_summary_has_no_invented_overall_score(self):
        result = EvaluationResult(
            question="q",
            response="r",
            relevance=RelevanceOutput(score=5, label="Completely relevant", reasoning="r"),
            accuracy=AccuracyOutput(score=5, label="Completely correct", reasoning="r"),
            hallucination=HallucinationOutput(hallucination_status="No hallucination", reasoning="r"),
        )
        summary = result.summary()
        self.assertEqual(summary["relevance"], "5/5 (Completely relevant)")
        self.assertEqual(summary["hallucination"], "No hallucination")
        self.assertNotIn("overall_score", summary)
if __name__ == "__main__":
    unittest.main()
