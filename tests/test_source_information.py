"""
test_source_information.py
--------------------------
M3.4 - the optional `source_information` column.
Covers the full path of that column:
    1. It is part of the documented CSV contract (and its aliases parse).
    2. It is validated like the other free-text columns.
    3. It is preserved on every row result - evaluated, skipped and failed.
    4. It reaches the Orchestrator, which treats it as supplied evidence and
       keeps it on the EvaluationResult the row preserves.
    5. It appears in the detailed row results and in the CSV export.
"""
from unittest import mock
from tests.support import OfflineTestCase
from agents import batch_evaluator, orchestrator
from agents.schemas import (
    AccuracyOutput,
    EvaluationResult,
    HallucinationOutput,
    RelevanceOutput,
)
def _fake_result(question, ai_response, reference_answer=None, top_k=3, source_information=None):
    return EvaluationResult(
        question=question,
        response=ai_response,
        reference_answer=reference_answer,
        source_information=source_information,
        relevance=RelevanceOutput(score=4, label="Relevant", reasoning="ok"),
        accuracy=AccuracyOutput(score=4, label="Accurate", reasoning="ok"),
        hallucination=HallucinationOutput(
            hallucination_status="No hallucination", reasoning="ok"
        ),
    )
CSV_WITH_SOURCE = (
    "id,question,ai_response,reference_answer,source_information\n"
    "A1,What is the capital of France?,Paris is the capital of France.,Paris,"
    "Paris has been the capital of France since 987 AD.\n"
    "A2,Who wrote Hamlet?,Hamlet was written by William Shakespeare.,,\n"
)
class TestSourceInformationContract(OfflineTestCase):
    def test_source_information_is_an_optional_column(self):
        self.assertIn("source_information", batch_evaluator.OPTIONAL_COLUMNS)
        self.assertNotIn("source_information", batch_evaluator.REQUIRED_COLUMNS)
    def test_column_is_parsed_and_absence_is_not_an_error(self):
        rows, ignored = batch_evaluator.parse_csv(CSV_WITH_SOURCE)
        self.assertEqual(ignored, [])
        self.assertIn("987 AD", rows[0]["source_information"])
        self.assertEqual(rows[1]["source_information"], "")
        rows, _ = batch_evaluator.parse_csv("question,ai_response\nWhy?,Because.\n")
        self.assertEqual(rows[0].get("source_information"), None)
    def test_header_aliases_are_accepted(self):
        for header in ("Source Information", "source", "Source-Info", "context", "source_text"):
            rows, ignored = batch_evaluator.parse_csv(
                f"question,ai_response,{header}\nWhy?,Because.,Some source text\n"
            )
            self.assertEqual(ignored, [], header)
            self.assertEqual(rows[0]["source_information"], "Some source text", header)
    def test_oversized_source_information_is_skipped_with_a_reason(self):
        long_value = "x" * (batch_evaluator.MAX_FIELD_LENGTH + 1)
        rows, _ = batch_evaluator.parse_csv(
            f"question,ai_response,source_information\nWhy?,Because.,{long_value}\n"
        )
        valid, skipped = batch_evaluator.validate_rows(rows)
        self.assertEqual(valid, [])
        self.assertIn("source_information", skipped[0].error)
        self.assertIn("character limit", skipped[0].error)
class TestSourceInformationIsPreserved(OfflineTestCase):
    def test_evaluated_rows_keep_and_forward_the_value(self):
        seen = []
        def spy(question, ai_response, reference_answer=None, top_k=3, source_information=None):
            seen.append(source_information)
            return _fake_result(
                question, ai_response, reference_answer, top_k, source_information
            )
        report = batch_evaluator.evaluate_batch(CSV_WITH_SOURCE, evaluate_fn=spy)
        rows = report.evaluated_rows
        self.assertEqual(len(rows), 2)
        self.assertIn("987 AD", rows[0].source_information)
        self.assertIsNone(rows[1].source_information)
        self.assertIn("987 AD", seen[0])
        self.assertIsNone(seen[1])
        self.assertIn("987 AD", rows[0].evaluation.source_information)
        self.assertTrue(rows[0].evaluation.summary()["source_information_provided"])
        self.assertFalse(rows[1].evaluation.summary()["source_information_provided"])
    def test_skipped_rows_keep_the_value(self):
        report = batch_evaluator.evaluate_batch(
            "question,ai_response,source_information\n,No question here,Some source\n",
            evaluate_fn=_fake_result,
        )
        self.assertEqual(report.skipped_rows[0].source_information, "Some source")
    def test_failed_rows_keep_the_value(self):
        def boom(**kwargs):
            raise RuntimeError("LLM unavailable")
        report = batch_evaluator.evaluate_batch(
            "question,ai_response,source_information\nWhy?,Because.,Some source\n",
            evaluate_fn=boom,
        )
        failed = report.failed_rows[0]
        self.assertEqual(failed.source_information, "Some source")
        self.assertIn("LLM unavailable", failed.error)
    def test_an_evaluator_without_the_parameter_still_works(self):
        """Backwards compatibility: a row with no source information never
        passes the extra argument, so older evaluators keep running."""
        def legacy(question, ai_response, reference_answer=None, top_k=3):
            return _fake_result(question, ai_response, reference_answer, top_k)
        report = batch_evaluator.evaluate_batch(
            "question,ai_response\nWhy?,Because.\n", evaluate_fn=legacy
        )
        self.assertEqual(len(report.evaluated_rows), 1)
    def test_export_csv_contains_the_column(self):
        report = batch_evaluator.evaluate_batch(CSV_WITH_SOURCE, evaluate_fn=_fake_result)
        exported = batch_evaluator.results_to_csv(report)
        self.assertIn("source_information", exported.splitlines()[0])
        self.assertIn("987 AD", exported)
class TestOrchestratorUsesSourceInformation(OfflineTestCase):
    def test_source_information_is_given_to_the_judges_as_evidence(self):
        with mock.patch.object(orchestrator.retrieval, "retrieve_evidence", return_value=[]):
            result = orchestrator.evaluate_response(
                question="What is the capital of France?",
                ai_response="Paris is the capital of France.",
                source_information="Paris is the capital and largest city of France.",
            )
        self.assertEqual(
            result.source_information, "Paris is the capital and largest city of France."
        )
        self.assertEqual(result.accuracy.evidence_mode, "rag_evidence")
        self.assertIsNotNone(result.verdict)
    def test_blank_source_information_is_treated_as_absent(self):
        with mock.patch.object(orchestrator.retrieval, "retrieve_evidence", return_value=[]):
            result = orchestrator.evaluate_response(
                question="What is the capital of France?",
                ai_response="Paris is the capital of France.",
                source_information="   ",
            )
        self.assertIsNone(result.source_information)
        self.assertEqual(result.accuracy.evidence_mode, "none")
