"""
Milestone 4.3 - End-to-end testing and system validation.

Runs the real pipeline offline (deterministic embeddings, heuristic judges,
no LLM) through every stage:

    single: input -> RAG -> four judges -> verdict -> stored result ->
            dashboard -> PDF
    batch:  CSV -> validation -> evaluation -> row results -> aggregate ->
            stored batch -> dashboard -> PDF

Measured validation figures (consistency mean/variance, hallucination
detection rate, batch timing) are written to
evaluation/results/m43_validation_results.json so they are recorded from
real runs, not typed in by hand.
"""
import io
import json
import statistics
import tempfile
import time
from pathlib import Path
from unittest import mock

from tests.support import OfflineTestCase, PROJECT_ROOT

from agents import (
    accuracy_judge,
    batch_evaluator,
    completeness_judge,
    hallucination_judge,
    orchestrator,
    relevance_judge,
    verdict_agent,
)
from agents.schemas import FlaggedClaim, HallucinationOutput
from analytics import dashboard
from backend import database, retrieval
from reporting import pdf_report

RESULTS_PATH = PROJECT_ROOT / "evaluation" / "results" / "m43_validation_results.json"

QUESTION = "Where is the Eiffel Tower and when was it completed?"
REFERENCE = (
    "The Eiffel Tower is located in Paris, France. It was completed in 1889 "
    "and is about 330 metres tall."
)
CASES = {
    "correct": "The Eiffel Tower is located in Paris, France, and it was completed in 1889.",
    "incorrect": "The Eiffel Tower is located in Berlin, Germany, and it was completed in 1925.",
    "incomplete": "The Eiffel Tower is in Paris.",
    "irrelevant": "Bananas are a good source of potassium and grow in tropical climates.",
    "unsupported_claim": (
        "The Eiffel Tower is located in Paris, France. It was designed to host a "
        "secret radio laboratory for the moon program."
    ),
}
HALLUCINATION_LABELS = {
    "correct": False,
    "incomplete": False,
    "incorrect": True,
    "irrelevant": True,
    "unsupported_claim": True,
}


def _record_results(section, data):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if RESULTS_PATH.exists():
        try:
            existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        except ValueError:
            existing = {}
    existing[section] = data
    RESULTS_PATH.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")


def _pdf_text(data):
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _csv(rows):
    buffer = io.StringIO()
    import csv

    writer = csv.writer(buffer)
    writer.writerow(["id", "question", "ai_response", "reference_answer"])
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


class E2EBase(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        db_patch = mock.patch.object(database, "DB_PATH", Path(self.temp_dir.name) / "submissions.db")
        db_patch.start()
        self.addCleanup(db_patch.stop)
        rag_patch = mock.patch.object(retrieval, "retrieve_evidence", return_value=[])
        self.rag = rag_patch.start()
        self.addCleanup(rag_patch.stop)
        database.init_db()

    def evaluate(self, key, reference=REFERENCE):
        return orchestrator.evaluate_response(QUESTION, CASES[key], reference)


class SingleFlowTests(E2EBase):
    def test_complete_single_flow_input_to_pdf(self):
        self.rag.return_value = [
            {"text": REFERENCE, "source": "kb", "question": QUESTION, "distance": 0.1},
        ]
        result = orchestrator.evaluate_response(QUESTION, CASES["correct"], REFERENCE)
        self.rag.assert_called_once()
        self.assertEqual(len(result.retrieved_evidence), 1)
        for section in (result.relevance, result.accuracy, result.hallucination, result.completeness, result.verdict):
            self.assertIsNotNone(section)
        expected = verdict_agent.compute_verdict(
            relevance=result.relevance,
            accuracy=result.accuracy,
            hallucination=result.hallucination,
            completeness=result.completeness,
        )
        self.assertEqual(result.verdict.weighted_overall_score, expected.weighted_overall_score)
        self.assertEqual(result.verdict.verdict, "Pass")

        submission_id = database.save_submission(QUESTION, CASES["correct"], REFERENCE)
        database.save_evaluation_result(submission_id, result)
        records = dashboard.records_from_history(database.get_evaluation_records())
        self.assertEqual(len(records), 1)
        stats = dashboard.compute_statistics(records)
        self.assertEqual(stats["total_responses"], 1)
        self.assertEqual(stats["pass_count"], 1)
        self.assertEqual(stats["average_relevance"], float(result.relevance.score))
        self.assertEqual(stats["average_accuracy"], float(result.accuracy.score))
        self.assertEqual(stats["average_overall_score"], result.verdict.weighted_overall_score)

        data = pdf_report.generate_pdf(records, generated_at="2026-01-01 00:00:00")
        self.assertTrue(data.startswith(b"%PDF"))
        text = _pdf_text(data)
        if text is None:
            self.skipTest("No PDF text extractor installed.")
        self.assertIn("Pass", text)
        self.assertIn(str(result.verdict.weighted_overall_score), text)
        self.assertIn("Eiffel Tower", text)

    def test_correct_response_passes(self):
        result = self.evaluate("correct")
        self.assertEqual(result.accuracy.score, 5)
        self.assertEqual(result.hallucination.hallucination_status, "No hallucination")
        self.assertEqual(result.verdict.verdict, "Pass")

    def test_incorrect_response_does_not_pass(self):
        result = self.evaluate("incorrect")
        self.assertLessEqual(result.accuracy.score, 2)
        self.assertNotEqual(result.verdict.verdict, "Pass")
        self.assertNotEqual(result.hallucination.hallucination_status, "No hallucination")

    def test_incomplete_response_scores_lower_completeness(self):
        complete = self.evaluate("correct")
        incomplete = self.evaluate("incomplete")
        self.assertLess(incomplete.completeness.score, complete.completeness.score)
        self.assertTrue(incomplete.completeness.missing_aspects or incomplete.completeness.partial_aspects)

    def test_irrelevant_response_fails(self):
        result = self.evaluate("irrelevant")
        self.assertLessEqual(result.relevance.score, 2)
        self.assertEqual(result.verdict.verdict, "Fail")

    def test_supported_and_unsupported_claims_are_separated(self):
        result = self.evaluate("unsupported_claim")
        statuses = [c.claim_status for c in result.hallucination.flagged_claims]
        self.assertIn("Supported", statuses)
        self.assertIn("Unsupported", statuses)
        self.assertEqual(result.hallucination.hallucination_status, "Partially hallucinated")
        self.assertNotEqual(result.verdict.verdict, "Pass")

    def test_missing_reference_answer_uses_evidence_path(self):
        result = self.evaluate("correct", reference=None)
        self.assertIsNone(result.reference_answer)
        self.assertIsNotNone(result.verdict.verdict)
        self.assertGreaterEqual(result.accuracy.score, 1)

    def test_invalid_inputs_are_rejected(self):
        for question, response in (("", "x"), ("   ", "x"), ("q", ""), ("q", "  ")):
            with self.assertRaises(ValueError):
                orchestrator.evaluate_response(question, response)

    def test_retrieval_failure_still_produces_a_result(self):
        self.rag.side_effect = RuntimeError("vector store offline")
        result = self.evaluate("correct")
        self.assertEqual(result.retrieved_evidence, [])
        self.assertIsNotNone(result.verdict.verdict)

    def test_agent_failure_surfaces_instead_of_inventing_scores(self):
        with mock.patch.object(accuracy_judge, "evaluate_accuracy", side_effect=RuntimeError("judge crashed")):
            with self.assertRaises(RuntimeError):
                self.evaluate("correct")


class WeightingAndThresholdTests(OfflineTestCase):
    def test_weights_and_thresholds(self):
        self.assertEqual(verdict_agent.WEIGHTS, {"accuracy": 0.35, "hallucination": 0.30, "completeness": 0.20, "relevance": 0.15})
        self.assertEqual(verdict_agent._verdict_from_score(75), "Pass")
        self.assertEqual(verdict_agent._verdict_from_score(74.99), "Needs Improvement")
        self.assertEqual(verdict_agent._verdict_from_score(50), "Needs Improvement")
        self.assertEqual(verdict_agent._verdict_from_score(49.99), "Fail")

    def test_weighted_score_matches_manual_formula_on_real_results(self):
        with mock.patch.object(retrieval, "retrieve_evidence", return_value=[]):
            for key, response in CASES.items():
                r = orchestrator.evaluate_response(QUESTION, response, REFERENCE)
                manual = (
                    0.35 * (r.accuracy.score - 1) / 4 * 100
                    + 0.30 * verdict_agent._normalize_hallucination(r.hallucination)
                    + 0.20 * (r.completeness.score - 1) / 4 * 100
                    + 0.15 * (r.relevance.score - 1) / 4 * 100
                )
                self.assertAlmostEqual(r.verdict.weighted_overall_score, round(manual, 2), places=1, msg=key)

    def test_critical_rule_forces_fail_for_hallucinated(self):
        with mock.patch.object(retrieval, "retrieve_evidence", return_value=[]):
            base = orchestrator.evaluate_response(QUESTION, CASES["correct"], REFERENCE)
        claim = base.hallucination.flagged_claims[0]
        contradicted = FlaggedClaim(**{**claim.model_dump(), "claim_status": "Contradicted"})
        hallucinated = HallucinationOutput(
            hallucination_status="Hallucinated",
            reasoning="Every checked claim is contradicted.",
            flagged_claims=[contradicted],
        )
        verdict = verdict_agent.compute_verdict(
            relevance=base.relevance, accuracy=base.accuracy, hallucination=hallucinated, completeness=base.completeness
        )
        self.assertEqual(verdict.verdict, "Fail")


class ConsistencyTests(E2EBase):
    RUNS = 5

    def test_repeated_scoring_is_consistent(self):
        summary = {}
        for key in CASES:
            scores = [self.evaluate(key).verdict.weighted_overall_score for _ in range(self.RUNS)]
            verdicts = {self.evaluate(key).verdict.verdict for _ in range(2)}
            summary[key] = {
                "runs": self.RUNS,
                "scores": scores,
                "mean": round(statistics.mean(scores), 4),
                "variance": round(statistics.pvariance(scores), 6),
                "verdicts": sorted(verdicts),
            }
            self.assertEqual(statistics.pvariance(scores), 0.0, key)
            self.assertEqual(len(verdicts), 1, key)
        _record_results("consistency", summary)


class HallucinationDetectionTests(E2EBase):
    def test_detection_rate_is_measured_and_recorded(self):
        tp = fn = fp = tn = 0
        details = {}
        for key, should_flag in HALLUCINATION_LABELS.items():
            status = self.evaluate(key).hallucination.hallucination_status
            flagged = status != "No hallucination"
            details[key] = {"expected_flag": should_flag, "status": status}
            if should_flag and flagged:
                tp += 1
            elif should_flag:
                fn += 1
            elif flagged:
                fp += 1
            else:
                tn += 1
        rate = round(tp / (tp + fn) * 100, 2)
        _record_results(
            "hallucination_detection",
            {
                "correct_detections": tp,
                "missed_cases": fn,
                "false_positives": fp,
                "true_negatives": tn,
                "detection_rate_percent": rate,
                "cases": details,
            },
        )
        self.assertEqual(fp, 0)
        self.assertEqual(tp + fn, 3)
        self.assertEqual(rate, 100.0)


class BatchFlowTests(E2EBase):
    def _sample_csv(self):
        return _csv(
            [
                ["r1", QUESTION, CASES["correct"], REFERENCE],
                ["r2", QUESTION, CASES["incorrect"], REFERENCE],
                ["r3", QUESTION, CASES["incomplete"], REFERENCE],
                ["r4", QUESTION, CASES["irrelevant"], REFERENCE],
                ["r5", QUESTION, CASES["unsupported_claim"], ""],
                ["r6", "", CASES["correct"], REFERENCE],
                ["r7", QUESTION, "", REFERENCE],
            ]
        )

    def test_complete_batch_flow_csv_to_pdf(self):
        report = batch_evaluator.evaluate_batch(self._sample_csv())
        self.assertEqual(report.total_rows_in_file, 7)
        self.assertEqual(len(report.evaluated_rows), 5)
        self.assertEqual(len(report.skipped_rows), 2)
        self.assertEqual(len(report.failed_rows), 0)
        for row in report.evaluated_rows:
            self.assertIsNotNone(row.evaluation.verdict.verdict)
        evaluations = [row.evaluation for row in report.evaluated_rows]

        expected_counts = {"Pass": 0, "Needs Improvement": 0, "Fail": 0}
        for e in evaluations:
            expected_counts[e.verdict.verdict] += 1
        manual_rel = round(sum(e.relevance.score for e in evaluations) / 5, 2)
        manual_acc = round(sum(e.accuracy.score for e in evaluations) / 5, 2)
        manual_comp = round(sum(e.completeness.score for e in evaluations) / 5, 2)
        manual_overall = round(sum(e.verdict.weighted_overall_score for e in evaluations) / 5, 2)
        manual_halluc = sum(1 for e in evaluations if e.hallucination.hallucination_status != "No hallucination")

        aggregate = report.aggregate()
        self.assertEqual(aggregate["skipped"], 2)
        self.assertEqual(aggregate["failed"], 0)

        batch_id = database.save_batch_report(report, label="E2E batch")
        records = dashboard.records_from_history(database.get_evaluation_records())
        self.assertEqual(len(records), 5)
        stats = dashboard.compute_statistics(records)
        self.assertEqual(stats["total_responses"], 5)
        self.assertEqual(stats["verdict_counts"], expected_counts)
        for verdict, count in expected_counts.items():
            self.assertEqual(stats["verdict_percentages"][verdict], round(count / 5 * 100, 1))
        self.assertEqual(stats["average_relevance"], manual_rel)
        self.assertEqual(stats["average_accuracy"], manual_acc)
        self.assertEqual(stats["average_completeness"], manual_comp)
        self.assertEqual(stats["average_overall_score"], manual_overall)
        self.assertEqual(stats["hallucinated_responses"], manual_halluc)
        self.assertEqual(stats["hallucination_frequency_percent"], round(manual_halluc / 5 * 100, 1))
        self.assertEqual(stats["batches_covered"], 1)
        self.assertEqual(stats["batch_statistics"][0]["total_responses"] if "total_responses" in stats["batch_statistics"][0] else len(records), 5)
        self.assertTrue(any(b for b in database.get_batches() if batch_id in json.dumps(b, default=str)))

        data = pdf_report.generate_pdf(records, generated_at="2026-01-01 00:00:00")
        text = _pdf_text(data)
        if text is None:
            self.skipTest("No PDF text extractor installed.")
        for e in evaluations:
            self.assertIn(str(e.verdict.weighted_overall_score), text)
        self.assertIn("Needs Improvement", text)
        self.assertIn("Fail", text)
        self.assertIn("secret radio laboratory", text.replace("\n", " "))

    def test_individual_row_failure_does_not_stop_batch(self):
        real = orchestrator.evaluate_response

        def flaky(**kwargs):
            if "Bananas" in kwargs["ai_response"]:
                raise RuntimeError("agent timeout")
            return real(**kwargs)

        report = batch_evaluator.evaluate_batch(self._sample_csv(), evaluate_fn=flaky)
        self.assertEqual(len(report.failed_rows), 1)
        self.assertIn("agent timeout", report.failed_rows[0].error)
        self.assertEqual(len(report.evaluated_rows), 4)
        database.save_batch_report(report, label="flaky")
        records = dashboard.records_from_history(database.get_evaluation_records())
        self.assertEqual(len(records), 4)

    def test_malformed_csv_inputs_are_rejected(self):
        for raw in ("", "   ", "foo,bar\n1,2\n", "question,ai_response\n", b"\xff\xfe\x00"):
            with self.assertRaises(batch_evaluator.BatchValidationError):
                batch_evaluator.evaluate_batch(raw)

    def test_row_with_extra_fields_is_skipped_not_guessed(self):
        raw = "question,ai_response\n" + QUESTION.replace(",", "") + ",Paris,extra\n"
        report = batch_evaluator.evaluate_batch(raw)
        self.assertEqual(len(report.skipped_rows), 1)
        self.assertEqual(len(report.evaluated_rows), 0)

    def test_over_limit_batch_is_rejected(self):
        rows = [[f"r{i}", QUESTION, CASES["correct"], REFERENCE] for i in range(batch_evaluator.MAX_ROWS + 1)]
        with self.assertRaises(batch_evaluator.BatchValidationError):
            batch_evaluator.evaluate_batch(_csv(rows))


class LargeBatchPerformanceTests(E2EBase):
    def test_max_size_batch_dashboard_and_pdf(self):
        keys = list(CASES)
        rows = [
            [f"r{i}", QUESTION, CASES[keys[i % len(keys)]], REFERENCE]
            for i in range(batch_evaluator.MAX_ROWS)
        ]
        start = time.perf_counter()
        report = batch_evaluator.evaluate_batch(_csv(rows))
        eval_seconds = time.perf_counter() - start
        self.assertEqual(len(report.evaluated_rows), batch_evaluator.MAX_ROWS)
        database.save_batch_report(report, label="large")
        records = dashboard.records_from_history(database.get_evaluation_records())
        self.assertEqual(len(records), batch_evaluator.MAX_ROWS)
        start = time.perf_counter()
        stats = dashboard.compute_statistics(records)
        dash_seconds = time.perf_counter() - start
        expected_pass = sum(1 for r in report.evaluated_rows if r.evaluation.verdict.verdict == "Pass")
        self.assertEqual(stats["pass_count"], expected_pass)
        start = time.perf_counter()
        data = pdf_report.generate_pdf(records, generated_at="2026-01-01 00:00:00")
        pdf_seconds = time.perf_counter() - start
        self.assertTrue(data.startswith(b"%PDF"))
        pages = None
        try:
            from pypdf import PdfReader

            pages = len(PdfReader(io.BytesIO(data)).pages)
        except ImportError:
            pass
        _record_results(
            "large_batch_performance",
            {
                "rows": batch_evaluator.MAX_ROWS,
                "evaluation_seconds": round(eval_seconds, 3),
                "dashboard_seconds": round(dash_seconds, 3),
                "pdf_seconds": round(pdf_seconds, 3),
                "pdf_bytes": len(data),
                "pdf_pages": pages,
            },
        )
        self.assertLess(eval_seconds + dash_seconds + pdf_seconds, 120)

    def test_long_text_is_clipped_with_notice_in_pdf(self):
        long_response = "The Eiffel Tower is located in Paris, France. " * 400
        result = orchestrator.evaluate_response(QUESTION, long_response, REFERENCE)
        sid = database.save_submission(QUESTION, long_response, REFERENCE)
        database.save_evaluation_result(sid, result)
        records = dashboard.records_from_history(database.get_evaluation_records())
        data = pdf_report.generate_pdf(records, generated_at="2026-01-01 00:00:00")
        text = _pdf_text(data)
        if text is None:
            self.skipTest("No PDF text extractor installed.")
        self.assertIn("truncated", text.replace("\n", " "))
