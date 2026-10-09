
from tests.support import OfflineTestCase
from agents import batch_evaluator
from agents.schemas import BatchValidationError, EvaluationResult
def _fake_result(question, ai_response, reference_answer=None, top_k=3):
    """A stand-in evaluate_response: lets the batch tests stay fast and
    deterministic while still exercising the real batch code path."""
    from agents.schemas import AccuracyOutput, HallucinationOutput, RelevanceOutput
    return EvaluationResult(
        question=question,
        response=ai_response,
        reference_answer=reference_answer,
        relevance=RelevanceOutput(score=4, label="Relevant", reasoning="ok"),
        accuracy=AccuracyOutput(score=4, label="Accurate", reasoning="ok"),
        hallucination=HallucinationOutput(
            hallucination_status="No hallucination", reasoning="ok"
        ),
    )
GOOD_CSV = (
    "question,ai_response,reference_answer\n"
    "What is the capital of France?,Paris is the capital of France.,Paris\n"
    "Who wrote Hamlet?,Hamlet was written by William Shakespeare.,\n"
)
class TestFileAndColumnValidation(OfflineTestCase):
    def test_empty_file_is_rejected(self):
        with self.assertRaises(BatchValidationError):
            batch_evaluator.parse_csv(b"")
    def test_binary_file_is_rejected(self):
        with self.assertRaises(BatchValidationError):
            batch_evaluator.parse_csv(b"PNG\x00\x01binary")
    def test_missing_required_column_is_rejected(self):
        with self.assertRaises(BatchValidationError) as ctx:
            batch_evaluator.parse_csv("question,notes\nWhy?,none\n")
        self.assertIn("ai_response", str(ctx.exception))
    def test_header_only_file_is_rejected(self):
        with self.assertRaises(BatchValidationError) as ctx:
            batch_evaluator.parse_csv("question,ai_response\n")
        self.assertIn("no data rows", str(ctx.exception))
    def test_duplicate_column_is_rejected(self):
        with self.assertRaises(BatchValidationError) as ctx:
            batch_evaluator.parse_csv("question,question,ai_response\na,b,c\n")
        self.assertIn("more than once", str(ctx.exception))
    def test_row_limit_is_enforced(self):
        rows = "".join(f"q{i},r{i}\n" for i in range(batch_evaluator.MAX_ROWS + 1))
        with self.assertRaises(BatchValidationError) as ctx:
            batch_evaluator.parse_csv("question,ai_response\n" + rows)
        self.assertIn("limit", str(ctx.exception))
    def test_optional_columns_may_be_absent(self):
        rows, ignored = batch_evaluator.parse_csv("question,ai_response\nWhy?,Because.\n")
        self.assertEqual(len(rows), 1)
        self.assertEqual(ignored, [])
        self.assertEqual(rows[0]["question"], "Why?")
    def test_header_aliases_and_case_are_accepted(self):
        rows, ignored = batch_evaluator.parse_csv(
            "Prompt,AI Answer,Expected Answer,Notes\nWhy?,Because.,Because of X,ignore me\n"
        )
        self.assertEqual(rows[0]["question"], "Why?")
        self.assertEqual(rows[0]["ai_response"], "Because.")
        self.assertEqual(rows[0]["reference_answer"], "Because of X")
        self.assertEqual(ignored, ["Notes"])
    def test_row_numbers_follow_the_file_lines(self):
        rows, _ = batch_evaluator.parse_csv(GOOD_CSV)
        self.assertEqual([row["row_number"] for row in rows], [2, 3])
class TestRowValidation(OfflineTestCase):
    def test_blank_required_fields_are_reported(self):
        rows, _ = batch_evaluator.parse_csv(
            "question,ai_response\n,Some response\nA question?,\n"
        )
        valid, skipped = batch_evaluator.validate_rows(rows)
        self.assertEqual(valid, [])
        self.assertEqual(len(skipped), 2)
        self.assertIn("'question' is empty", skipped[0].error)
        self.assertIn("'ai_response' is empty", skipped[1].error)
        for row in skipped:
            self.assertEqual(row.status, "skipped")
            self.assertIsNone(row.evaluation)
    def test_row_with_only_an_id_is_reported_as_both_fields_empty(self):
        rows, _ = batch_evaluator.parse_csv("id,question,ai_response\n7,,\n")
        valid, skipped = batch_evaluator.validate_rows(rows)
        self.assertEqual(valid, [])
        self.assertIn("Both", skipped[0].error)
        self.assertEqual(skipped[0].row_id, "7")
    def test_completely_blank_lines_are_ignored_not_reported(self):
        rows, _ = batch_evaluator.parse_csv("question,ai_response\nQ?,R.\n,\n\n")
        valid, skipped = batch_evaluator.validate_rows(rows)
        self.assertEqual(len(valid), 1)
        self.assertEqual(skipped, [])
    def test_oversized_field_is_skipped(self):
        long_value = "x" * (batch_evaluator.MAX_FIELD_LENGTH + 1)
        rows, _ = batch_evaluator.parse_csv(f"question,ai_response\nWhy?,{long_value}\n")
        valid, skipped = batch_evaluator.validate_rows(rows)
        self.assertEqual(valid, [])
        self.assertIn("character limit", skipped[0].error)
    def test_extra_values_in_a_row_are_skipped(self):
        rows, _ = batch_evaluator.parse_csv("question,ai_response\nWhy?,Because,extra\n")
        valid, skipped = batch_evaluator.validate_rows(rows)
        self.assertEqual(valid, [])
        self.assertIn("more values than the header", skipped[0].error)
class TestEvaluateBatch(OfflineTestCase):
    def test_valid_rows_are_evaluated_and_results_preserved(self):
        report = batch_evaluator.evaluate_batch(GOOD_CSV, evaluate_fn=_fake_result)
        self.assertEqual(len(report.evaluated_rows), 2)
        for row in report.evaluated_rows:
            self.assertEqual(row.status, "evaluated")
            self.assertIsInstance(row.evaluation, EvaluationResult)
            self.assertEqual(row.evaluation.question, row.question)
            self.assertEqual(row.evaluation.relevance.score, 4)
            self.assertIsNotNone(row.summary)
        self.assertEqual(report.evaluated_rows[0].reference_answer, "Paris")
        self.assertIsNone(report.evaluated_rows[1].reference_answer)
    def test_invalid_rows_are_skipped_and_the_batch_continues(self):
        csv_text = (
            "question,ai_response\n"
            "Good question?,Good response.\n"
            ",Missing question\n"
            "Another good question?,Another good response.\n"
        )
        report = batch_evaluator.evaluate_batch(csv_text, evaluate_fn=_fake_result)
        self.assertEqual(len(report.evaluated_rows), 2)
        self.assertEqual(len(report.skipped_rows), 1)
        self.assertEqual(report.skipped_rows[0].row_number, 3)
        self.assertEqual(report.total_rows_in_file, 3)
    def test_a_failing_row_does_not_stop_the_batch(self):
        calls = {"n": 0}
        def flaky(question, ai_response, reference_answer=None, top_k=3):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("API temporarily unavailable")
            return _fake_result(question, ai_response, reference_answer)
        csv_text = (
            "question,ai_response\n"
            "Q one?,R one.\n"
            "Q two?,R two.\n"
            "Q three?,R three.\n"
        )
        report = batch_evaluator.evaluate_batch(csv_text, evaluate_fn=flaky)
        self.assertEqual(calls["n"], 3)
        self.assertEqual(len(report.evaluated_rows), 2)
        self.assertEqual(len(report.failed_rows), 1)
        failed = report.failed_rows[0]
        self.assertEqual(failed.row_number, 3)
        self.assertIn("API temporarily unavailable", failed.error)
        self.assertIsNone(failed.evaluation)
    def test_progress_is_reported_for_every_valid_row(self):
        seen = []
        csv_text = "question,ai_response\nQ1?,R1.\n,skip me\nQ2?,R2.\n"
        batch_evaluator.evaluate_batch(
            csv_text,
            evaluate_fn=_fake_result,
            progress_callback=lambda done, total, row: seen.append((done, total, row.row_number)),
        )
        self.assertEqual(seen, [(1, 2, 2), (2, 2, 4)])
    def test_a_broken_progress_callback_does_not_stop_the_batch(self):
        def boom(done, total, row):
            raise RuntimeError("UI exploded")
        report = batch_evaluator.evaluate_batch(
            GOOD_CSV, evaluate_fn=_fake_result, progress_callback=boom
        )
        self.assertEqual(len(report.evaluated_rows), 2)
    def test_aggregate_counts_and_averages(self):
        csv_text = "question,ai_response\nQ1?,R1.\n,skip\nQ2?,R2.\n"
        report = batch_evaluator.evaluate_batch(csv_text, evaluate_fn=_fake_result)
        stats = report.aggregate()
        self.assertEqual(stats["evaluated"], 2)
        self.assertEqual(stats["skipped"], 1)
        self.assertEqual(stats["failed"], 0)
        self.assertEqual(stats["average_relevance"], 4)
        self.assertEqual(stats["average_accuracy"], 4)
    def test_results_export_includes_every_row(self):
        csv_text = "question,ai_response\nQ1?,R1.\n,skip\n"
        report = batch_evaluator.evaluate_batch(csv_text, evaluate_fn=_fake_result)
        exported = batch_evaluator.results_to_csv(report)
        lines = [line for line in exported.splitlines() if line.strip()]
        self.assertEqual(len(lines), 3)
        self.assertIn("row_number", lines[0])
        self.assertIn("evaluated", exported)
        self.assertIn("skipped", exported)
class TestEvaluateBatchWithRealOrchestrator(OfflineTestCase):
    """One end-to-end run through the actual Orchestrator (offline,
    heuristic mode) so the batch really does drive Relevance, Accuracy,
    Hallucination, Completeness and the Verdict Agent."""
    def test_real_orchestrator_produces_all_four_judges_and_a_verdict(self):
        report = batch_evaluator.evaluate_batch(
            "question,ai_response,reference_answer\n"
            "What is the capital of France?,Paris is the capital of France.,"
            "Paris is the capital of France.\n"
        )
        self.assertEqual(len(report.evaluated_rows), 1)
        evaluation = report.evaluated_rows[0].evaluation
        self.assertIsNotNone(evaluation.relevance)
        self.assertIsNotNone(evaluation.accuracy)
        self.assertIsNotNone(evaluation.hallucination)
        self.assertIsNotNone(evaluation.completeness)
        self.assertIsNotNone(evaluation.verdict)
        self.assertIn(evaluation.verdict.verdict, ("Pass", "Needs Improvement", "Fail"))
class TestResultsTableAndStatistics(OfflineTestCase):
    """M3.4 (Task 2C): the batch results table and the batch-level
    statistics - averages, Pass/Needs Improvement/Fail counts and
    hallucination frequency as a percentage."""
    def _report(self):
        def mixed(question, ai_response, reference_answer=None, top_k=3):
            from agents.schemas import (
                AccuracyOutput,
                CompletenessOutput,
                HallucinationOutput,
                RelevanceOutput,
                VerdictOutput,
            )
            hallucinated = "bad" in ai_response
            return EvaluationResult(
                question=question,
                response=ai_response,
                reference_answer=reference_answer,
                relevance=RelevanceOutput(score=4, label="Relevant", reasoning="ok"),
                accuracy=AccuracyOutput(
                    score=2 if hallucinated else 5,
                    label="Mostly incorrect" if hallucinated else "Completely accurate",
                    reasoning="ok",
                ),
                hallucination=HallucinationOutput(
                    hallucination_status="Hallucinated" if hallucinated else "No hallucination",
                    reasoning="ok",
                ),
                completeness=CompletenessOutput(
                    score=3, label="Mostly complete", reasoning="ok"
                ),
                verdict=VerdictOutput(
                    relevance_score=4,
                    accuracy_score=2 if hallucinated else 5,
                    completeness_score=3,
                    hallucination_status="Hallucinated" if hallucinated else "No hallucination",
                    normalized_scores={
                        "accuracy": 40.0,
                        "hallucination": 0.0 if hallucinated else 100.0,
                        "completeness": 60.0,
                        "relevance": 80.0,
                    },
                    weights={
                        "accuracy": 0.35,
                        "hallucination": 0.30,
                        "completeness": 0.20,
                        "relevance": 0.15,
                    },
                    weighted_overall_score=40.0 if hallucinated else 90.0,
                    verdict="Fail" if hallucinated else "Pass",
                    major_issues=[],
                    reasoning="ok",
                ),
            )
        csv_text = (
            "id,question,ai_response\n"
            "r1,Q one?,Good response.\n"
            "r2,Q two?,A bad response.\n"
            "r3,,Missing question\n"
        )
        return batch_evaluator.evaluate_batch(csv_text, evaluate_fn=mixed)
    def test_results_table_has_the_required_columns_for_every_row(self):
        table = self._report().results_table()
        self.assertEqual(len(table), 3)
        self.assertEqual(
            list(table[0].keys()),
            [
                "ID",
                "Relevance",
                "Accuracy",
                "Hallucination",
                "Completeness",
                "Overall",
                "Verdict",
                "Status",
            ],
        )
        self.assertEqual(table[0]["ID"], "r1")
        self.assertEqual(table[0]["Relevance"], 4)
        self.assertEqual(table[0]["Accuracy"], 5)
        self.assertEqual(table[0]["Completeness"], 3)
        self.assertEqual(table[0]["Overall"], 90.0)
        self.assertEqual(table[0]["Verdict"], "Pass")
        self.assertEqual(table[0]["Hallucination"], "No hallucination")
    def test_skipped_row_keeps_its_place_in_the_table_with_its_reason(self):
        table = self._report().results_table()
        skipped_line = table[2]
        self.assertEqual(skipped_line["Verdict"], "-")
        self.assertEqual(skipped_line["Overall"], "-")
        self.assertTrue(skipped_line["Status"].startswith("skipped:"))
    def test_results_table_can_exclude_unevaluated_rows(self):
        table = self._report().results_table(include_unevaluated=False)
        self.assertEqual(len(table), 2)
    def test_averages_and_verdict_counts(self):
        stats = self._report().aggregate()
        self.assertEqual(stats["average_relevance"], 4.0)
        self.assertEqual(stats["average_accuracy"], 3.5)
        self.assertEqual(stats["average_completeness"], 3.0)
        self.assertEqual(stats["average_overall_score"], 65.0)
        self.assertEqual(
            stats["verdict_counts"], {"Pass": 1, "Needs Improvement": 0, "Fail": 1}
        )
    def test_hallucination_frequency_is_a_percentage_of_evaluated_rows(self):
        stats = self._report().aggregate()
        self.assertEqual(stats["hallucination_rows"], 1)
        self.assertEqual(stats["hallucination_frequency_percent"], 50.0)
        self.assertEqual(stats["hallucination_counts"]["Hallucinated"], 1)
        self.assertEqual(stats["hallucination_counts"]["No hallucination"], 1)
        self.assertEqual(stats["hallucination_counts"]["Partially hallucinated"], 0)
    def test_statistics_are_safe_when_nothing_could_be_evaluated(self):
        report = batch_evaluator.evaluate_batch(
            "question,ai_response\n,Missing question\n", evaluate_fn=_fake_result
        )
        stats = report.aggregate()
        self.assertEqual(stats["evaluated"], 0)
        self.assertEqual(stats["skipped"], 1)
        self.assertIsNone(stats["hallucination_frequency_percent"])
        self.assertEqual(stats["verdict_counts"]["Pass"], 0)
        self.assertEqual(len(report.results_table()), 1)
