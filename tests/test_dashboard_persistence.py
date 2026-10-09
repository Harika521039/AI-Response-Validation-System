"""
test_dashboard_persistence.py
-----------------------------
Focused tests for the storage the M4.1 dashboard reads from.
A batch run has to survive as individual evaluation records, otherwise
"quality trends across batches" would have nothing real to plot. These
tests store an actual batch report in a temporary database and check that
what comes back out still matches the agent results that went in.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from agents import batch_evaluator
from analytics import dashboard
from backend import database
from tests.support import OfflineTestCase
BATCH_CSV = (
    "id,question,ai_response,reference_answer\n"
    "case-1,What is the capital of France?,The capital of France is Paris.,"
    "Paris is the capital of France.\n"
    "case-2,What is the capital of France?,The capital of France is Berlin.,"
    "Paris is the capital of France.\n"
).encode("utf-8")
class TemporaryDatabaseTestCase(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        patcher = mock.patch.object(
            database, "DB_PATH", Path(self.temp_dir.name) / "submissions.db"
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        database.init_db()
class BatchPersistenceTests(TemporaryDatabaseTestCase):
    def setUp(self):
        super().setUp()
        self.report = batch_evaluator.evaluate_batch(BATCH_CSV)
    def test_batch_rows_are_stored_as_individual_records(self):
        batch_id = database.save_batch_report(self.report, label="Batch A")
        stored = database.get_evaluation_records()
        self.assertEqual(len(stored), len(self.report.evaluated_rows))
        for item in stored:
            self.assertEqual(item["batch_id"], batch_id)
            self.assertEqual(item["source"], "batch")
            self.assertEqual(item["batch_label"], "Batch A")
    def test_stored_payload_matches_the_original_agent_results(self):
        database.save_batch_report(self.report, label="Batch A")
        stored = {item["row_id"]: item for item in database.get_evaluation_records()}
        for row in self.report.evaluated_rows:
            item = stored[row.display_id]
            self.assertEqual(item["evaluation"], row.evaluation.model_dump())
            self.assertEqual(item["relevance_score"], row.evaluation.relevance.score)
            self.assertEqual(item["verdict"], row.evaluation.verdict.verdict)
            self.assertAlmostEqual(
                item["weighted_overall_score"], row.evaluation.verdict.weighted_overall_score
            )
    def test_batch_index_records_the_report_counts(self):
        batch_id = database.save_batch_report(self.report, label="Batch A")
        batches = {item["batch_id"]: item for item in database.get_batches()}
        stats = self.report.aggregate()
        self.assertEqual(batches[batch_id]["evaluated"], stats["evaluated"])
        self.assertEqual(batches[batch_id]["skipped"], stats["skipped"])
        self.assertEqual(batches[batch_id]["failed"], stats["failed"])
        self.assertEqual(batches[batch_id]["total_rows_in_file"], stats["total_rows_in_file"])
    def test_saving_the_same_batch_twice_does_not_duplicate_records(self):
        batch_id = database.save_batch_report(self.report, label="Batch A")
        database.save_batch_report(self.report, label="Batch A", batch_id=batch_id)
        stored = database.get_evaluation_records()
        self.assertEqual(len(stored), len(self.report.evaluated_rows))
        self.assertEqual(len(database.get_batches()), 1)
    def test_dashboard_statistics_over_stored_records_match_the_report(self):
        database.save_batch_report(self.report, label="Batch A")
        records = dashboard.records_from_history(database.get_evaluation_records())
        stats = dashboard.compute_statistics(records)
        aggregate = self.report.aggregate()
        self.assertEqual(stats["total_responses"], aggregate["evaluated"])
        self.assertAlmostEqual(stats["average_relevance"], aggregate["average_relevance"])
        self.assertAlmostEqual(stats["average_accuracy"], aggregate["average_accuracy"])
        self.assertAlmostEqual(stats["average_completeness"], aggregate["average_completeness"])
        self.assertAlmostEqual(stats["average_overall_score"], aggregate["average_overall_score"])
        self.assertEqual(stats["verdict_counts"], aggregate["verdict_counts"])
        self.assertAlmostEqual(
            stats["hallucination_frequency_percent"],
            aggregate["hallucination_frequency_percent"],
        )
    def test_two_batches_produce_a_two_point_trend(self):
        database.save_batch_report(self.report, label="Batch A")
        database.save_batch_report(batch_evaluator.evaluate_batch(BATCH_CSV), label="Batch B")
        records = dashboard.records_from_history(database.get_evaluation_records())
        stats = dashboard.compute_statistics(records)
        self.assertEqual(stats["batches_covered"], 2)
        self.assertEqual(len(stats["quality_trend"]), 2)
        self.assertEqual({item["Batch"] for item in stats["quality_trend"]}, {"Batch A", "Batch B"})
    def test_records_from_batch_report_match_records_from_storage(self):
        batch_id = database.save_batch_report(self.report, label="Batch A")
        live = dashboard.records_from_batch_report(self.report, batch_id=batch_id, batch_label="Batch A")
        stored = dashboard.records_from_history(database.get_evaluation_records())
        self.assertEqual(len(live), len(stored))
        self.assertEqual(
            sorted(record.payload["response"] for record in live),
            sorted(record.payload["response"] for record in stored),
        )
    def test_single_evaluations_are_kept_separate_from_batches(self):
        database.save_batch_report(self.report, label="Batch A")
        row = self.report.evaluated_rows[0]
        submission_id = database.save_submission(row.question, row.ai_response)
        database.save_evaluation_result(submission_id, row.evaluation)
        records = dashboard.records_from_history(database.get_evaluation_records())
        sources = {record.source for record in records}
        self.assertEqual(sources, {"batch", "single"})
        singles = [record for record in records if record.source == "single"]
        self.assertEqual(singles[0].group_label, "Single evaluations")
    def test_history_without_a_payload_is_not_treated_as_a_record(self):
        database.save_submission("A question?", "A response.")
        records = dashboard.records_from_history(database.get_evaluation_records())
        self.assertEqual(records, [])
    def test_empty_database_produces_an_empty_dashboard(self):
        records = dashboard.records_from_history(database.get_evaluation_records())
        stats = dashboard.compute_statistics(records)
        self.assertEqual(stats["total_responses"], 0)
        self.assertIsNone(stats["average_accuracy"])
if __name__ == "__main__":
    unittest.main()
