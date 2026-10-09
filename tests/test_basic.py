"""
test_basic.py
-------------
Milestone 1 regression tests: preprocessing, the demo-dataset ingestion
fallback, and SQLite submission storage. These must keep passing so that
Milestone 2 work cannot quietly break the foundation.
No network access is required.
"""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from tests.support import OfflineTestCase
from backend import database, ingestion, preprocessing
class TestPreprocessing(unittest.TestCase):
    def test_clean_text_collapses_whitespace(self):
        self.assertEqual(preprocessing.clean_text("  This   has\n\nextra   spaces.  "), "This has extra spaces.")
    def test_clean_text_handles_empty_input(self):
        self.assertEqual(preprocessing.clean_text(""), "")
        self.assertEqual(preprocessing.clean_text(None), "")
    def test_short_text_stays_as_one_chunk(self):
        chunks = preprocessing.chunk_text("This is a short sentence.", max_words=80)
        self.assertEqual(chunks, ["This is a short sentence."])
    def test_long_text_is_split_into_multiple_chunks(self):
        chunks = preprocessing.chunk_text(" ".join(["word"] * 200), max_words=50, overlap=10)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.split(" ")), 50)
    def test_standardize_record_fills_missing_fields(self):
        result = preprocessing.standardize_record({"question": " What?  ", "answer": "An answer."})
        self.assertEqual(result["question"], "What?")
        self.assertEqual(result["context"], "")
        self.assertEqual(result["source"], "unknown")
    def test_records_without_text_are_skipped(self):
        documents = preprocessing.build_chunked_documents(
            [
                {"source": "demo", "question": "Q1", "answer": "", "context": ""},
                {"source": "demo", "question": "Q2", "answer": "Has an answer", "context": ""},
            ]
        )
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["text"], "Has an answer")
class TestIngestion(unittest.TestCase):
    def test_demo_dataset_records_are_normalized(self):
        records = ingestion.load_demo_dataset()
        self.assertGreater(len(records), 0)
        for record in records:
            self.assertEqual(set(record.keys()), {"source", "question", "answer", "context"})
            self.assertEqual(record["source"], "demo")
    def test_falls_back_to_the_demo_dataset_without_huggingface(self):
        records = ingestion.load_reference_data(use_huggingface=False)
        self.assertGreater(len(records), 0)
        self.assertTrue(all(r["source"] == "demo" for r in records))
    def test_falls_back_when_huggingface_raises(self):
        with mock.patch.object(ingestion, "load_truthful_qa", side_effect=RuntimeError("no network")):
            records = ingestion.load_reference_data(use_huggingface=True)
        self.assertTrue(all(r["source"] == "demo" for r in records))
class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        patcher = mock.patch.object(database, "DB_PATH", Path(self.tmp_dir.name) / "test_submissions.db")
        patcher.start()
        self.addCleanup(patcher.stop)
        database.init_db()
    def test_save_and_retrieve_a_submission(self):
        submission_id = database.save_submission("What is 2+2?", "4", "Basic arithmetic")
        self.assertEqual(submission_id, 1)
        recent = database.get_recent_submissions(limit=5)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["question"], "What is 2+2?")
        self.assertEqual(recent[0]["reference_text"], "Basic arithmetic")
    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            database.save_submission(question="", ai_response="Some answer")
    def test_empty_response_is_rejected(self):
        with self.assertRaises(ValueError):
            database.save_submission(question="Some question?", ai_response="   ")
    def test_counting_submissions(self):
        self.assertEqual(database.count_submissions(), 0)
        database.save_submission("Q1", "A1")
        database.save_submission("Q2", "A2")
        self.assertEqual(database.count_submissions(), 2)
    def test_the_table_really_exists_on_disk(self):
        database.save_submission("Q", "A")
        with sqlite3.connect(str(database.DB_PATH)) as conn:
            names = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        self.assertIn("submissions", names)
if __name__ == "__main__":
    unittest.main()
