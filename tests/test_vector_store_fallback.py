"""
test_vector_store_fallback.py
-----------------------------
The knowledge base normally lives in ChromaDB. When chromadb is not
installed or cannot start, backend/vector_store.py switches to a local JSON
store with plain cosine search so Milestone 1 retrieval keeps working.
These tests exercise that fallback directly, against a temporary directory
so the real chroma_db/ folder is untouched.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from tests.support import OfflineTestCase
from backend import vector_store
class TestJsonFallbackStore(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        tmp_path = Path(self.tmp_dir.name)
        patches = [
            mock.patch.object(vector_store, "CHROMA_DIR", tmp_path),
            mock.patch.object(vector_store, "FALLBACK_PATH", tmp_path / "fallback_store.json"),
            mock.patch.object(vector_store, "_use_fallback", True),
            mock.patch.object(vector_store, "_fallback_docs", []),
            mock.patch.object(vector_store, "_fallback_loaded", True),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
    def test_store_starts_empty(self):
        self.assertTrue(vector_store.collection_is_empty())
        self.assertEqual(vector_store.count_documents(), 0)
        self.assertEqual(vector_store.backend_name(), "json-fallback")
    def test_documents_can_be_added_and_searched(self):
        documents = [
            {"text": "Paris is the capital of France.", "source": "demo", "question": "capital?"},
            {"text": "Jupiter is the largest planet.", "source": "demo", "question": "planet?"},
        ]
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        vector_store.add_documents(documents, vectors)
        self.assertEqual(vector_store.count_documents(), 2)
        results = vector_store.search([1.0, 0.0], top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("Paris", results[0]["text"])
        self.assertEqual(results[0]["source"], "demo")
        self.assertLess(results[0]["distance"], 0.01)
    def test_search_returns_nothing_when_the_store_is_empty(self):
        self.assertEqual(vector_store.search([1.0, 0.0], top_k=3), [])
    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            vector_store.add_documents([{"text": "one"}], [[1.0], [0.0]])
    def test_reset_empties_the_store(self):
        vector_store.add_documents([{"text": "Paris is the capital."}], [[1.0, 0.0]])
        vector_store.reset()
        self.assertEqual(vector_store.count_documents(), 0)
if __name__ == "__main__":
    unittest.main()
