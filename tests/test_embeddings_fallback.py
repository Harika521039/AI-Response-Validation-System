"""
test_embeddings_fallback.py
---------------------------
backend/embeddings.py downloads a sentence-transformers model the first
time it runs. On a machine with no internet that download fails, so the
module falls back to a deterministic hashing vectorizer. These tests force
that failure and check the fallback behaves properly instead of crashing
the whole pipeline.
"""
import math
import unittest
from unittest import mock
from tests.support import OfflineTestCase
from backend import embeddings
class TestEmbeddingFallback(unittest.TestCase):
    def setUp(self):
        embeddings._use_hashing_fallback = False
        self.addCleanup(setattr, embeddings, "_use_hashing_fallback", False)
    def _force_failure(self):
        patcher = mock.patch.object(embeddings, "get_model", side_effect=RuntimeError("no network"))
        patcher.start()
        self.addCleanup(patcher.stop)
    def test_reports_sentence_transformers_before_any_failure(self):
        self.assertEqual(embeddings.embedding_mode(), "sentence-transformers")
    def test_falls_back_when_the_model_cannot_load(self):
        self._force_failure()
        vectors = embeddings.embed_texts(["Paris is the capital of France."])
        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), embeddings.HASHING_DIM)
        self.assertEqual(embeddings.embedding_mode(), "hashing-fallback")
    def test_fallback_vectors_are_unit_length(self):
        self._force_failure()
        vector = embeddings.embed_texts(["some non-empty text here"])[0]
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in vector)), 1.0, places=6)
    def test_fallback_is_deterministic(self):
        self._force_failure()
        first = embeddings.embed_texts(["Jupiter is the largest planet."])[0]
        second = embeddings.embed_texts(["Jupiter is the largest planet."])[0]
        self.assertEqual(first, second)
    def test_unrelated_texts_have_low_similarity(self):
        self._force_failure()
        a = embeddings.embed_texts(["Jupiter is the largest planet in the solar system."])[0]
        b = embeddings.embed_texts(["Bananas are a great source of potassium."])[0]
        self.assertLess(sum(x * y for x, y in zip(a, b)), 0.3)
    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            embeddings.embed_texts([])
        with self.assertRaises(ValueError):
            embeddings.embed_single("   ")
if __name__ == "__main__":
    unittest.main()
