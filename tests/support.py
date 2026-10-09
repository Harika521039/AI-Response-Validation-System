
import os
import sys
import unittest
from pathlib import Path
from unittest import mock
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from backend import embeddings
class OfflineTestCase(unittest.TestCase):
    """Base class: deterministic embeddings and no LLM, for every test."""
    def setUp(self):
        super().setUp()
        patcher_texts = mock.patch.object(
            embeddings, "embed_texts", side_effect=embeddings._hashing_embed_texts
        )
        patcher_single = mock.patch.object(
            embeddings, "embed_single", side_effect=lambda text: embeddings._hashing_embed(text)
        )
        patcher_env = mock.patch.dict(os.environ, {}, clear=False)
        patcher_texts.start()
        patcher_single.start()
        patcher_env.start()
        os.environ.pop("ANTHROPIC_API_KEY", None)
        self.addCleanup(patcher_texts.stop)
        self.addCleanup(patcher_single.stop)
        self.addCleanup(patcher_env.stop)
