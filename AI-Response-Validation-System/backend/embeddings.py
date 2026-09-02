"""
embeddings.py
-------------
Thin wrapper around sentence-transformers to convert text into vector
embeddings using the all-MiniLM-L6-v2 model.

The model is loaded once and cached (module-level singleton) so it is
not reloaded on every call, which matters a lot inside a Streamlit app
that reruns the script on every interaction.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None  # lazy-loaded singleton


def get_model():
    """Load (once) and return the sentence-transformers model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # lazy import

        logger.info("Loading embedding model: %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Convert a list of strings into a list of embedding vectors.

    Raises
    ------
    ValueError
        If `texts` is empty.
    """
    if not texts:
        raise ValueError("embed_texts() requires at least one text string.")

    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def embed_single(text: str) -> List[float]:
    """Convenience wrapper to embed a single string (e.g. a user question)."""
    if not text or not text.strip():
        raise ValueError("embed_single() requires non-empty text.")
    return embed_texts([text])[0]
