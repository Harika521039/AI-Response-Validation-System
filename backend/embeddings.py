import hashlib
import logging
import math
import re
from typing import List
logger = logging.getLogger(__name__)
MODEL_NAME = "all-MiniLM-L6-v2"
HASHING_DIM = 512
EMBEDDING_BATCH_SIZE = 32
_model = None
_use_hashing_fallback = False
def get_model():
    """Load (once) and return the sentence-transformers model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
    return _model
def embedding_mode() -> str:
    """
    Returns which embedding backend is currently active:
        "sentence-transformers" -> real semantic embeddings (normal case)
        "hashing-fallback"      -> offline, lexical-only fallback
    Useful for the UI / evaluation report to be transparent about which
    mode produced a given result.
    """
    return "hashing-fallback" if _use_hashing_fallback else "sentence-transformers"
_WORD_RE = re.compile(r"[a-z0-9]+")
def _hash_bucket(word: str) -> int:
    digest = hashlib.md5(word.encode("utf-8")).hexdigest()
    return int(digest, 16) % HASHING_DIM
def _hashing_embed(text: str) -> List[float]:
    vec = [0.0] * HASHING_DIM
    for word in _WORD_RE.findall((text or "").lower()):
        vec[_hash_bucket(word)] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec
def _hashing_embed_texts(texts: List[str]) -> List[List[float]]:
    return [_hashing_embed(t) for t in texts]
def embed_texts(texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> List[List[float]]:
  
    if not texts:
        raise ValueError("embed_texts() requires at least one text string.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    global _use_hashing_fallback
    if not _use_hashing_fallback:
        try:
            model = get_model()
            output: List[List[float]] = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                vectors = model.encode(batch, show_progress_bar=False, batch_size=batch_size)
                output.extend(vectors.tolist())
            return output
        except Exception as exc:
            logger.warning(
                "Could not load/use the sentence-transformers model (%s). "
                "Falling back to an offline hashing-based embedding for the "
                "rest of this session. Semantic quality will be reduced "
                "(lexical overlap only) until the model can be downloaded.",
                exc,
            )
            _use_hashing_fallback = True
    return _hashing_embed_texts(texts)
def embed_single(text: str) -> List[float]:
    """Convenience wrapper to embed a single string (e.g. a user question)."""
    if not text or not text.strip():
        raise ValueError("embed_single() requires non-empty text.")
    return embed_texts([text])[0]
