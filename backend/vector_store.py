
import json
import hashlib
import logging
import math
from pathlib import Path
from typing import Dict, List
logger = logging.getLogger(__name__)
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
FALLBACK_PATH = CHROMA_DIR / "fallback_store.json"
COLLECTION_NAME = "reference_knowledge_base"
STORE_BATCH_SIZE = 64
_client = None
_collection = None
_use_fallback = False
_fallback_docs: List[Dict] = []
_fallback_loaded = False
_namespace = ""
def use_namespace(name: str = "") -> None:
    """
    Switch to a separate, isolated knowledge base.
    The Streamlit app uses the default namespace. The Milestone 2 benchmark
    uses its own ("benchmark"), so its results depend only on the curated
    reference data it builds for itself and not on whatever the app happened
    to index earlier. Same pipeline, separate storage.
    """
    global _namespace, _collection, _fallback_docs, _fallback_loaded
    _namespace = name or ""
    _collection = None
    _fallback_docs = []
    _fallback_loaded = False
def _collection_name() -> str:
    return f"{COLLECTION_NAME}_{_namespace}" if _namespace else COLLECTION_NAME
def _fallback_path() -> Path:
    return CHROMA_DIR / (f"fallback_store_{_namespace}.json" if _namespace else "fallback_store.json")
def backend_name() -> str:
    """Which storage backend is currently in use: 'chromadb' or 'json-fallback'."""
    return "json-fallback" if _use_fallback else "chromadb"
def _load_fallback() -> List[Dict]:
    global _fallback_docs, _fallback_loaded
    if not _fallback_loaded:
        path = _fallback_path()
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    _fallback_docs = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not read the fallback store (%s); starting empty.", exc)
                _fallback_docs = []
        else:
            _fallback_docs = []
        _fallback_loaded = True
    return _fallback_docs
def _save_fallback() -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_fallback_path(), "w", encoding="utf-8") as handle:
        json.dump(_fallback_docs, handle)
def _cosine(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
def get_collection():
    """
    Return the persistent ChromaDB collection, or None when the JSON
    fallback is in use.
    """
    global _client, _collection, _use_fallback
    if _use_fallback:
        return None
    if _collection is not None:
        return _collection
    try:
        import chromadb
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection(name=_collection_name())
        return _collection
    except Exception as exc:
        logger.warning(
            "ChromaDB is unavailable (%s). Using the local JSON vector store instead. "
            "Install chromadb to use the full vector database.",
            exc,
        )
        _use_fallback = True
        return None
def collection_is_empty() -> bool:
    """True when the knowledge base has not been populated yet."""
    collection = get_collection()
    if collection is None:
        return len(_load_fallback()) == 0
    return collection.count() == 0
def count_documents() -> int:
    """How many chunks are currently stored."""
    collection = get_collection()
    if collection is None:
        return len(_load_fallback())
    return collection.count()
def _document_id(document: Dict) -> str:
    payload = "\x1f".join(
        [document.get("source", "unknown"), document.get("question", ""), document.get("text", "")]
    )
    return "doc_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
def add_documents(
    documents: List[Dict], embeddings: List[List[float]], batch_size: int = STORE_BATCH_SIZE
) -> None:
    """
    Add a batch of documents and their embeddings to the store.
    Parameters
    ----------
    documents : List[Dict]
        Each item needs a "text" key; "source" and "question" are kept as metadata.
    embeddings : List[List[float]]
        Must be the same length as `documents`.
    """
    if len(documents) != len(embeddings):
        raise ValueError("documents and embeddings must be the same length.")
    if not documents:
        return
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    collection = get_collection()
    if collection is None:
        store = _load_fallback()
        existing_ids = {item.get("id") for item in store}
        for doc, vector in zip(documents, embeddings):
            document_id = _document_id(doc)
            if document_id not in existing_ids:
                store.append(
                    {
                        "id": document_id,
                        "text": doc["text"],
                        "source": doc.get("source", "unknown"),
                        "question": doc.get("question", ""),
                        "embedding": list(vector),
                    }
                )
                existing_ids.add(document_id)
        _save_fallback()
        logger.info("Added %d documents to the JSON fallback store.", len(documents))
        return
    for start in range(0, len(documents), batch_size):
        docs_batch = documents[start : start + batch_size]
        embeddings_batch = embeddings[start : start + batch_size]
        collection.upsert(
            ids=[_document_id(doc) for doc in docs_batch],
            documents=[doc["text"] for doc in docs_batch],
            embeddings=embeddings_batch,
            metadatas=[
                {"source": doc.get("source", "unknown"), "question": doc.get("question", "")}
                for doc in docs_batch
            ],
        )
    logger.info("Added %d documents to ChromaDB.", len(documents))
def search(query_embedding: List[float], top_k: int = 3, query_text: str = "") -> List[Dict]:
    """
    Return the top_k most relevant chunks for a query embedding.
    Each result: {"text", "source", "question", "distance"}, where distance
    is a cosine distance (lower means more relevant).
    """
    collection = get_collection()
    if collection is None:
        store = _load_fallback()
        if not store:
            return []
        expected_dim = len(query_embedding)
        usable = [doc for doc in store if len(doc.get("embedding", [])) == expected_dim]
        if not usable:
            logger.warning(
                "The stored vectors were built with a different embedding model. "
                "Clearing the local store so it can be rebuilt."
            )
            reset()
            return []
        query_terms = set(query_text.lower().split())
        scored = []
        for doc in usable:
            similarity = _cosine(query_embedding, doc["embedding"])
            metadata_terms = set((doc.get("question", "") + " " + doc.get("text", "")).lower().split())
            lexical = len(query_terms & metadata_terms) / len(query_terms) if query_terms else 0.0
            scored.append((0.8 * similarity + 0.2 * lexical, similarity, doc))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {
                "text": doc["text"],
                "source": doc.get("source", "unknown"),
                "question": doc.get("question", ""),
                "distance": round(1.0 - similarity, 6),
            }
            for _, similarity, doc in scored[:top_k]
        ]
    if collection.count() == 0:
        return []
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(max(top_k * 4, top_k), collection.count()),
    )
    output = []
    for text, meta, distance in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        output.append(
            {
                "text": text,
                "source": (meta or {}).get("source", "unknown"),
                "question": (meta or {}).get("question", ""),
                "distance": float(distance),
            }
        )
    if query_text:
        terms = set(query_text.lower().split())
        output.sort(
            key=lambda item: (
                item["distance"] - 0.2 * (
                    len(terms & set((item.get("question", "") + " " + item["text"]).lower().split()))
                    / len(terms)
                    if terms else 0.0
                )
            )
        )
    return output[:top_k]
def reset() -> None:
    """
    Delete everything in the knowledge base. Used by tests and when the
    reference data changes; the store is rebuilt on the next run.
    """
    global _collection, _client, _fallback_docs, _fallback_loaded
    collection = get_collection()
    if collection is None:
        _fallback_docs = []
        _fallback_loaded = True
        _save_fallback()
        return
    try:
        _client.delete_collection(_collection_name())
    except Exception as exc:
        logger.warning("Could not delete the Chroma collection (%s).", exc)
    _collection = None
