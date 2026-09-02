"""
vector_store.py
----------------
Thin wrapper around ChromaDB for storing and searching text embeddings.

Responsibilities:
    - Create/open a persistent Chroma collection on disk (chroma_db/).
    - Add documents (text + embedding + metadata) to the collection.
    - Query the collection with a question embedding and return the
      most relevant reference chunks.
"""

import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "reference_knowledge_base"

_client = None
_collection = None


def get_collection():
    """Return (creating if necessary) the persistent Chroma collection."""
    global _client, _collection
    if _collection is None:
        import chromadb  # lazy import

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def collection_is_empty() -> bool:
    """Check whether the vector store already has any documents in it."""
    collection = get_collection()
    return collection.count() == 0


def add_documents(documents: List[Dict], embeddings: List[List[float]]) -> None:
    """
    Add a batch of documents + their embeddings to ChromaDB.

    Parameters
    ----------
    documents : List[Dict]
        Each item must have at least a "text" key. "source" and
        "question" are stored as metadata if present.
    embeddings : List[List[float]]
        Must be the same length as `documents`.
    """
    if len(documents) != len(embeddings):
        raise ValueError("documents and embeddings must be the same length.")
    if not documents:
        return

    collection = get_collection()
    existing_count = collection.count()

    ids = [f"doc_{existing_count + i}" for i in range(len(documents))]
    texts = [doc["text"] for doc in documents]
    metadatas = [
        {
            "source": doc.get("source", "unknown"),
            "question": doc.get("question", ""),
        }
        for doc in documents
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    logger.info("Added %d documents to ChromaDB.", len(documents))


def search(query_embedding: List[float], top_k: int = 3) -> List[Dict]:
    """
    Search the vector store for the most relevant chunks to a query embedding.

    Returns
    -------
    List[Dict]
        Each item: {"text": str, "source": str, "question": str, "distance": float}
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    output = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, meta, distance in zip(documents, metadatas, distances):
        output.append(
            {
                "text": text,
                "source": meta.get("source", "unknown"),
                "question": meta.get("question", ""),
                "distance": distance,
            }
        )
    return output
