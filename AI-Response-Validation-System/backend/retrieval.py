"""
retrieval.py
------------
High-level orchestration for the retrieval step of Milestone 1:

    1. Build the knowledge base (ingest -> preprocess -> embed -> store)
       ONCE, if it hasn't been built yet.
    2. Given a user's question, embed it and search ChromaDB for the
       most relevant reference evidence.

This is the module app.py calls directly, so the Streamlit UI does not
need to know about datasets, embeddings, or ChromaDB internals.
"""

import logging
from typing import List, Dict

from backend import ingestion, preprocessing, embeddings, vector_store

logger = logging.getLogger(__name__)


def build_knowledge_base_if_needed(use_huggingface: bool = True, limit_per_dataset: int = 50) -> int:
    """
    Populate ChromaDB with reference documents if it is currently empty.

    Returns
    -------
    int
        Number of chunks added (0 if the store already had data).
    """
    if not vector_store.collection_is_empty():
        logger.info("Vector store already populated. Skipping ingestion.")
        return 0

    raw_records = ingestion.load_reference_data(
        use_huggingface=use_huggingface, limit_per_dataset=limit_per_dataset
    )
    documents = preprocessing.build_chunked_documents(raw_records)

    if not documents:
        logger.warning("No documents produced from ingestion; vector store remains empty.")
        return 0

    texts = [doc["text"] for doc in documents]
    vectors = embeddings.embed_texts(texts)
    vector_store.add_documents(documents, vectors)

    return len(documents)


def retrieve_evidence(question: str, top_k: int = 3) -> List[Dict]:
    """
    Given a user's question, return the top_k most relevant reference
    chunks from the vector store.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty for retrieval.")

    query_vector = embeddings.embed_single(question)
    return vector_store.search(query_vector, top_k=top_k)
