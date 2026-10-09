
import logging
from typing import List, Dict
from agents.utils import content_words
from backend import ingestion, preprocessing, embeddings, vector_store
logger = logging.getLogger(__name__)
MIN_QUESTION_OVERLAP_RATIO = 0.25
MAX_EVIDENCE_DISTANCE = 0.95
def _is_relevant(question: str, item: Dict) -> bool:
    """True when a retrieved chunk is actually about the question."""
    distance = item.get("distance")
    if distance is not None and distance > MAX_EVIDENCE_DISTANCE:
        return False
    question_words = set(content_words(question))
    if not question_words:
        return True
    chunk_words = set(content_words(f"{item.get('question', '')} {item.get('text', '')}"))
    overlap = len(question_words & chunk_words) / len(question_words)
    return overlap >= MIN_QUESTION_OVERLAP_RATIO
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
    Given a user's question, return the reference chunks that are actually
    relevant to it - at most top_k, and possibly none.
    Returning nothing is a valid, deliberate outcome: the judges handle "no
    evidence" by falling back to their own reasoning, which is far better
    than being handed an unrelated chunk and treating it as ground truth.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty for retrieval.")
    query_vector = embeddings.embed_single(question)
    results = vector_store.search(query_vector, top_k=top_k, query_text=question)
    relevant = [item for item in results if _is_relevant(question, item)]
    if results and not relevant:
        logger.info(
            "Retrieved %d chunk(s) for the question but none passed the relevance cutoff; "
            "continuing without reference evidence.",
            len(results),
        )
    return relevant[:top_k]
