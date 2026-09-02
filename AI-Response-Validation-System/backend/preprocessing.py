"""
preprocessing.py
-----------------
Cleans and standardizes raw records coming from ingestion.py, and splits
long context text into smaller chunks so embeddings are more meaningful
for semantic search.

Kept deliberately simple for Milestone 1 (no NLP libraries beyond
plain string operations), but structured so it can be extended later.
"""

import re
from typing import List, Dict


def clean_text(text: str) -> str:
    """Basic text cleanup: strip whitespace, collapse repeated spaces/newlines."""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def chunk_text(text: str, max_words: int = 80, overlap: int = 15) -> List[str]:
    """
    Split text into overlapping word chunks.

    Parameters
    ----------
    text : str
        The text to chunk.
    max_words : int
        Maximum number of words per chunk.
    overlap : int
        Number of words shared between consecutive chunks, to preserve
        context across chunk boundaries.

    Returns
    -------
    List[str]
        One or more chunks. If the text is shorter than max_words,
        a single chunk is returned.
    """
    text = clean_text(text)
    if not text:
        return []

    words = text.split(" ")
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap  # move forward, keeping some overlap

    return chunks


def standardize_record(record: Dict) -> Dict:
    """Clean the question/answer/context fields of a single raw record."""
    return {
        "source": record.get("source", "unknown"),
        "question": clean_text(record.get("question", "")),
        "answer": clean_text(record.get("answer", "")),
        "context": clean_text(record.get("context", "")),
    }


def build_chunked_documents(records: List[Dict], max_words: int = 80, overlap: int = 15) -> List[Dict]:
    """
    Convert a list of raw records into a flat list of "documents" ready
    for embedding. Each document is one chunk of context text, tagged
    with metadata so we know where it came from.

    Returns
    -------
    List[Dict]
        Each item has:
            {
                "text": str,          # the chunk of text to embed
                "source": str,        # e.g. "truthful_qa", "squad", "demo"
                "question": str,      # original question (for reference)
            }
    """
    documents = []
    for raw in records:
        clean = standardize_record(raw)
        # Prefer context; fall back to answer if context is empty.
        base_text = clean["context"] or clean["answer"]
        if not base_text:
            continue

        for chunk in chunk_text(base_text, max_words=max_words, overlap=overlap):
            documents.append(
                {
                    "text": chunk,
                    "source": clean["source"],
                    "question": clean["question"],
                }
            )

    return documents
