"""
ingestion.py
------------
Responsible for loading the reference knowledge base.

Milestone 1 design:
    - The "real" path uses the Hugging Face `datasets` library to load
      TruthfulQA and SQuAD.
    - Because those downloads can be slow (or unavailable in an offline /
      restricted environment), a small local demo dataset
      (data/demo_dataset.json) is used as an automatic fallback so the
      rest of the pipeline (embeddings, ChromaDB, retrieval, UI) can
      always be demonstrated even without internet access.
    - Every record returned by this module is normalized into the same
      shape, regardless of source:

        {
            "source": "truthful_qa" | "squad" | "demo",
            "question": str,
            "answer": str,
            "context": str,
        }

This keeps preprocessing.py, embeddings.py, and vector_store.py source-agnostic.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

DEMO_DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "demo_dataset.json"


def load_demo_dataset() -> List[Dict]:
    """Load the small local demo dataset shipped with the project."""
    if not DEMO_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Demo dataset not found at {DEMO_DATASET_PATH}. "
            "Make sure data/demo_dataset.json exists."
        )
    with open(DEMO_DATASET_PATH, "r", encoding="utf-8") as f:
        raw_records = json.load(f)

    return [
        {
            "source": "demo",
            "question": r.get("question", ""),
            "answer": r.get("answer", ""),
            "context": r.get("context", r.get("answer", "")),
        }
        for r in raw_records
    ]


def load_truthful_qa(limit: int = 50) -> List[Dict]:
    """
    Load a subset of the TruthfulQA dataset from Hugging Face.

    Requires internet access and the `datasets` library. If it is not
    available, this raises an exception which callers should catch and
    fall back to the demo dataset (see load_reference_data()).
    """
    from datasets import load_dataset  # imported lazily so the app can run without it

    ds = load_dataset("truthful_qa", "generation", split="validation")
    records = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        records.append(
            {
                "source": "truthful_qa",
                "question": row.get("question", ""),
                "answer": row.get("best_answer", ""),
                "context": row.get("best_answer", ""),
            }
        )
    return records


def load_squad(limit: int = 50) -> List[Dict]:
    """
    Load a subset of the SQuAD dataset from Hugging Face.

    Requires internet access and the `datasets` library. If it is not
    available, this raises an exception which callers should catch and
    fall back to the demo dataset (see load_reference_data()).
    """
    from datasets import load_dataset  # imported lazily so the app can run without it

    ds = load_dataset("squad", split="train")
    records = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        answers = row.get("answers", {}).get("text", [])
        answer = answers[0] if answers else ""
        records.append(
            {
                "source": "squad",
                "question": row.get("question", ""),
                "answer": answer,
                "context": row.get("context", ""),
            }
        )
    return records


def load_reference_data(use_huggingface: bool = True, limit_per_dataset: int = 50) -> List[Dict]:
    """
    Main entry point used by the rest of the app.

    Tries TruthfulQA + SQuAD from Hugging Face first (if use_huggingface=True).
    On ANY failure (no internet, library missing, slow download, etc.) it
    automatically falls back to the bundled demo dataset so Milestone 1
    can always be demoed.
    """
    if use_huggingface:
        try:
            records = []
            records.extend(load_truthful_qa(limit=limit_per_dataset))
            records.extend(load_squad(limit=limit_per_dataset))
            logger.info("Loaded %d records from TruthfulQA + SQuAD.", len(records))
            return records
        except Exception as exc:  # noqa: BLE001 - broad by design, this is a fallback path
            logger.warning(
                "Falling back to demo dataset because Hugging Face datasets "
                "could not be loaded (%s).",
                exc,
            )

    return load_demo_dataset()
