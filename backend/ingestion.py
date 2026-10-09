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
    from datasets import load_dataset
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
    from datasets import load_dataset
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
    Loads TruthfulQA and SQuAD from Hugging Face when use_huggingface=True,
    and ALWAYS adds the bundled demo records on top. If the Hugging Face
    download fails for any reason (no internet, library missing, dataset
    moved), the demo records alone keep the whole pipeline working.
    """
    records = []
    if use_huggingface:
        try:
            records.extend(load_truthful_qa(limit=limit_per_dataset))
            records.extend(load_squad(limit=limit_per_dataset))
            logger.info("Loaded %d records from TruthfulQA and SQuAD.", len(records))
        except Exception as exc:
            logger.warning(
                "Could not load the Hugging Face datasets (%s). Continuing with the "
                "bundled demo dataset only.",
                exc,
            )
    records.extend(load_demo_dataset())
    return records
