"""
test_basic.py
-------------
Basic unit tests for Milestone 1.

These tests intentionally avoid downloading models or datasets from the
internet (sentence-transformers / Hugging Face `datasets`) so they can
run quickly and offline. They cover:
    - preprocessing.py (cleaning + chunking)
    - database.py (SQLite storage, using a temporary DB file)
    - ingestion.py demo dataset fallback

Run with:  pytest tests/
"""

import sys
import sqlite3
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import preprocessing, ingestion, database


# ---------------------------------------------------------------------------
# preprocessing.py tests
# ---------------------------------------------------------------------------
def test_clean_text_collapses_whitespace():
    dirty = "  This   has\n\nextra   spaces.  "
    assert preprocessing.clean_text(dirty) == "This has extra spaces."


def test_clean_text_handles_empty_string():
    assert preprocessing.clean_text("") == ""
    assert preprocessing.clean_text(None) == ""


def test_chunk_text_short_text_returns_single_chunk():
    text = "This is a short sentence."
    chunks = preprocessing.chunk_text(text, max_words=80)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_long_text_produces_multiple_chunks():
    text = " ".join(["word"] * 200)
    chunks = preprocessing.chunk_text(text, max_words=50, overlap=10)
    assert len(chunks) > 1
    # Every chunk should have at most max_words words
    for chunk in chunks:
        assert len(chunk.split(" ")) <= 50


def test_standardize_record_fills_missing_fields():
    raw = {"question": " What?  ", "answer": "An answer."}
    result = preprocessing.standardize_record(raw)
    assert result["question"] == "What?"
    assert result["answer"] == "An answer."
    assert result["context"] == ""
    assert result["source"] == "unknown"


def test_build_chunked_documents_skips_empty_records():
    records = [
        {"source": "demo", "question": "Q1", "answer": "", "context": ""},
        {"source": "demo", "question": "Q2", "answer": "Has an answer", "context": ""},
    ]
    docs = preprocessing.build_chunked_documents(records)
    assert len(docs) == 1
    assert docs[0]["text"] == "Has an answer"


# ---------------------------------------------------------------------------
# ingestion.py tests (demo fallback only — no network required)
# ---------------------------------------------------------------------------
def test_load_demo_dataset_returns_normalized_records():
    records = ingestion.load_demo_dataset()
    assert len(records) > 0
    for r in records:
        assert set(r.keys()) == {"source", "question", "answer", "context"}
        assert r["source"] == "demo"


def test_load_reference_data_falls_back_when_huggingface_disabled():
    records = ingestion.load_reference_data(use_huggingface=False)
    assert len(records) > 0
    assert all(r["source"] == "demo" for r in records)


# ---------------------------------------------------------------------------
# database.py tests (uses a temporary DB file, isolated from real data)
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_db(monkeypatch):
    """Point database.DB_PATH at a temporary file for the duration of the test."""
    tmp_dir = tempfile.mkdtemp()
    tmp_path = Path(tmp_dir) / "test_submissions.db"
    monkeypatch.setattr(database, "DB_PATH", tmp_path)
    database.init_db()
    yield tmp_path


def test_save_and_retrieve_submission(temp_db):
    submission_id = database.save_submission(
        question="What is 2+2?",
        ai_response="4",
        reference_text="Basic arithmetic",
    )
    assert submission_id == 1

    recent = database.get_recent_submissions(limit=5)
    assert len(recent) == 1
    assert recent[0]["question"] == "What is 2+2?"
    assert recent[0]["ai_response"] == "4"
    assert recent[0]["reference_text"] == "Basic arithmetic"


def test_save_submission_rejects_empty_question(temp_db):
    with pytest.raises(ValueError):
        database.save_submission(question="", ai_response="Some answer")


def test_save_submission_rejects_empty_response(temp_db):
    with pytest.raises(ValueError):
        database.save_submission(question="Some question?", ai_response="   ")


def test_count_submissions(temp_db):
    assert database.count_submissions() == 0
    database.save_submission("Q1", "A1")
    database.save_submission("Q2", "A2")
    assert database.count_submissions() == 2
