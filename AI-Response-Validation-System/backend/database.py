"""
database.py
------------
Handles all SQLite storage for Milestone 1.

Responsibility (kept intentionally small for Milestone 1):
    - Create a local SQLite database file.
    - Store each user submission: question, AI-generated response,
      optional reference text, and a timestamp.
    - Provide a simple function to fetch recent submissions (useful for
      debugging / demo purposes).

No judging, scoring, or hallucination-detection logic lives here.
That is intentionally left for later milestones.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Database file lives inside the data/ folder so it's easy to find and delete.
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "submissions.db"


def _ensure_data_dir() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection():
    """Context manager that yields a SQLite connection and closes it safely."""
    _ensure_data_dir()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the submissions table if it does not already exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                reference_text TEXT,
                created_at TEXT NOT NULL
            )
            """
        )


def save_submission(question: str, ai_response: str, reference_text: Optional[str] = None) -> int:
    """
    Save a single submission to the database.

    Returns
    -------
    int
        The row id of the newly inserted submission.

    Raises
    ------
    ValueError
        If question or ai_response is empty/whitespace-only.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    if not ai_response or not ai_response.strip():
        raise ValueError("AI-generated response cannot be empty.")

    init_db()  # safe no-op if already created
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO submissions (question, ai_response, reference_text, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                question.strip(),
                ai_response.strip(),
                reference_text.strip() if reference_text else None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return cursor.lastrowid


def get_recent_submissions(limit: int = 10):
    """Return the most recent submissions, newest first."""
    init_db()
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM submissions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def count_submissions() -> int:
    """Return the total number of stored submissions."""
    init_db()
    with get_connection() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM submissions")
        return cursor.fetchone()[0]
