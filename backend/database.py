import sqlite3
import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
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
        existing = {row[1] for row in conn.execute("PRAGMA table_info(submissions)").fetchall()}
        additions = {
            "relevance_score": "INTEGER",
            "accuracy_score": "INTEGER",
            "completeness_score": "INTEGER",
            "hallucination_status": "TEXT",
            "weighted_overall_score": "REAL",
            "verdict": "TEXT",
            "evaluation_json": "TEXT",
            "source": "TEXT",
            "batch_id": "TEXT",
            "batch_label": "TEXT",
            "row_id": "TEXT",
        }
        for column, sql_type in additions.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE submissions ADD COLUMN {column} {sql_type}")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS batches (
                batch_id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                created_at TEXT NOT NULL,
                total_rows_in_file INTEGER NOT NULL,
                evaluated INTEGER NOT NULL,
                skipped INTEGER NOT NULL,
                failed INTEGER NOT NULL
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
    init_db()
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
def save_evaluation_result(submission_id: int, result) -> None:
    """Attach a completed orchestrator result to an existing submission."""
    if result.verdict is None or result.completeness is None:
        raise ValueError("A complete Milestone 3 evaluation result is required.")
    init_db()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            UPDATE submissions
            SET relevance_score = ?, accuracy_score = ?, completeness_score = ?,
                hallucination_status = ?, weighted_overall_score = ?, verdict = ?,
                evaluation_json = ?
            WHERE id = ?
            """,
            (
                result.relevance.score,
                result.accuracy.score,
                result.completeness.score,
                result.hallucination.hallucination_status,
                result.verdict.weighted_overall_score,
                result.verdict.verdict,
                json.dumps(result.model_dump()),
                int(submission_id),
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"Submission {submission_id} does not exist.")
def get_evaluation_history(limit: int = 100):
    """Return completed evaluations only, newest first."""
    init_db()
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT * FROM submissions
            WHERE evaluation_json IS NOT NULL
            ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
def get_dashboard_statistics():
    """Aggregate only persisted, completed evaluations; never infer missing scores."""
    init_db()
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT COUNT(*) AS total_evaluations,
                   AVG(accuracy_score) AS average_accuracy,
                   AVG(relevance_score) AS average_relevance,
                   AVG(completeness_score) AS average_completeness,
                   AVG(CASE WHEN hallucination_status IN ('Hallucinated', 'Partially hallucinated')
                            THEN 1.0 ELSE 0.0 END) * 100 AS hallucination_rate,
                   AVG(CASE WHEN verdict = 'Pass' THEN 1.0 ELSE 0.0 END) * 100 AS pass_rate
            FROM submissions WHERE evaluation_json IS NOT NULL
            """
        ).fetchone()
        stats = dict(row)
        verdict_rows = conn.execute(
            """SELECT verdict, COUNT(*) AS count FROM submissions
               WHERE evaluation_json IS NOT NULL GROUP BY verdict"""
        ).fetchall()
        stats["verdict_counts"] = {item["verdict"]: item["count"] for item in verdict_rows}
        return stats
def save_batch_report(report, label: Optional[str] = None, batch_id: Optional[str] = None) -> str:
    """
    Persist a finished BatchEvaluationReport so the Milestone 4 dashboard can
    show quality trends across batches.
    Only rows the orchestrator actually evaluated are stored, each as its own
    submission carrying its complete evaluation payload. Skipped and failed
    rows are counted on the batch record but never given invented scores.
    Returns
    -------
    str
        The identifier of the stored batch.
    """
    init_db()
    stats = report.aggregate()
    created_at = datetime.now(timezone.utc).isoformat()
    identifier = batch_id or f"batch-{uuid.uuid4().hex[:12]}"
    batch_label = (label or "").strip() or f"Batch {created_at[:19].replace('T', ' ')} UTC"
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO batches
                (batch_id, label, created_at, total_rows_in_file, evaluated, skipped, failed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                identifier,
                batch_label,
                created_at,
                int(stats["total_rows_in_file"]),
                int(stats["evaluated"]),
                int(stats["skipped"]),
                int(stats["failed"]),
            ),
        )
        conn.execute("DELETE FROM submissions WHERE batch_id = ?", (identifier,))
        for row in report.evaluated_rows:
            evaluation = row.evaluation
            if evaluation is None or evaluation.verdict is None:
                continue
            conn.execute(
                """
                INSERT INTO submissions
                    (question, ai_response, reference_text, created_at,
                     relevance_score, accuracy_score, completeness_score,
                     hallucination_status, weighted_overall_score, verdict,
                     evaluation_json, source, batch_id, batch_label, row_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.question,
                    row.ai_response,
                    row.reference_answer,
                    created_at,
                    evaluation.relevance.score,
                    evaluation.accuracy.score,
                    evaluation.completeness.score if evaluation.completeness else None,
                    evaluation.hallucination.hallucination_status,
                    evaluation.verdict.weighted_overall_score,
                    evaluation.verdict.verdict,
                    json.dumps(evaluation.model_dump()),
                    "batch",
                    identifier,
                    batch_label,
                    row.display_id,
                ),
            )
    return identifier
def get_batches():
    """Every stored batch, oldest first, with the counts recorded at save time."""
    init_db()
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM batches ORDER BY created_at ASC, batch_id ASC").fetchall()
        return [dict(row) for row in rows]
def get_evaluation_records(limit: Optional[int] = None):
    """
    Completed evaluations with their payload already decoded, oldest first.
    This is what the Milestone 4 dashboard reads. A row whose stored payload
    cannot be decoded is left out rather than shown with missing numbers.
    """
    init_db()
    query = """
        SELECT * FROM submissions
        WHERE evaluation_json IS NOT NULL
        ORDER BY id ASC
    """
    parameters: tuple = ()
    if limit is not None:
        query += " LIMIT ?"
        parameters = (int(limit),)
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, parameters).fetchall()
    records = []
    for row in rows:
        item = dict(row)
        try:
            item["evaluation"] = json.loads(item["evaluation_json"])
        except (TypeError, ValueError):
            continue
        item["source"] = item.get("source") or ("batch" if item.get("batch_id") else "single")
        records.append(item)
    return records
