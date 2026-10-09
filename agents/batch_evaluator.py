
import csv
import io
import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from agents import orchestrator
from agents.schemas import (
    BatchEvaluationReport,
    BatchRowResult,
    BatchValidationError,
)
logger = logging.getLogger(__name__)
REQUIRED_COLUMNS = ("question", "ai_response")
OPTIONAL_COLUMNS = ("reference_answer", "id", "source_information")
COLUMN_ALIASES = {
    "question": "question",
    "prompt": "question",
    "query": "question",
    "ai_response": "ai_response",
    "response": "ai_response",
    "answer": "ai_response",
    "ai_answer": "ai_response",
    "generated_response": "ai_response",
    "reference_answer": "reference_answer",
    "reference": "reference_answer",
    "expected_answer": "reference_answer",
    "ground_truth": "reference_answer",
    "id": "id",
    "row_id": "id",
    "case_id": "id",
    "source_information": "source_information",
    "source": "source_information",
    "source_info": "source_information",
    "source_text": "source_information",
    "context": "source_information",
    "supporting_information": "source_information",
}
MAX_ROWS = 200
MAX_FIELD_LENGTH = 20000
def _normalize_header(header: str) -> str:
    return (header or "").strip().lower().replace(" ", "_").replace("-", "_")
def decode_csv_bytes(raw: bytes) -> str:
    """
    Turn uploaded bytes into text.
    Raises
    ------
    BatchValidationError
        If the upload is empty or is not a text/CSV file at all (for example
        a PDF or an image renamed to .csv).
    """
    if raw is None:
        raise BatchValidationError("No file was received.")
    if not raw.strip():
        raise BatchValidationError("The uploaded file is empty.")
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise BatchValidationError(
            "The uploaded file is not a readable text CSV file. Please upload a UTF-8 CSV."
        )
    if "\x00" in text:
        raise BatchValidationError(
            "The uploaded file looks like a binary file, not a CSV file."
        )
    return text
def map_columns(headers: Sequence[str]) -> Tuple[Dict[str, int], List[str]]:
    """
    Map the CSV's own headers onto the canonical column names.
    Returns
    -------
    (mapping, ignored_columns)
        mapping maps canonical name -> column index.
        ignored_columns lists headers that are not part of the contract; they
        are reported to the user and then ignored, not treated as errors.
    Raises
    ------
    BatchValidationError
        If a required column is missing, or the same canonical column is
        supplied twice (which would make the row ambiguous).
    """
    mapping: Dict[str, int] = {}
    ignored: List[str] = []
    duplicates: List[str] = []
    for index, header in enumerate(headers):
        canonical = COLUMN_ALIASES.get(_normalize_header(header))
        if canonical is None:
            if (header or "").strip():
                ignored.append(header.strip())
            continue
        if canonical in mapping:
            duplicates.append(canonical)
            continue
        mapping[canonical] = index
    if duplicates:
        raise BatchValidationError(
            "These columns appear more than once: "
            + ", ".join(sorted(set(duplicates)))
            + ". Please supply each column only once."
        )
    missing = [name for name in REQUIRED_COLUMNS if name not in mapping]
    if missing:
        raise BatchValidationError(
            "The CSV is missing the required column(s): "
            + ", ".join(missing)
            + ". Required columns are: "
            + ", ".join(REQUIRED_COLUMNS)
            + ". Optional columns are: "
            + ", ".join(OPTIONAL_COLUMNS)
            + "."
        )
    return mapping, ignored
def parse_csv(raw) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    File-level + column-level validation, then extract the raw rows.
    Parameters
    ----------
    raw : bytes or str
        The uploaded CSV content.
    Returns
    -------
    (raw_rows, ignored_columns)
        raw_rows is a list of dicts with the keys 'row_number' (the CSV line
        number, header counted as line 1) plus whichever canonical columns
        were present. No row-level validation has happened yet.
    Raises
    ------
    BatchValidationError
        For problems that make the whole file unusable.
    """
    text = decode_csv_bytes(raw) if isinstance(raw, (bytes, bytearray)) else (raw or "")
    if not text.strip():
        raise BatchValidationError("The uploaded file is empty.")
    try:
        reader = csv.reader(io.StringIO(text))
        all_rows = [row for row in reader]
    except csv.Error as exc:
        raise BatchValidationError(f"The CSV file could not be parsed: {exc}") from exc
    all_rows = [row for row in all_rows if any((cell or "").strip() for cell in row)]
    if not all_rows:
        raise BatchValidationError("The uploaded file contains no data.")
    headers = all_rows[0]
    mapping, ignored = map_columns(headers)
    data_rows = all_rows[1:]
    if not data_rows:
        raise BatchValidationError(
            "The CSV has a valid header row but no data rows to evaluate."
        )
    if len(data_rows) > MAX_ROWS:
        raise BatchValidationError(
            f"This batch has {len(data_rows)} rows, which is above the limit of "
            f"{MAX_ROWS} rows per upload. Please split the file."
        )
    raw_rows: List[Dict[str, str]] = []
    for offset, row in enumerate(data_rows):
        entry: Dict[str, str] = {"row_number": offset + 2}
        for canonical, index in mapping.items():
            entry[canonical] = row[index] if index < len(row) else ""
        entry["_field_count"] = len(row)
        entry["_header_count"] = len(headers)
        raw_rows.append(entry)
    return raw_rows, ignored
def validate_row(entry: Dict[str, str]) -> Optional[str]:
    """
    Check one raw row. Returns None when the row is valid, otherwise a
    human-readable reason why it must be skipped.
    """
    if entry.get("_field_count", 0) > entry.get("_header_count", 0):
        return (
            "The row has more values than the header has columns "
            "(a comma inside a value probably needs quoting)."
        )
    question = (entry.get("question") or "").strip()
    response = (entry.get("ai_response") or "").strip()
    if not question and not response:
        return "Both 'question' and 'ai_response' are empty."
    if not question:
        return "'question' is empty (it is a required column)."
    if not response:
        return "'ai_response' is empty (it is a required column)."
    for column in ("question", "ai_response", "reference_answer", "source_information"):
        value = entry.get(column) or ""
        if len(value) > MAX_FIELD_LENGTH:
            return (
                f"'{column}' is {len(value)} characters long, above the "
                f"{MAX_FIELD_LENGTH}-character limit for a single field."
            )
    return None
def validate_rows(raw_rows: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[BatchRowResult]]:
    """
    Split the raw rows into the valid ones (to evaluate) and skipped ones.
    Invalid rows are never repaired or guessed at: they are returned as
    BatchRowResult entries with status 'skipped' and the exact reason, so
    the user can see what to fix.
    """
    valid: List[Dict[str, str]] = []
    skipped: List[BatchRowResult] = []
    for entry in raw_rows:
        reason = validate_row(entry)
        if reason is None:
            valid.append(entry)
        else:
            skipped.append(
                BatchRowResult(
                    row_number=int(entry["row_number"]),
                    row_id=(entry.get("id") or "").strip() or None,
                    question=(entry.get("question") or "").strip(),
                    ai_response=(entry.get("ai_response") or "").strip(),
                    reference_answer=(entry.get("reference_answer") or "").strip() or None,
                    source_information=(entry.get("source_information") or "").strip() or None,
                    status="skipped",
                    error=reason,
                )
            )
    return valid, skipped
def evaluate_batch(
    raw,
    progress_callback: Optional[Callable[[int, int, "BatchRowResult"], None]] = None,
    evaluate_fn: Optional[Callable] = None,
    top_k: int = orchestrator.DEFAULT_TOP_K,
) -> BatchEvaluationReport:
    """
    Run the full batch: validate the file, skip invalid rows, evaluate every
    valid row through the existing Orchestrator, and keep each row's
    complete result.
    Parameters
    ----------
    raw : bytes or str
        The uploaded CSV content.
    progress_callback : callable, optional
        Called as progress_callback(completed, total, row_result) after each
        evaluated row, where total is the number of valid rows. This is what
        drives the real progress indicator in the UI.
    evaluate_fn : callable, optional
        Injection point for tests; defaults to
        orchestrator.evaluate_response.
    top_k : int
        Passed straight through to the Orchestrator's RAG retrieval.
    Raises
    ------
    BatchValidationError
        Only for file-level/column-level problems, i.e. when there is
        nothing sensible to evaluate at all. Individual bad rows never
        raise - they come back as skipped rows.
    """
    evaluate = evaluate_fn or orchestrator.evaluate_response
    raw_rows, ignored_columns = parse_csv(raw)
    valid_rows, skipped_results = validate_rows(raw_rows)
    total = len(valid_rows)
    results: List[BatchRowResult] = []
    for position, entry in enumerate(valid_rows, start=1):
        question = (entry.get("question") or "").strip()
        response = (entry.get("ai_response") or "").strip()
        reference = (entry.get("reference_answer") or "").strip() or None
        source_information = (entry.get("source_information") or "").strip() or None
        row_result = BatchRowResult(
            row_number=int(entry["row_number"]),
            row_id=(entry.get("id") or "").strip() or None,
            question=question,
            ai_response=response,
            reference_answer=reference,
            source_information=source_information,
            status="evaluated",
        )
        try:
            extra = {"source_information": source_information} if source_information else {}
            row_result.evaluation = evaluate(
                question=question,
                ai_response=response,
                reference_answer=reference,
                top_k=top_k,
                **extra,
            )
        except Exception as exc:
            logger.warning("Row %s failed to evaluate (%s); continuing.", row_result.row_number, exc)
            row_result.status = "failed"
            row_result.error = f"{type(exc).__name__}: {exc}"
        results.append(row_result)
        if progress_callback is not None:
            try:
                progress_callback(position, total, row_result)
            except Exception as exc:
                logger.warning("Progress callback failed (%s); continuing.", exc)
    return BatchEvaluationReport(
        rows=results + skipped_results,
        total_rows_in_file=len(raw_rows),
        ignored_columns=ignored_columns,
    )
EXPORT_COLUMNS = (
    "row_number",
    "id",
    "status",
    "question",
    "ai_response",
    "reference_answer",
    "source_information",
    "relevance_score",
    "accuracy_score",
    "completeness_score",
    "hallucination_status",
    "weighted_overall_score",
    "verdict",
    "error",
)
def results_to_csv(report: BatchEvaluationReport) -> str:
    """Flatten a finished report into a downloadable CSV (one row per input row)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(EXPORT_COLUMNS)
    for row in report.ordered_rows():
        evaluation = row.evaluation
        verdict = getattr(evaluation, "verdict", None) if evaluation else None
        writer.writerow(
            [
                row.row_number,
                row.row_id or "",
                row.status,
                row.question,
                row.ai_response,
                row.reference_answer or "",
                row.source_information or "",
                evaluation.relevance.score if evaluation else "",
                evaluation.accuracy.score if evaluation else "",
                evaluation.completeness.score if evaluation and evaluation.completeness else "",
                evaluation.hallucination.hallucination_status if evaluation else "",
                verdict.weighted_overall_score if verdict else "",
                verdict.verdict if verdict else "",
                row.error or "",
            ]
        )
    return buffer.getvalue()
