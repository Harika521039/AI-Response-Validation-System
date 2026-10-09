
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from agents.schemas import HALLUCINATION_STATUSES, VERDICTS
DIMENSION_KEYS = ("relevance", "accuracy", "completeness", "hallucination")
DIMENSION_LABELS = {
    "relevance": "Relevance",
    "accuracy": "Accuracy",
    "completeness": "Completeness",
    "hallucination": "Hallucination",
}
FIVE_POINT_DIMENSIONS = ("relevance", "accuracy", "completeness")
HALLUCINATED_STATUSES = ("Partially hallucinated", "Hallucinated")
PROBLEM_CLAIM_STATUSES = ("Unsupported", "Contradicted")
SOURCE_SINGLE = "single"
SOURCE_BATCH = "batch"
def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    return None if value is None else round(float(value), digits)
def _mean(values: Sequence[float], digits: int = 2) -> Optional[float]:
    usable = [float(v) for v in values if v is not None]
    if not usable:
        return None
    return round(sum(usable) / len(usable), digits)
def _percent(part: int, whole: int, digits: int = 1) -> Optional[float]:
    if not whole:
        return None
    return round(100.0 * part / whole, digits)
def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""
@dataclass
class EvaluationRecord:
   
    record_id: str
    payload: Dict[str, Any]
    created_at: str = ""
    source: str = SOURCE_SINGLE
    batch_id: str = ""
    batch_label: str = ""
    sequence: int = 0
    def __post_init__(self):
        if not isinstance(self.payload, dict):
            raise ValueError("EvaluationRecord.payload must be an EvaluationResult.model_dump() dict.")
        self.record_id = str(self.record_id)
    def _section(self, name: str) -> Dict[str, Any]:
        section = self.payload.get(name)
        return section if isinstance(section, dict) else {}
    @property
    def question(self) -> str:
        return _text(self.payload.get("question"))
    @property
    def response(self) -> str:
        return _text(self.payload.get("response"))
    @property
    def reference_answer(self) -> str:
        return _text(self.payload.get("reference_answer"))
    @property
    def source_information(self) -> str:
        return _text(self.payload.get("source_information"))
    @property
    def relevance(self) -> Dict[str, Any]:
        return self._section("relevance")
    @property
    def accuracy(self) -> Dict[str, Any]:
        return self._section("accuracy")
    @property
    def completeness(self) -> Dict[str, Any]:
        return self._section("completeness")
    @property
    def hallucination(self) -> Dict[str, Any]:
        return self._section("hallucination")
    @property
    def verdict_section(self) -> Dict[str, Any]:
        return self._section("verdict")
    @property
    def relevance_score(self) -> Optional[int]:
        return self.relevance.get("score")
    @property
    def accuracy_score(self) -> Optional[int]:
        return self.accuracy.get("score")
    @property
    def completeness_score(self) -> Optional[int]:
        return self.completeness.get("score")
    @property
    def overall_score(self) -> Optional[float]:
        return self.verdict_section.get("weighted_overall_score")
    @property
    def verdict(self) -> Optional[str]:
        return self.verdict_section.get("verdict")
    @property
    def hallucination_status(self) -> Optional[str]:
        return self.hallucination.get("hallucination_status")
    @property
    def hallucination_normalized(self) -> Optional[float]:
        normalized = self.verdict_section.get("normalized_scores")
        if isinstance(normalized, dict):
            return normalized.get("hallucination")
        return None
    @property
    def normalized_scores(self) -> Dict[str, float]:
        normalized = self.verdict_section.get("normalized_scores")
        return normalized if isinstance(normalized, dict) else {}
    @property
    def is_hallucinated(self) -> bool:
        return self.hallucination_status in HALLUCINATED_STATUSES
    @property
    def flagged_claims(self) -> List[Dict[str, Any]]:
        claims = self.hallucination.get("flagged_claims")
        return [c for c in claims if isinstance(c, dict)] if isinstance(claims, list) else []
    @property
    def problem_claims(self) -> List[Dict[str, Any]]:
        return [c for c in self.flagged_claims if c.get("claim_status") in PROBLEM_CLAIM_STATUSES]
    @property
    def missing_aspects(self) -> List[str]:
        aspects = self.completeness.get("missing_aspects")
        return [a for a in aspects if isinstance(a, str)] if isinstance(aspects, list) else []
    @property
    def partial_aspects(self) -> List[str]:
        aspects = self.completeness.get("partial_aspects")
        return [a for a in aspects if isinstance(a, str)] if isinstance(aspects, list) else []
    @property
    def major_issues(self) -> List[str]:
        issues = self.verdict_section.get("major_issues")
        return [i for i in issues if isinstance(i, str)] if isinstance(issues, list) else []
    @property
    def group_id(self) -> str:
        """The batch this record belongs to, or its own single-evaluation group."""
        return self.batch_id or SOURCE_SINGLE
    @property
    def group_label(self) -> str:
        if self.batch_id:
            return self.batch_label or self.batch_id
        return "Single evaluations"
    def table_row(self) -> Dict[str, Any]:
        """One line of the dashboard drill-down table."""
        return {
            "ID": self.record_id,
            "Source": self.group_label,
            "Question": self.question,
            "Relevance": self.relevance_score,
            "Accuracy": self.accuracy_score,
            "Completeness": self.completeness_score,
            "Hallucination": self.hallucination_status,
            "Overall": self.overall_score,
            "Verdict": self.verdict,
            "Date/time": self.created_at,
        }
def record_from_history_row(row: Dict[str, Any]) -> Optional[EvaluationRecord]:
    """
    Build a record from one backend.database row.
    A row without a decoded evaluation payload is not a completed
    evaluation and returns None instead of a half-empty record.
    """
    payload = row.get("evaluation")
    if not isinstance(payload, dict):
        return None
    batch_id = _text(row.get("batch_id"))
    return EvaluationRecord(
        record_id=str(row.get("id", "")),
        payload=payload,
        created_at=_text(row.get("created_at")),
        source=_text(row.get("source")) or (SOURCE_BATCH if batch_id else SOURCE_SINGLE),
        batch_id=batch_id,
        batch_label=_text(row.get("batch_label")),
        sequence=int(row.get("id") or 0),
    )
def records_from_history(rows: Iterable[Dict[str, Any]]) -> List[EvaluationRecord]:
    records = [record_from_history_row(row) for row in rows]
    return [record for record in records if record is not None]
def records_from_batch_report(
    report,
    batch_id: str = "",
    batch_label: str = "",
    created_at: str = "",
) -> List[EvaluationRecord]:
    """
    Build records straight from an in-memory BatchEvaluationReport, so a
    batch can be analysed before or without being persisted.
    """
    records: List[EvaluationRecord] = []
    for row in report.evaluated_rows:
        if row.evaluation is None:
            continue
        records.append(
            EvaluationRecord(
                record_id=row.display_id,
                payload=row.evaluation.model_dump(),
                created_at=created_at,
                source=SOURCE_BATCH,
                batch_id=batch_id or "current-batch",
                batch_label=batch_label or "Current batch",
                sequence=row.row_number,
            )
        )
    return records
@dataclass
class DashboardFilters:
    """
    The dashboard filter set. Every field is optional; an unset field does
    not restrict anything.
    """
    batch_ids: Optional[Sequence[str]] = None
    sources: Optional[Sequence[str]] = None
    verdicts: Optional[Sequence[str]] = None
    hallucination_statuses: Optional[Sequence[str]] = None
    min_overall_score: Optional[float] = None
    max_overall_score: Optional[float] = None
    search_text: str = ""
    hallucinated_only: bool = False
    def is_active(self) -> bool:
        return any(
            [
                bool(self.batch_ids),
                bool(self.sources),
                bool(self.verdicts),
                bool(self.hallucination_statuses),
                self.min_overall_score is not None,
                self.max_overall_score is not None,
                bool(self.search_text.strip()),
                self.hallucinated_only,
            ]
        )
    def matches(self, record: EvaluationRecord) -> bool:
        if self.batch_ids and record.group_id not in set(self.batch_ids):
            return False
        if self.sources and record.source not in set(self.sources):
            return False
        if self.verdicts and record.verdict not in set(self.verdicts):
            return False
        if self.hallucination_statuses and record.hallucination_status not in set(self.hallucination_statuses):
            return False
        if self.hallucinated_only and not record.is_hallucinated:
            return False
        score = record.overall_score
        if self.min_overall_score is not None:
            if score is None or float(score) < float(self.min_overall_score):
                return False
        if self.max_overall_score is not None:
            if score is None or float(score) > float(self.max_overall_score):
                return False
        needle = self.search_text.strip().lower()
        if needle:
            haystack = " ".join([record.record_id, record.question, record.response]).lower()
            if needle not in haystack:
                return False
        return True
def apply_filters(
    records: Sequence[EvaluationRecord],
    filters: Optional[DashboardFilters] = None,
) -> List[EvaluationRecord]:
    if filters is None:
        return list(records)
    return [record for record in records if filters.matches(record)]
def filter_options(records: Sequence[EvaluationRecord]) -> Dict[str, List[Any]]:
    """The choices actually present in the data, so no filter is ever empty."""
    groups: List[Tuple[str, str]] = []
    seen = set()
    for record in records:
        if record.group_id not in seen:
            seen.add(record.group_id)
            groups.append((record.group_id, record.group_label))
    return {
        "groups": groups,
        "sources": sorted({record.source for record in records if record.source}),
        "verdicts": [v for v in VERDICTS if any(r.verdict == v for r in records)],
        "hallucination_statuses": [
            s for s in HALLUCINATION_STATUSES if any(r.hallucination_status == s for r in records)
        ],
    }
def verdict_counts(records: Sequence[EvaluationRecord]) -> Dict[str, int]:
    counts = {verdict: 0 for verdict in VERDICTS}
    for record in records:
        verdict = record.verdict
        if verdict in counts:
            counts[verdict] += 1
    return counts
def verdict_percentages(records: Sequence[EvaluationRecord]) -> Dict[str, Optional[float]]:
    counts = verdict_counts(records)
    total = sum(counts.values())
    return {verdict: _percent(count, total) for verdict, count in counts.items()}
def hallucination_counts(records: Sequence[EvaluationRecord]) -> Dict[str, int]:
    counts = {status: 0 for status in HALLUCINATION_STATUSES}
    for record in records:
        status = record.hallucination_status
        if status in counts:
            counts[status] += 1
    return counts
def average_dimension_scores(records: Sequence[EvaluationRecord]) -> Dict[str, Optional[float]]:
    """Average 1-5 score per judged dimension, plus the 0-100 hallucination score."""
    return {
        "relevance": _mean([r.relevance_score for r in records]),
        "accuracy": _mean([r.accuracy_score for r in records]),
        "completeness": _mean([r.completeness_score for r in records]),
        "hallucination": _mean([r.hallucination_normalized for r in records]),
    }
def average_normalized_scores(records: Sequence[EvaluationRecord]) -> Dict[str, Optional[float]]:
    """Every dimension on the same 0-100 scale the Verdict Agent uses."""
    averages: Dict[str, Optional[float]] = {}
    for dimension in DIMENSION_KEYS:
        averages[dimension] = _mean(
            [r.normalized_scores.get(dimension) for r in records if dimension in r.normalized_scores]
        )
    return averages
def claim_statistics(records: Sequence[EvaluationRecord]) -> Dict[str, Any]:
    checked = 0
    unsupported = 0
    contradicted = 0
    for record in records:
        for claim in record.flagged_claims:
            checked += 1
            if claim.get("claim_status") == "Unsupported":
                unsupported += 1
            elif claim.get("claim_status") == "Contradicted":
                contradicted += 1
    problem = unsupported + contradicted
    return {
        "claims_checked": checked,
        "unsupported_claims": unsupported,
        "contradicted_claims": contradicted,
        "problem_claims": problem,
        "problem_claim_percent": _percent(problem, checked),
    }
def hallucination_statistics(records: Sequence[EvaluationRecord]) -> Dict[str, Any]:
    counts = hallucination_counts(records)
    judged = sum(counts.values())
    flagged = sum(counts[status] for status in HALLUCINATED_STATUSES)
    stats: Dict[str, Any] = {
        "status_counts": counts,
        "responses_judged": judged,
        "hallucinated_responses": flagged,
        "hallucination_frequency_percent": _percent(flagged, judged),
        "average_hallucination_score": _mean([r.hallucination_normalized for r in records]),
    }
    stats.update(claim_statistics(records))
    return stats
def dimension_score_chart(records: Sequence[EvaluationRecord]) -> List[Dict[str, Any]]:
    """Average score per dimension on the common 0-100 scale, for a bar chart."""
    averages = average_normalized_scores(records)
    series = []
    for dimension in DIMENSION_KEYS:
        value = averages.get(dimension)
        if value is None:
            continue
        series.append({"Dimension": DIMENSION_LABELS[dimension], "Average score (0-100)": value})
    return series
def five_point_dimension_chart(records: Sequence[EvaluationRecord]) -> List[Dict[str, Any]]:
    """Average raw judge score per 1-5 dimension, for a bar chart."""
    averages = average_dimension_scores(records)
    series = []
    for dimension in FIVE_POINT_DIMENSIONS:
        value = averages.get(dimension)
        if value is None:
            continue
        series.append({"Dimension": DIMENSION_LABELS[dimension], "Average score (1-5)": value})
    return series
def verdict_distribution_chart(records: Sequence[EvaluationRecord]) -> List[Dict[str, Any]]:
    counts = verdict_counts(records)
    percentages = verdict_percentages(records)
    return [
        {"Verdict": verdict, "Count": counts[verdict], "Percent": percentages[verdict]}
        for verdict in VERDICTS
    ]
def score_distribution_chart(records: Sequence[EvaluationRecord]) -> List[Dict[str, Any]]:
    """How the weighted overall scores spread across 0-100 in ten-point bands."""
    bands = [(low, low + 10) for low in range(0, 100, 10)]
    counts = {low: 0 for low, _ in bands}
    for record in records:
        score = record.overall_score
        if score is None:
            continue
        value = max(0.0, min(100.0, float(score)))
        index = min(int(value // 10), 9)
        counts[index * 10] += 1
    return [
        {"Score band": f"{low}-{high if high < 100 else 100}", "Count": counts[low]}
        for low, high in bands
    ]
def group_records(records: Sequence[EvaluationRecord]) -> List[Tuple[str, str, List[EvaluationRecord]]]:
    """Records bucketed by batch, in first-seen order, singles kept together."""
    order: List[str] = []
    buckets: Dict[str, List[EvaluationRecord]] = {}
    labels: Dict[str, str] = {}
    for record in sorted(records, key=lambda r: (r.created_at, r.sequence)):
        key = record.group_id
        if key not in buckets:
            buckets[key] = []
            labels[key] = record.group_label
            order.append(key)
        buckets[key].append(record)
    return [(key, labels[key], buckets[key]) for key in order]
def batch_statistics(records: Sequence[EvaluationRecord]) -> List[Dict[str, Any]]:
    """Per-batch statistics, each computed from that batch's own records."""
    rows = []
    for group_id, label, group in group_records(records):
        counts = verdict_counts(group)
        percentages = verdict_percentages(group)
        averages = average_dimension_scores(group)
        halluc = hallucination_statistics(group)
        rows.append(
            {
                "batch_id": group_id,
                "batch_label": label,
                "total_responses": len(group),
                "average_overall_score": _mean([r.overall_score for r in group]),
                "average_relevance": averages["relevance"],
                "average_accuracy": averages["accuracy"],
                "average_completeness": averages["completeness"],
                "average_hallucination_score": averages["hallucination"],
                "verdict_counts": counts,
                "verdict_percentages": percentages,
                "pass_rate_percent": percentages.get("Pass"),
                "hallucinated_responses": halluc["hallucinated_responses"],
                "hallucination_frequency_percent": halluc["hallucination_frequency_percent"],
                "first_evaluated_at": group[0].created_at,
                "last_evaluated_at": group[-1].created_at,
            }
        )
    return rows
def quality_trend(records: Sequence[EvaluationRecord]) -> List[Dict[str, Any]]:
    """Quality across batches, in evaluation order, for a trend chart."""
    trend = []
    for stats in batch_statistics(records):
        trend.append(
            {
                "Batch": stats["batch_label"],
                "Responses": stats["total_responses"],
                "Average overall score": stats["average_overall_score"],
                "Pass rate (%)": stats["pass_rate_percent"],
                "Hallucination rate (%)": stats["hallucination_frequency_percent"],
            }
        )
    return trend
def frequent_issues(records: Sequence[EvaluationRecord], limit: int = 10) -> List[Dict[str, Any]]:
    """
    The evaluation issues that came up most often, counted across records.
    Three real sources, all produced by the agents themselves:
    the Verdict Agent's major_issues, the Completeness Judge's
    missing_aspects, and the Hallucination Detector's problem claims.
    """
    counts: Dict[Tuple[str, str], int] = {}
    examples: Dict[Tuple[str, str], str] = {}
    def add(category: str, issue: str, record: EvaluationRecord) -> None:
        text = (issue or "").strip()
        if not text:
            return
        key = (category, text)
        counts[key] = counts.get(key, 0) + 1
        examples.setdefault(key, record.record_id)
    for record in records:
        for issue in record.major_issues:
            add("Verdict issue", issue, record)
        for aspect in record.missing_aspects:
            add("Missing aspect", aspect, record)
        for claim in record.problem_claims:
            status = claim.get("claim_status", "Unsupported")
            add(f"{status} claim", _text(claim.get("claim")), record)
    total = len(records)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    issues = []
    for (category, text), count in ranked[:limit]:
        issues.append(
            {
                "category": category,
                "issue": text,
                "count": count,
                "affected_percent": _percent(count, total),
                "example_record_id": examples[(category, text)],
            }
        )
    return issues
def compute_statistics(records: Sequence[EvaluationRecord]) -> Dict[str, Any]:
    """
    The complete M4.1 statistics block for the supplied records.
    Callers pass already-filtered records, so the same function serves the
    combined view, a single batch view and any filtered subset.
    """
    records = list(records)
    total = len(records)
    counts = verdict_counts(records)
    percentages = verdict_percentages(records)
    averages = average_dimension_scores(records)
    halluc = hallucination_statistics(records)
    verdicts_present = sum(counts.values())
    stats: Dict[str, Any] = {
        "total_responses": total,
        "responses_with_verdict": verdicts_present,
        "verdict_counts": counts,
        "verdict_percentages": percentages,
        "pass_count": counts["Pass"],
        "needs_improvement_count": counts["Needs Improvement"],
        "fail_count": counts["Fail"],
        "pass_rate_percent": percentages["Pass"],
        "needs_improvement_rate_percent": percentages["Needs Improvement"],
        "fail_rate_percent": percentages["Fail"],
        "average_relevance": averages["relevance"],
        "average_accuracy": averages["accuracy"],
        "average_completeness": averages["completeness"],
        "average_hallucination_score": averages["hallucination"],
        "average_overall_score": _mean([r.overall_score for r in records]),
        "average_normalized_scores": average_normalized_scores(records),
        "hallucination": halluc,
        "hallucinated_responses": halluc["hallucinated_responses"],
        "hallucination_frequency_percent": halluc["hallucination_frequency_percent"],
        "dimension_chart": dimension_score_chart(records),
        "five_point_chart": five_point_dimension_chart(records),
        "verdict_distribution": verdict_distribution_chart(records),
        "score_distribution": score_distribution_chart(records),
        "batch_statistics": batch_statistics(records),
        "quality_trend": quality_trend(records),
        "frequent_issues": frequent_issues(records),
        "batches_covered": len({r.group_id for r in records}),
    }
    scores = [r.overall_score for r in records if r.overall_score is not None]
    stats["highest_overall_score"] = _round(max(scores)) if scores else None
    stats["lowest_overall_score"] = _round(min(scores)) if scores else None
    return stats
def results_table(records: Sequence[EvaluationRecord]) -> List[Dict[str, Any]]:
    return [record.table_row() for record in records]
def find_record(records: Sequence[EvaluationRecord], record_id: str) -> Optional[EvaluationRecord]:
    """Drill-down lookup: the one record behind a table line."""
    for record in records:
        if record.record_id == str(record_id):
            return record
    return None
def improvement_recommendations(records: Sequence[EvaluationRecord]) -> List[str]:
    """
    Recommendations derived from the supplied records only.
    Each entry states the measured figure it came from, so the reader can
    check it against the same records. An empty list means the data gave
    no reason to recommend anything.
    """
    records = list(records)
    if not records:
        return []
    stats = compute_statistics(records)
    total = stats["total_responses"]
    recommendations: List[str] = []
    fail_rate = stats["fail_rate_percent"]
    if fail_rate is not None and fail_rate > 0:
        recommendations.append(
            f"{stats['fail_count']} of {total} responses ({fail_rate}%) were graded Fail. "
            "Review those responses first: they carry the critical accuracy or hallucination rules."
        )
    needs = stats["needs_improvement_rate_percent"]
    if needs is not None and needs > 0:
        recommendations.append(
            f"{stats['needs_improvement_count']} of {total} responses ({needs}%) need improvement, "
            "so they cleared the failure rules but stayed below the 75-point pass threshold."
        )
    frequency = stats["hallucination_frequency_percent"]
    if frequency:
        recommendations.append(
            f"{stats['hallucinated_responses']} of {total} responses ({frequency}%) were flagged as "
            "partially or fully hallucinated. Ground those answers in the supplied evidence before reuse."
        )
    claims = stats["hallucination"]
    if claims["contradicted_claims"]:
        recommendations.append(
            f"{claims['contradicted_claims']} claims directly contradicted the evidence. "
            "Contradictions are the most severe class of error and should be corrected first."
        )
    if claims["unsupported_claims"]:
        recommendations.append(
            f"{claims['unsupported_claims']} claims could not be verified against any evidence. "
            "Either supply supporting sources or remove the unverifiable statements."
        )
    for dimension, label, threshold in (
        ("relevance", "Relevance", 4.0),
        ("accuracy", "Accuracy", 4.0),
        ("completeness", "Completeness", 4.0),
    ):
        value = stats.get(f"average_{dimension}")
        if value is not None and value < threshold:
            recommendations.append(
                f"Average {label} is {value}/5, below the {threshold}/5 target. "
                f"{label} is the weakest lever available on this set of responses."
            )
    missing = [issue for issue in stats["frequent_issues"] if issue["category"] == "Missing aspect"]
    if missing:
        top = missing[0]
        recommendations.append(
            f"The most frequently missing aspect was \"{top['issue']}\" "
            f"({top['count']} of {total} responses). Cover it explicitly in future answers."
        )
    return recommendations
