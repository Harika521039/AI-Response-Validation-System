
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
VALID_SCORES = (1, 2, 3, 4, 5)
CLAIM_STATUSES = ("Supported", "Unsupported", "Contradicted")
HALLUCINATION_STATUSES = ("No hallucination", "Partially hallucinated", "Hallucinated")
EVALUATION_MODES = ("llm", "heuristic")
REQUIREMENT_STATUSES = ("Addressed", "Partially Addressed", "Missing")
VERDICTS = ("Pass", "Needs Improvement", "Fail")
class SchemaValidationError(ValueError):
    """Raised when an agent tries to build a result with an invalid value."""
def _require_score(value: Any, field_name: str) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(f"{field_name} must be an integer 1-5, got {value!r}.") from exc
    if score not in VALID_SCORES:
        raise SchemaValidationError(f"{field_name} must be one of {VALID_SCORES}, got {score}.")
    return score
def _require_choice(value: Any, allowed: tuple, field_name: str) -> str:
    if value not in allowed:
        raise SchemaValidationError(f"{field_name} must be one of {allowed}, got {value!r}.")
    return value
def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{field_name} must be a non-empty string.")
    return value
class _DumpMixin:
    """Gives every model a model_dump() returning a plain, JSON-safe dict."""
    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)
@dataclass
class RelevanceInput(_DumpMixin):
    question: str
    response: str
    def __post_init__(self):
        _require_text(self.question, "question")
        _require_text(self.response, "response")
@dataclass
class RelevanceOutput(_DumpMixin):
    score: int
    label: str
    reasoning: str
    mode: str = "heuristic"
    def __post_init__(self):
        self.score = _require_score(self.score, "relevance score")
        _require_text(self.label, "relevance label")
        _require_text(self.reasoning, "relevance reasoning")
        _require_choice(self.mode, EVALUATION_MODES, "mode")
@dataclass
class AccuracyInput(_DumpMixin):
    question: str
    response: str
    reference_answer: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    def __post_init__(self):
        _require_text(self.question, "question")
        _require_text(self.response, "response")
        if self.evidence is None:
            self.evidence = []
@dataclass
class AccuracyOutput(_DumpMixin):
    score: int
    label: str
    reasoning: str
    supporting_evidence: List[str] = field(default_factory=list)
    used_reference_answer: bool = False
    evidence_mode: str = "none"
    mode: str = "heuristic"
    def __post_init__(self):
        self.score = _require_score(self.score, "accuracy score")
        _require_text(self.label, "accuracy label")
        _require_text(self.reasoning, "accuracy reasoning")
        _require_choice(self.mode, EVALUATION_MODES, "mode")
        if self.supporting_evidence is None:
            self.supporting_evidence = []
@dataclass
class FlaggedClaim(_DumpMixin):
    claim: str
    claim_status: str
    reasoning: str
    evidence: Optional[str] = None
    def __post_init__(self):
        _require_text(self.claim, "claim")
        _require_choice(self.claim_status, CLAIM_STATUSES, "claim_status")
        _require_text(self.reasoning, "claim reasoning")
@dataclass
class HallucinationInput(_DumpMixin):
    response: str
    question: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    def __post_init__(self):
        _require_text(self.response, "response")
        if self.evidence is None:
            self.evidence = []
@dataclass
class HallucinationOutput(_DumpMixin):
    hallucination_status: str
    reasoning: str
    flagged_claims: List[FlaggedClaim] = field(default_factory=list)
    mode: str = "heuristic"
    def __post_init__(self):
        _require_choice(self.hallucination_status, HALLUCINATION_STATUSES, "hallucination_status")
        _require_text(self.reasoning, "hallucination reasoning")
        _require_choice(self.mode, EVALUATION_MODES, "mode")
        if self.flagged_claims is None:
            self.flagged_claims = []
    @property
    def problem_claims(self) -> List[FlaggedClaim]:
        """Only the claims that are NOT supported - handy for the UI."""
        return [c for c in self.flagged_claims if c.claim_status != "Supported"]
@dataclass
class RequirementAssessment(_DumpMixin):
    """One identified requirement / sub-question of the original question,
    together with how well the response addressed it."""
    requirement: str
    status: str
    reasoning: str
    def __post_init__(self):
        _require_text(self.requirement, "requirement")
        _require_choice(self.status, REQUIREMENT_STATUSES, "requirement status")
        _require_text(self.reasoning, "requirement reasoning")
@dataclass
class CompletenessInput(_DumpMixin):
    question: str
    response: str
    reference_answer: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    def __post_init__(self):
        _require_text(self.question, "question")
        _require_text(self.response, "response")
        if self.evidence is None:
            self.evidence = []
@dataclass
class CompletenessOutput(_DumpMixin):
    score: int
    label: str
    reasoning: str
    requirement_assessments: List[RequirementAssessment] = field(default_factory=list)
    addressed_aspects: List[str] = field(default_factory=list)
    partial_aspects: List[str] = field(default_factory=list)
    missing_aspects: List[str] = field(default_factory=list)
    evidence_mode: str = "none"
    mode: str = "heuristic"
    def __post_init__(self):
        self.score = _require_score(self.score, "completeness score")
        _require_text(self.label, "completeness label")
        _require_text(self.reasoning, "completeness reasoning")
        _require_choice(self.mode, EVALUATION_MODES, "mode")
        for lst_name in ("requirement_assessments", "addressed_aspects", "partial_aspects", "missing_aspects"):
            if getattr(self, lst_name) is None:
                setattr(self, lst_name, [])
@dataclass
class VerdictOutput(_DumpMixin):
    relevance_score: int
    accuracy_score: int
    completeness_score: int
    hallucination_status: str
    normalized_scores: Dict[str, float]
    weights: Dict[str, float]
    weighted_overall_score: float
    verdict: str
    major_issues: List[str]
    reasoning: str
    def __post_init__(self):
        _require_choice(self.verdict, VERDICTS, "verdict")
        _require_text(self.reasoning, "verdict reasoning")
        self.weighted_overall_score = round(float(self.weighted_overall_score), 2)
        if self.major_issues is None:
            self.major_issues = []
        if not isinstance(self.normalized_scores, dict) or not self.normalized_scores:
            raise SchemaValidationError("normalized_scores must be a non-empty dict.")
        if not isinstance(self.weights, dict) or not self.weights:
            raise SchemaValidationError("weights must be a non-empty dict.")
        total_weight = round(sum(self.weights.values()), 4)
        if total_weight != 1.0:
            raise SchemaValidationError(f"weights must total exactly 1.0 (100%), got {total_weight}.")
@dataclass
class EvidenceItem(_DumpMixin):
    text: str
    source: str = "unknown"
    question: str = ""
    distance: float = 0.0
    def __post_init__(self):
        _require_text(self.text, "evidence text")
        self.distance = float(self.distance)
@dataclass
class EvaluationResult(_DumpMixin):
    question: str
    response: str
    relevance: RelevanceOutput
    accuracy: AccuracyOutput
    hallucination: HallucinationOutput
    completeness: Optional[CompletenessOutput] = None
    verdict: Optional[VerdictOutput] = None
    reference_answer: Optional[str] = None
    source_information: Optional[str] = None
    retrieved_evidence: List[EvidenceItem] = field(default_factory=list)
    def summary(self) -> Dict[str, Any]:
        """
        The "FINAL EVALUATION SUMMARY" shown in the UI and the report.
        Milestone 2 behaviour (unchanged): restates the three original
        agents' own results, with no invented combined score.
        Milestone 3 addition: when a Completeness Judge result and/or a
        Verdict Agent result are attached, their own fields are folded in
        too - still nothing computed here that the responsible agent did
        not itself produce.
        """
        result = {
            "relevance": f"{self.relevance.score}/5 ({self.relevance.label})",
            "accuracy": f"{self.accuracy.score}/5 ({self.accuracy.label})",
            "hallucination": self.hallucination.hallucination_status,
            "accuracy_evidence_mode": self.accuracy.evidence_mode,
            "unsupported_or_contradicted_claims": len(self.hallucination.problem_claims),
            "total_claims_checked": len(self.hallucination.flagged_claims),
            "source_information_provided": bool(
                self.source_information and self.source_information.strip()
            ),
        }
        if self.completeness is not None:
            result["completeness"] = f"{self.completeness.score}/5 ({self.completeness.label})"
            result["completeness_evidence_mode"] = self.completeness.evidence_mode
            result["missing_aspects"] = list(self.completeness.missing_aspects)
        if self.verdict is not None:
            result["weighted_overall_score"] = self.verdict.weighted_overall_score
            result["verdict"] = self.verdict.verdict
            result["major_issues"] = list(self.verdict.major_issues)
        return result
BATCH_ROW_STATUSES = ("evaluated", "skipped", "failed")
class BatchValidationError(ValueError):
    """
    Raised for file-level or column-level problems with an uploaded batch
    file, i.e. problems that make the whole upload unusable (not decodable,
    empty, no header, a required column missing, too many rows).
    An individual bad row never raises: it is skipped and reported, so the
    rest of the batch still runs.
    """
@dataclass
class BatchRowResult(_DumpMixin):
    """
    One input row of the batch.
    status
        "evaluated" - the row passed validation and the Orchestrator ran;
                      `evaluation` holds that row's COMPLETE
                      EvaluationResult (all four judges plus the verdict),
                      not a summary line.
        "skipped"   - the row failed row-level validation and was never
                      evaluated; `error` says exactly why.
        "failed"    - the row was valid but the evaluation itself raised
                      (for example an LLM/API error); `error` holds the
                      message and the batch continued with the next row.
    """
    row_number: int
    question: str
    ai_response: str
    status: str = "evaluated"
    row_id: Optional[str] = None
    reference_answer: Optional[str] = None
    source_information: Optional[str] = None
    evaluation: Optional[EvaluationResult] = None
    error: Optional[str] = None
    def __post_init__(self):
        _require_choice(self.status, BATCH_ROW_STATUSES, "batch row status")
        self.row_number = int(self.row_number)
        if self.status == "evaluated" and self.error:
            raise SchemaValidationError("An evaluated batch row cannot carry an error message.")
        if self.status in ("skipped", "failed"):
            _require_text(self.error, "batch row error")
    @property
    def summary(self) -> Optional[Dict[str, Any]]:
        """This row's own evaluation summary, or None if it never ran."""
        return self.evaluation.summary() if self.evaluation is not None else None
    @property
    def display_id(self) -> str:
        """The label used for this row in the results table."""
        return self.row_id or f"Row {self.row_number}"
    def table_row(self) -> Dict[str, Any]:
        """
        One line of the M3.4 batch results table:
        ID, Relevance, Accuracy, Hallucination, Completeness, Overall, Verdict.
        Every value is read off this row's own preserved EvaluationResult;
        nothing is recomputed. A skipped/failed row keeps its place in the
        table with "-" cells and its reason in the Status column.
        """
        line: Dict[str, Any] = {
            "ID": self.display_id,
            "Relevance": "-",
            "Accuracy": "-",
            "Hallucination": "-",
            "Completeness": "-",
            "Overall": "-",
            "Verdict": "-",
            "Status": self.status if self.status == "evaluated" else f"{self.status}: {self.error}",
        }
        if self.evaluation is None:
            return line
        line["Relevance"] = self.evaluation.relevance.score
        line["Accuracy"] = self.evaluation.accuracy.score
        line["Hallucination"] = self.evaluation.hallucination.hallucination_status
        if self.evaluation.completeness is not None:
            line["Completeness"] = self.evaluation.completeness.score
        if self.evaluation.verdict is not None:
            line["Overall"] = self.evaluation.verdict.weighted_overall_score
            line["Verdict"] = self.evaluation.verdict.verdict
        return line
@dataclass
class BatchEvaluationReport(_DumpMixin):
    """The finished batch: every input row, in file order, with its result."""
    rows: List[BatchRowResult] = field(default_factory=list)
    total_rows_in_file: int = 0
    ignored_columns: List[str] = field(default_factory=list)
    def __post_init__(self):
        if self.rows is None:
            self.rows = []
        if self.ignored_columns is None:
            self.ignored_columns = []
    def ordered_rows(self) -> List["BatchRowResult"]:
        return sorted(self.rows, key=lambda row: row.row_number)
    def rows_with_status(self, status: str) -> List["BatchRowResult"]:
        return [row for row in self.ordered_rows() if row.status == status]
    @property
    def evaluated_rows(self) -> List["BatchRowResult"]:
        return self.rows_with_status("evaluated")
    @property
    def skipped_rows(self) -> List["BatchRowResult"]:
        return self.rows_with_status("skipped")
    @property
    def failed_rows(self) -> List["BatchRowResult"]:
        return self.rows_with_status("failed")
    def results_table(self, include_unevaluated: bool = True) -> List[Dict[str, Any]]:
        """
        The M3.4 batch results table, in file order: one line per row with
        ID, Relevance, Accuracy, Hallucination, Completeness, Overall and
        Verdict (plus a Status column carrying any skip/failure reason).
        """
        rows = self.ordered_rows() if include_unevaluated else self.evaluated_rows
        return [row.table_row() for row in rows]
    def aggregate(self) -> Dict[str, Any]:
        """
        Batch-level counts and averages.
        Every number here is read straight off the per-row agent results -
        no new scoring logic, in keeping with the rest of the project.
        """
        evaluated = [row for row in self.evaluated_rows if row.evaluation is not None]
        stats: Dict[str, Any] = {
            "total_rows_in_file": self.total_rows_in_file,
            "evaluated": len(evaluated),
            "skipped": len(self.skipped_rows),
            "failed": len(self.failed_rows),
        }
        stats["verdict_counts"] = {verdict: 0 for verdict in VERDICTS}
        stats["hallucination_counts"] = {status: 0 for status in HALLUCINATION_STATUSES}
        stats["hallucination_rows"] = 0
        stats["hallucination_frequency_percent"] = None
        if not evaluated:
            return stats
        def _mean(values):
            values = [v for v in values if v is not None]
            return round(sum(values) / len(values), 2) if values else None
        stats["average_relevance"] = _mean([row.evaluation.relevance.score for row in evaluated])
        stats["average_accuracy"] = _mean([row.evaluation.accuracy.score for row in evaluated])
        stats["average_completeness"] = _mean(
            [
                row.evaluation.completeness.score
                for row in evaluated
                if row.evaluation.completeness is not None
            ]
        )
        stats["average_overall_score"] = _mean(
            [
                row.evaluation.verdict.weighted_overall_score
                for row in evaluated
                if row.evaluation.verdict is not None
            ]
        )
        for row in evaluated:
            if row.evaluation.verdict is not None:
                key = row.evaluation.verdict.verdict
                stats["verdict_counts"][key] = stats["verdict_counts"].get(key, 0) + 1
        for row in evaluated:
            key = row.evaluation.hallucination.hallucination_status
            stats["hallucination_counts"][key] = stats["hallucination_counts"].get(key, 0) + 1
        flagged = sum(
            count
            for status, count in stats["hallucination_counts"].items()
            if status != "No hallucination"
        )
        stats["hallucination_rows"] = flagged
        stats["hallucination_frequency_percent"] = round(100.0 * flagged / len(evaluated), 1)
        return stats
