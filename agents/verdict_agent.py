
from typing import Dict, List
from agents.schemas import AccuracyOutput, CompletenessOutput, HallucinationOutput, RelevanceOutput, VerdictOutput
WEIGHTS: Dict[str, float] = {
    "accuracy": 0.35,
    "hallucination": 0.30,
    "completeness": 0.20,
    "relevance": 0.15,
}
assert round(sum(WEIGHTS.values()), 4) == 1.0, "Verdict Agent weights must total exactly 100%."
PASS_THRESHOLD = 75
NEEDS_IMPROVEMENT_THRESHOLD = 50
_HALLUCINATION_STATUS_FALLBACK = {
    "No hallucination": 100.0,
    "Partially hallucinated": 50.0,
    "Hallucinated": 0.0,
}
def _normalize_1_to_5(score: int) -> float:
    """Map a 1-5 judge score onto a common 0-100 scale (1->0, 3->50, 5->100)."""
    return round((score - 1) / 4 * 100, 2)
UNSUPPORTED_CLAIM_CREDIT = 0.5
def _normalize_hallucination(hallucination: HallucinationOutput) -> float:
    """
    Map the Hallucination Agent's result onto the same 0-100 scale.
    Supported claims score full credit, Unsupported claims half credit and
    Contradicted claims none. Prefers this granular per-claim figure when
    claims were actually checked, since that is more informative than the
    three-way status alone; falls back to the fixed status mapping only
    when there is no per-claim data (for example, the response had no
    checkable claims at all).
    """
    total = len(hallucination.flagged_claims)
    if total > 0:
        credit = 0.0
        for claim in hallucination.flagged_claims:
            if claim.claim_status == "Supported":
                credit += 1.0
            elif claim.claim_status == "Unsupported":
                credit += UNSUPPORTED_CLAIM_CREDIT
        return round(credit / total * 100, 2)
    return _HALLUCINATION_STATUS_FALLBACK.get(hallucination.hallucination_status, 50.0)
def _contradicted_ratio(hallucination: HallucinationOutput) -> float:
    total = len(hallucination.flagged_claims)
    if total == 0:
        return 0.0
    contradicted = sum(1 for c in hallucination.flagged_claims if c.claim_status == "Contradicted")
    return contradicted / total
def _verdict_from_score(weighted_score: float) -> str:
    if weighted_score >= PASS_THRESHOLD:
        return "Pass"
    if weighted_score >= NEEDS_IMPROVEMENT_THRESHOLD:
        return "Needs Improvement"
    return "Fail"
_VERDICT_SEVERITY = {"Pass": 2, "Needs Improvement": 1, "Fail": 0}
def _downgrade(current: str, new: str) -> str:
    """Return whichever of the two verdicts is more severe (never upgrades)."""
    return new if _VERDICT_SEVERITY[new] < _VERDICT_SEVERITY[current] else current
def _apply_critical_rules(verdict: str, accuracy: AccuracyOutput, hallucination: HallucinationOutput) -> (str, List[str]):
    """
    Apply the critical rules described in the module docstring. Can only
    ever move the verdict to a MORE severe outcome, never a less severe
    one - these are safety rails, not a way to rescue a low score.
    """
    issues: List[str] = []
    if hallucination.hallucination_status == "Hallucinated":
        verdict = _downgrade(verdict, "Fail")
        issues.append(
            "Critical rule triggered: hallucination status is 'Hallucinated' (every checked "
            "claim is contradicted by the evidence), so the verdict cannot be higher than Fail."
        )
    if accuracy.score == 1:
        verdict = _downgrade(verdict, "Fail")
        issues.append(
            "Critical rule triggered: accuracy score is 1 (completely incorrect / contradictory), "
            "so the verdict cannot be higher than Fail."
        )
    contradicted_ratio = _contradicted_ratio(hallucination)
    if contradicted_ratio > 0.5:
        verdict = _downgrade(verdict, "Fail")
        issues.append(
            f"Critical rule triggered: {contradicted_ratio:.0%} of checked claims are directly "
            "contradicted by the evidence (severe contradiction), so the verdict cannot be higher "
            "than Fail."
        )
    if accuracy.score <= 2 and verdict == "Pass":
        verdict = _downgrade(verdict, "Needs Improvement")
        issues.append(
            f"Critical rule triggered: accuracy score is {accuracy.score} (mostly incorrect or "
            "worse), so the verdict cannot be higher than Needs Improvement even though the "
            "weighted overall score alone would have passed."
        )
    return verdict, issues
def compute_verdict(
    relevance: RelevanceOutput,
    accuracy: AccuracyOutput,
    hallucination: HallucinationOutput,
    completeness: CompletenessOutput,
) -> VerdictOutput:
    """
    Public entry point used by the Evaluation Orchestrator.
    Combines the four judges' own results into one weighted overall score
    and a final Pass / Needs Improvement / Fail verdict, applying the
    critical rules on top. Every individual judge result is preserved
    unchanged elsewhere on the EvaluationResult; this function only reads
    them, it never mutates them.
    """
    normalized_scores = {
        "relevance": _normalize_1_to_5(relevance.score),
        "accuracy": _normalize_1_to_5(accuracy.score),
        "completeness": _normalize_1_to_5(completeness.score),
        "hallucination": _normalize_hallucination(hallucination),
    }
    weighted_overall_score = sum(WEIGHTS[dim] * normalized_scores[dim] for dim in WEIGHTS)
    weighted_overall_score = round(weighted_overall_score, 2)
    base_verdict = _verdict_from_score(weighted_overall_score)
    final_verdict, critical_issues = _apply_critical_rules(base_verdict, accuracy, hallucination)
    major_issues: List[str] = []
    if relevance.score <= 2:
        major_issues.append(f"Low relevance ({relevance.score}/5): {relevance.label}.")
    if accuracy.score <= 2:
        major_issues.append(f"Low accuracy ({accuracy.score}/5): {accuracy.label}.")
    if hallucination.hallucination_status != "No hallucination":
        major_issues.append(
            f"Hallucination status is '{hallucination.hallucination_status}' "
            f"({len(hallucination.problem_claims)} of {len(hallucination.flagged_claims)} claims flagged)."
        )
    if completeness.score <= 2:
        major_issues.append(f"Low completeness ({completeness.score}/5): {completeness.label}.")
    if completeness.missing_aspects:
        major_issues.append(f"Missing from the response: {'; '.join(completeness.missing_aspects)}.")
    major_issues.extend(critical_issues)
    reasoning = (
        f"Weighted overall score {weighted_overall_score:.2f}/100 "
        f"(accuracy {normalized_scores['accuracy']:.0f}, hallucination {normalized_scores['hallucination']:.0f}, "
        f"completeness {normalized_scores['completeness']:.0f}, relevance {normalized_scores['relevance']:.0f}; "
        f"weights {int(WEIGHTS['accuracy']*100)}/{int(WEIGHTS['hallucination']*100)}/"
        f"{int(WEIGHTS['completeness']*100)}/{int(WEIGHTS['relevance']*100)}) "
        f"maps to a base verdict of '{base_verdict}'."
    )
    if critical_issues:
        reasoning += f" Final verdict adjusted to '{final_verdict}' by {len(critical_issues)} critical rule(s)."
    else:
        reasoning += f" Final verdict: '{final_verdict}'."
    if major_issues:
        reasoning += " Consolidated issues: " + " ".join(major_issues)
    else:
        reasoning += " No major issues identified across any of the four judges."
    return VerdictOutput(
        relevance_score=relevance.score,
        accuracy_score=accuracy.score,
        completeness_score=completeness.score,
        hallucination_status=hallucination.hallucination_status,
        normalized_scores=normalized_scores,
        weights=dict(WEIGHTS),
        weighted_overall_score=weighted_overall_score,
        verdict=final_verdict,
        major_issues=major_issues,
        reasoning=reasoning,
    )
