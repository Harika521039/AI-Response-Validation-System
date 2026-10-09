
from typing import List, Optional
from agents.schemas import FlaggedClaim
from agents.utils import (
    content_coverage,
    answer_entity_conflict,
    evidence_sentences,
    match_claim_to_evidence,
    numbers_conflict,
    refutation_conflict,
    subject_conflict,
)
SUPPORT_COVERAGE_THRESHOLD = 0.60
CONFLICT_MIN_COVERAGE = 0.40
CONFLICT_COVERAGE_WINDOW = 0.15
def _find_conflict(claim: str, sentences: List[str]):
    """
    Look for an evidence sentence that actively disagrees with the claim.
    Returns (sentence, reason) or (None, None).
    Sentences are scanned best-overlap-first so the explanation quotes the
    most relevant line of evidence rather than the first one that happens
    to trip a rule.
    """
    coverages = {sentence: content_coverage(claim, sentence) for sentence in sentences}
    best_coverage = max(coverages.values()) if coverages else 0.0
    eligible = [s for s in sentences if coverages[s] >= best_coverage - CONFLICT_COVERAGE_WINDOW]
    ranked = sorted(eligible, key=lambda s: coverages[s], reverse=True)
    for sentence in ranked:
        coverage = coverages[sentence]
        if refutation_conflict(claim, sentence):
            return sentence, "the evidence explicitly describes this as untrue or exaggerated"
        if subject_conflict(claim, sentence):
            return sentence, "the evidence names a different subject for the same statement"
        if answer_entity_conflict(claim, sentence):
            return sentence, "the evidence gives a different named answer for the same fact"
        if coverage >= CONFLICT_MIN_COVERAGE and numbers_conflict(claim, sentence):
            return sentence, "the evidence states a different number for the same fact"
    return None, None
def check_claim(claim: str, evidence_texts: List[str], embed_fn=None) -> FlaggedClaim:
    """
    Classify one claim against the available evidence.
    Parameters
    ----------
    claim : str
        A single factual claim extracted from the AI response.
    evidence_texts : List[str]
        Evidence chunks. These are either chunks retrieved by the
        Milestone 1 RAG pipeline or a supplied reference answer. Nothing
        else is ever used - no evidence is invented.
    embed_fn : callable, optional
        Function mapping a list of strings to a list of vectors. Used only
        to report a similarity number alongside the verdict; the verdict
        itself does not depend on it, which keeps results identical no
        matter which embedding backend is installed.
    """
    sentences = evidence_sentences(evidence_texts)
    if not sentences:
        return FlaggedClaim(
            claim=claim,
            claim_status="Unsupported",
            evidence=None,
            reasoning=(
                "No evidence was available for this question, so this claim cannot be "
                "verified. Unsupported here means unverified, not false."
            ),
        )
    conflict_sentence, conflict_reason = _find_conflict(claim, sentences)
    if conflict_sentence is not None:
        return FlaggedClaim(
            claim=claim,
            claim_status="Contradicted",
            evidence=conflict_sentence,
            reasoning=(
                f"Classified as Contradicted because {conflict_reason}. "
                f"Evidence used: \"{conflict_sentence}\""
            ),
        )
    match = match_claim_to_evidence(claim, sentences, embed_fn=embed_fn)
    coverage = float(match["coverage"])
    similarity = float(match["similarity"])
    best_sentence: Optional[str] = match["evidence"]
    if coverage >= SUPPORT_COVERAGE_THRESHOLD:
        return FlaggedClaim(
            claim=claim,
            claim_status="Supported",
            evidence=best_sentence,
            reasoning=(
                f"Classified as Supported: {coverage:.0%} of the claim's meaningful words appear "
                f"in the evidence (embedding similarity {similarity:.2f}). "
                f"Evidence used: \"{best_sentence}\""
            ),
        )
    return FlaggedClaim(
        claim=claim,
        claim_status="Unsupported",
        evidence=best_sentence if coverage > 0 else None,
        reasoning=(
            f"Classified as Unsupported: only {coverage:.0%} of the claim's meaningful words appear "
            f"in the evidence (embedding similarity {similarity:.2f}), which is below the "
            f"{SUPPORT_COVERAGE_THRESHOLD:.0%} support threshold. The evidence does not confirm this "
            "claim and does not contradict it either, so it is unverified rather than false."
        ),
    )
def check_all_claims(claims: List[str], evidence_texts: List[str], embed_fn=None) -> List[FlaggedClaim]:
    """Run check_claim() over every extracted claim."""
    return [check_claim(claim, evidence_texts, embed_fn=embed_fn) for claim in claims]
