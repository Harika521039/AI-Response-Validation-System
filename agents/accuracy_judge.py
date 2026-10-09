
import logging
from backend import embeddings, llm_client
from agents.claim_checker import check_all_claims
from agents.schemas import AccuracyInput, AccuracyOutput
from agents.utils import split_into_claims
logger = logging.getLogger(__name__)
SCORE_LABELS = {
    5: "Completely correct",
    4: "Mostly correct (minor issue)",
    3: "Partially correct",
    2: "Mostly incorrect",
    1: "Completely incorrect / contradictory",
}
SCORING_CRITERIA = """
5 = Completely correct: every claim in the response is confirmed by the evidence.
4 = Mostly correct: the main claims are confirmed, with a minor unverified detail.
3 = Partially correct: some claims are confirmed, others are unsupported or wrong.
2 = Mostly incorrect: the evidence does not back the response, or backs very little of it.
1 = Completely incorrect or contradictory: the response directly contradicts the evidence.
""".strip()
SYSTEM_PROMPT = f"""You are the Accuracy Judge in an AI response validation system.
Your ONLY job is to decide whether the information in an AI-generated response is
factually correct, using ONLY the evidence provided to you. Do not use outside
knowledge to override the evidence, and never invent evidence. If the evidence does
not settle a claim, say so rather than guessing.
IMPORTANT: sharing a subject or topic with the evidence is not enough to call a
response correct. If the evidence says "X is a place of prayer" and the response says
"X is a football stadium", they are about the same subject but the response is wrong.
Scoring scale (1-5):
{SCORING_CRITERIA}
Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"score": <int 1-5>, "label": "<short label>", "reasoning": "<2-4 sentences naming which claims are correct or incorrect and why>", "supporting_evidence": ["<evidence snippet actually used>", ...]}}
"""
def _build_user_prompt(question: str, response: str, evidence_label: str, evidence_texts) -> str:
    evidence_block = "\n".join(f"- {e}" for e in evidence_texts) if evidence_texts else "(none provided)"
    return (
        f"Question:\n{question}\n\n"
        f"AI-generated response to check:\n{response}\n\n"
        f"{evidence_label}:\n{evidence_block}\n\n"
        "Compare the response against the evidence above ONLY. Identify correct, partially "
        "correct, incorrect, or contradictory claims. Return the JSON object now."
    )
def _select_evidence(payload: AccuracyInput):
    """
    Decide which of the two Milestone 2 evidence situations applies.
    Returns (evidence_texts, used_reference_answer, evidence_mode, label).
    """
    if payload.reference_answer and payload.reference_answer.strip():
        return [payload.reference_answer.strip()], True, "reference_answer", "Reference answer (ground truth)"
    usable = [e for e in payload.evidence if e and e.strip()]
    if usable:
        return usable, False, "rag_evidence", "Retrieved evidence from the reference knowledge base (RAG)"
    return [], False, "none", "No evidence available"
def _score_from_claims(claims) -> int:
    """
    Turn per-claim verdicts into a 1-5 accuracy score.
    Contradicted claims are treated far more harshly than unsupported ones,
    because "the evidence disagrees" is a factual error while "the evidence
    is silent" only means unverified.
    """
    total = len(claims)
    if total == 0:
        return 3
    supported = sum(1 for c in claims if c.claim_status == "Supported")
    contradicted = sum(1 for c in claims if c.claim_status == "Contradicted")
    supported_ratio = supported / total
    if contradicted:
        if supported_ratio == 0:
            return 1
        return 2 if supported_ratio <= 0.5 else 3
    if supported_ratio == 1.0:
        return 5
    if supported_ratio >= 0.6:
        return 4
    if supported_ratio > 0:
        return 3
    return 2
def _heuristic_evaluate(payload: AccuracyInput) -> AccuracyOutput:
    evidence_texts, used_reference, evidence_mode, _ = _select_evidence(payload)
    if not evidence_texts:
        return AccuracyOutput(
            score=3,
            label=SCORE_LABELS[3],
            reasoning=(
                "No reference answer was supplied and no evidence could be retrieved from the "
                "knowledge base for this question, so factual accuracy cannot be verified either "
                "way. A neutral score is reported rather than guessing."
            ),
            supporting_evidence=[],
            used_reference_answer=False,
            evidence_mode="none",
            mode="heuristic",
        )
    claims = split_into_claims(payload.response)
    checked = check_all_claims(claims, evidence_texts, embed_fn=embeddings.embed_texts)
    score = _score_from_claims(checked)
    supported = [c for c in checked if c.claim_status == "Supported"]
    unsupported = [c for c in checked if c.claim_status == "Unsupported"]
    contradicted = [c for c in checked if c.claim_status == "Contradicted"]
    source_name = "reference answer" if used_reference else "retrieved evidence"
    details = []
    if supported:
        details.append(f"{len(supported)} confirmed by the {source_name}")
    if unsupported:
        details.append(f"{len(unsupported)} not confirmed either way")
    if contradicted:
        details.append(f"{len(contradicted)} directly contradicted")
    reasoning = (
        f"Checked {len(checked)} claim(s) against the {source_name}: " + ", ".join(details) + ". "
        f"Verdict: {SCORE_LABELS[score]}."
    )
    if contradicted:
        reasoning += f" The contradicted claim is: \"{contradicted[0].claim}\""
    supporting_evidence = []
    for claim in checked:
        if claim.evidence and claim.evidence not in supporting_evidence:
            supporting_evidence.append(claim.evidence)
    return AccuracyOutput(
        score=score,
        label=SCORE_LABELS[score],
        reasoning=reasoning,
        supporting_evidence=supporting_evidence,
        used_reference_answer=used_reference,
        evidence_mode=evidence_mode,
        mode="heuristic",
    )
def _llm_evaluate(payload: AccuracyInput) -> AccuracyOutput:
    evidence_texts, used_reference, evidence_mode, evidence_label = _select_evidence(payload)
    user_prompt = _build_user_prompt(payload.question, payload.response, evidence_label, evidence_texts)
    raw = llm_client.call_llm_json(SYSTEM_PROMPT, user_prompt)
    score = int(raw["score"])
    checked = check_all_claims(
        split_into_claims(payload.response), evidence_texts, embed_fn=embeddings.embed_texts
    )
    deterministic_score = _score_from_claims(checked)
    score = min(score, deterministic_score) if any(
        claim.claim_status == "Contradicted" for claim in checked
    ) else score
    label = str(raw.get("label") or "").strip() or SCORE_LABELS.get(score, "Accuracy")
    reasoning = str(raw.get("reasoning", "")).strip() or "No reasoning provided by the model."
    supporting_evidence = [str(s) for s in (raw.get("supporting_evidence") or evidence_texts[:1])]
    return AccuracyOutput(
        score=score,
        label=label,
        reasoning=reasoning,
        supporting_evidence=supporting_evidence,
        used_reference_answer=used_reference,
        evidence_mode=evidence_mode,
        mode="llm",
    )
def evaluate_accuracy(payload: AccuracyInput) -> AccuracyOutput:
    """
    Public entry point used by the Evaluation Orchestrator.
    Uses the reference answer when one is supplied (CASE 1), otherwise the
    RAG-retrieved evidence (CASE 2).
    """
    if llm_client.is_llm_available():
        try:
            return _llm_evaluate(payload)
        except Exception as exc:
            logger.warning("Accuracy Judge: LLM mode failed (%s); using heuristic fallback.", exc)
    return _heuristic_evaluate(payload)
