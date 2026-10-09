
import logging
from typing import List
from backend import embeddings, llm_client
from agents.claim_checker import check_all_claims
from agents.schemas import FlaggedClaim, HallucinationInput, HallucinationOutput
from agents.utils import split_into_claims
logger = logging.getLogger(__name__)
SYSTEM_PROMPT = """You are the Hallucination Detection Agent in an AI response validation system.
Your job:
1. Extract the individual factual claims made in the AI-generated response.
2. For EACH claim, compare it against the provided evidence. Do not use outside
   knowledge and do not invent evidence.
3. Classify each claim as exactly one of:
   - "Supported": the evidence confirms the claim.
   - "Unsupported": the evidence does not mention or address the claim. This does NOT
     mean the claim is false, only that it cannot be verified from the given evidence.
   - "Contradicted": the evidence directly disagrees with the claim.
   Sharing a subject or topic with the evidence is NOT enough to call a claim
   "Supported" - the evidence must actually confirm the claim's specific content.
4. Determine an overall hallucination_status:
   - "No hallucination": every claim is Supported.
   - "Hallucinated": every claim is Contradicted by the evidence. Claims the evidence
     merely fails to mention are Unsupported, never grounds for "Hallucinated".
   - "Partially hallucinated": some claims are Supported and some are not.
Never mark an entire response "Hallucinated" just because ONE of several claims has a
problem - use "Partially hallucinated" in that case.
Respond with ONLY a JSON object, no other text, in exactly this shape:
{
  "hallucination_status": "No hallucination" | "Partially hallucinated" | "Hallucinated",
  "reasoning": "<1-3 sentences summarizing the overall verdict>",
  "flagged_claims": [
    {"claim": "<claim text>", "claim_status": "Supported|Unsupported|Contradicted", "evidence": "<evidence snippet used, or null>", "reasoning": "<why this status>"}
  ]
}
Include an entry in flagged_claims for EVERY claim you identified, not only the problematic ones.
"""
def _build_user_prompt(response: str, evidence_texts: List[str], question=None) -> str:
    evidence_block = "\n".join(f"- {e}" for e in evidence_texts) if evidence_texts else "(no evidence retrieved)"
    question_block = f"Original question (context only):\n{question}\n\n" if question else ""
    return (
        f"{question_block}"
        f"AI-generated response to check for hallucinations:\n{response}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        "Extract the claims, classify each one, and return the JSON object now."
    )
def aggregate_status(claims: List[FlaggedClaim]) -> str:
    """
    Roll individual claim verdicts up into one overall status.
    The distinction that matters here is Unsupported vs Contradicted.
    "Unsupported" only means the evidence is silent about the claim - which
    is routine when the knowledge base is thin - so it must never on its own
    condemn a response as hallucinated.
    All supported                          -> "No hallucination"
    Every claim contradicted by evidence   -> "Hallucinated"
    Anything else (incl. all unsupported)  -> "Partially hallucinated"
    """
    if not claims:
        return "No hallucination"
    statuses = [c.claim_status for c in claims]
    if all(s == "Supported" for s in statuses):
        return "No hallucination"
    if all(s == "Contradicted" for s in statuses):
        return "Hallucinated"
    return "Partially hallucinated"
def _heuristic_evaluate(payload: HallucinationInput) -> HallucinationOutput:
    claims = split_into_claims(payload.response)
    evidence_texts = [e for e in payload.evidence if e and e.strip()]
    flagged = check_all_claims(claims, evidence_texts, embed_fn=embeddings.embed_texts)
    status = aggregate_status(flagged)
    supported = sum(1 for c in flagged if c.claim_status == "Supported")
    unsupported = sum(1 for c in flagged if c.claim_status == "Unsupported")
    contradicted = sum(1 for c in flagged if c.claim_status == "Contradicted")
    reasoning = (
        f"Extracted {len(flagged)} claim(s) from the response and checked each against the "
        f"available evidence: {supported} supported, {unsupported} unsupported, "
        f"{contradicted} contradicted. Overall status: {status}. "
        "Unsupported claims are unverified by this evidence, not proven false."
    )
    return HallucinationOutput(
        hallucination_status=status,
        flagged_claims=flagged,
        reasoning=reasoning,
        mode="heuristic",
    )
def _llm_evaluate(payload: HallucinationInput) -> HallucinationOutput:
    evidence_texts = [e for e in payload.evidence if e and e.strip()]
    raw = llm_client.call_llm_json(
        SYSTEM_PROMPT,
        _build_user_prompt(payload.response, evidence_texts, payload.question),
        max_tokens=1500,
    )
    flagged = []
    for item in raw.get("flagged_claims", []):
        if item.get("claim_status") not in ("Supported", "Unsupported", "Contradicted"):
            continue
        flagged.append(
            FlaggedClaim(
                claim=str(item.get("claim", "")).strip() or "(unspecified claim)",
                claim_status=item["claim_status"],
                evidence=item.get("evidence"),
                reasoning=str(item.get("reasoning", "")).strip() or "No reasoning provided by the model.",
            )
        )
    deterministic = check_all_claims(
        split_into_claims(payload.response), evidence_texts, embed_fn=embeddings.embed_texts
    )
    if any(item.claim_status == "Contradicted" for item in deterministic):
        flagged = deterministic
    status = aggregate_status(flagged)
    reasoning = str(raw.get("reasoning", "")).strip() or "No overall reasoning provided by the model."
    return HallucinationOutput(
        hallucination_status=status,
        flagged_claims=flagged,
        reasoning=reasoning,
        mode="llm",
    )
def detect_hallucination(payload: HallucinationInput) -> HallucinationOutput:
    """
    Public entry point used by the Evaluation Orchestrator.
    Checks every claim in the response against the supplied evidence.
    """
    if llm_client.is_llm_available():
        try:
            return _llm_evaluate(payload)
        except Exception as exc:
            logger.warning("Hallucination Agent: LLM mode failed (%s); using heuristic fallback.", exc)
    return _heuristic_evaluate(payload)
