
import logging
from backend import embeddings, llm_client
from agents.schemas import RelevanceInput, RelevanceOutput
from agents.utils import (
    cosine_similarity,
    is_yes_no_question,
    keyword_overlap_ratio,
    starts_with_direct_answer,
)
logger = logging.getLogger(__name__)
SCORE_LABELS = {
    5: "Completely relevant",
    4: "Mostly relevant",
    3: "Partially relevant",
    2: "Mostly irrelevant",
    1: "Completely irrelevant / off-topic",
}
SCORING_CRITERIA = """
5 = Completely relevant: directly and fully answers every part of the question.
4 = Mostly relevant: answers the question but adds minor unnecessary information.
3 = Partially relevant: answers only some aspects of the question, or answers a narrower related question.
2 = Mostly irrelevant: only a small, tangential connection to the question.
1 = Completely irrelevant / off-topic: does not attempt to address the question at all.
Do NOT consider whether the response is factually correct. Only judge whether it
attempts to answer what was asked. A confidently wrong answer that directly
addresses the question still scores 5.
""".strip()
SYSTEM_PROMPT = f"""You are the Relevance Judge in an AI response validation system.
Your ONLY job is to decide whether an AI-generated response answers the question
that was asked. You must NOT judge factual accuracy - a response that is completely
wrong but directly attempts to answer the question is still relevant.
Scoring scale (1-5):
{SCORING_CRITERIA}
Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"score": <int 1-5>, "label": "<short label>", "reasoning": "<1-3 sentences explaining the score>"}}
"""
COVERAGE_WEIGHT = 0.7
SIMILARITY_WEIGHT = 0.3
DIRECT_ANSWER_SIGNAL = 0.72
DIRECT_ANSWER_MIN_COVERAGE = 0.2
SCORE_BANDS = (
    (0.70, 5),
    (0.50, 4),
    (0.30, 3),
    (0.12, 2),
)
def _build_user_prompt(question: str, response: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"AI-generated response:\n{response}\n\n"
        "Evaluate ONLY relevance (does the response attempt to answer the question), "
        "not factual correctness. Return the JSON object now."
    )
def _band_for(signal: float) -> int:
    for threshold, score in SCORE_BANDS:
        if signal >= threshold:
            return score
    return 1
def _heuristic_evaluate(payload: RelevanceInput) -> RelevanceOutput:
    coverage = keyword_overlap_ratio(payload.question, payload.response)
    try:
        question_vec = embeddings.embed_single(payload.question)
        response_vec = embeddings.embed_single(payload.response)
        similarity = cosine_similarity(question_vec, response_vec)
    except Exception as exc:
        logger.warning("Relevance Judge: embeddings unavailable (%s); using word coverage only.", exc)
        similarity = 0.0
    combined = (COVERAGE_WEIGHT * coverage) + (SIMILARITY_WEIGHT * similarity)
    direct_answer = is_yes_no_question(payload.question) and starts_with_direct_answer(payload.response)
    if direct_answer and coverage >= DIRECT_ANSWER_MIN_COVERAGE:
        combined = max(combined, DIRECT_ANSWER_SIGNAL)
    score = _band_for(combined)
    reasoning = (
        f"The response addresses {coverage:.0%} of what the question asks about "
        f"(semantic similarity {similarity:.2f}, combined relevance signal {combined:.2f}), "
        f"which falls in the '{SCORE_LABELS[score]}' band. "
    )
    if direct_answer and coverage >= DIRECT_ANSWER_MIN_COVERAGE:
        reasoning += (
            "The response also gives a direct yes/no answer to a yes/no question, which counts "
            "as directly addressing it. "
        )
    reasoning += (
        "This judge measures only whether the question was addressed; factual correctness "
        "is evaluated separately by the Accuracy Judge."
    )
    return RelevanceOutput(score=score, label=SCORE_LABELS[score], reasoning=reasoning, mode="heuristic")
def _llm_evaluate(payload: RelevanceInput) -> RelevanceOutput:
    raw = llm_client.call_llm_json(SYSTEM_PROMPT, _build_user_prompt(payload.question, payload.response))
    score = int(raw["score"])
    label = str(raw.get("label") or "").strip() or SCORE_LABELS.get(score, "Relevance")
    reasoning = str(raw.get("reasoning", "")).strip() or "No reasoning provided by the model."
    return RelevanceOutput(score=score, label=label, reasoning=reasoning, mode="llm")
def evaluate_relevance(payload: RelevanceInput) -> RelevanceOutput:
    """
    Public entry point used by the Evaluation Orchestrator.
    Tries the LLM evaluator when configured; otherwise (or on any failure)
    uses the deterministic heuristic evaluator.
    """
    if llm_client.is_llm_available():
        try:
            return _llm_evaluate(payload)
        except Exception as exc:
            logger.warning("Relevance Judge: LLM mode failed (%s); using heuristic fallback.", exc)
    return _heuristic_evaluate(payload)
