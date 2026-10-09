
import logging
from typing import Optional
from backend import retrieval
from agents import accuracy_judge, completeness_judge, hallucination_judge, relevance_judge, verdict_agent
from agents.schemas import (
    AccuracyInput,
    CompletenessInput,
    EvaluationResult,
    EvidenceItem,
    HallucinationInput,
    RelevanceInput,
)
logger = logging.getLogger(__name__)
DEFAULT_TOP_K = 3
def retrieve_evidence_safely(question: str, top_k: int = DEFAULT_TOP_K):
    
    try:
        raw = retrieval.retrieve_evidence(question, top_k=top_k)
    except Exception as exc:
        logger.warning("RAG retrieval failed (%s); continuing with no retrieved evidence.", exc)
        return []
    items = []
    for entry in raw:
        try:
            items.append(EvidenceItem(**entry))
        except Exception as exc:
            logger.warning("Skipping malformed evidence entry (%s).", exc)
    return items
def evaluate_response(
    question: str,
    ai_response: str,
    reference_answer: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    source_information: Optional[str] = None,
) -> EvaluationResult:
    """
    Run the full Milestone 2 evaluation for one (question, response) pair.
    Parameters
    ----------
    question : str
        The original user question.
    ai_response : str
        The AI-generated response being evaluated.
    reference_answer : str, optional
        A known-correct answer. When supplied, the Accuracy Judge compares
        against it (CASE 1) instead of the retrieved evidence (CASE 2).
    top_k : int
        How many reference chunks to retrieve from the Milestone 1 store.
    source_information : str, optional
        Extra source material supplied alongside the row (the optional
        `source_information` batch column). It is treated as user-supplied
        evidence: it is added, ahead of the retrieved chunks, to the evidence
        the Accuracy Judge, the Hallucination Agent and the Completeness
        Judge see. It never replaces the reference answer and never changes
        how any judge scores.
    Raises
    ------
    ValueError
        If the question or the response is empty.
    """
    if not question or not question.strip():
        raise ValueError("question cannot be empty.")
    if not ai_response or not ai_response.strip():
        raise ValueError("ai_response cannot be empty.")
    reference_answer = reference_answer.strip() if reference_answer and reference_answer.strip() else None
    source_information = (
        source_information.strip() if source_information and source_information.strip() else None
    )
    evidence_items = retrieve_evidence_safely(question, top_k=top_k)
    evidence_texts = [item.text for item in evidence_items]
    if source_information:
        evidence_texts.insert(0, source_information)
    relevance_result = relevance_judge.evaluate_relevance(
        RelevanceInput(question=question, response=ai_response)
    )
    accuracy_result = accuracy_judge.evaluate_accuracy(
        AccuracyInput(
            question=question,
            response=ai_response,
            reference_answer=reference_answer,
            evidence=evidence_texts,
        )
    )
    hallucination_evidence = list(evidence_texts)
    if reference_answer:
        hallucination_evidence.insert(0, reference_answer)
    hallucination_result = hallucination_judge.detect_hallucination(
        HallucinationInput(
            question=question,
            response=ai_response,
            evidence=hallucination_evidence,
        )
    )
    completeness_result = completeness_judge.evaluate_completeness(
        CompletenessInput(
            question=question,
            response=ai_response,
            reference_answer=reference_answer,
            evidence=evidence_texts,
        )
    )
    verdict_result = verdict_agent.compute_verdict(
        relevance=relevance_result,
        accuracy=accuracy_result,
        hallucination=hallucination_result,
        completeness=completeness_result,
    )
    return EvaluationResult(
        question=question,
        response=ai_response,
        reference_answer=reference_answer,
        source_information=source_information,
        retrieved_evidence=evidence_items,
        relevance=relevance_result,
        accuracy=accuracy_result,
        hallucination=hallucination_result,
        completeness=completeness_result,
        verdict=verdict_result,
    )
