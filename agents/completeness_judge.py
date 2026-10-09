
import logging
from backend import embeddings, llm_client
from agents.schemas import CompletenessInput, CompletenessOutput, RequirementAssessment
import re
from agents.utils import (
    content_words,
    evidence_sentences,
    extract_numbers,
    extract_requirements,
    match_claim_to_evidence,
    split_into_sentences,
    stem,
)
logger = logging.getLogger(__name__)
SCORE_LABELS = {
    5: "Completely complete",
    4: "Mostly complete (minor gap)",
    3: "Partially complete",
    2: "Mostly incomplete",
    1: "Incomplete",
}
SCORING_CRITERIA = """
5 = Completely complete: every identified requirement of the question is Addressed.
4 = Mostly complete: every requirement is Addressed or Partially Addressed, with at most one Partial and nothing Missing.
3 = Partially complete: at least half of the requirements are Addressed or Partially Addressed, but at least one is Missing.
2 = Mostly incomplete: fewer than half of the requirements are Addressed or Partially Addressed.
1 = Incomplete: none of the identified requirements are addressed at all.
""".strip()
SYSTEM_PROMPT = f"""You are the Completeness Judge in an AI response validation system.
Your ONLY job is to decide whether an AI-generated response addresses EVERY part of
the question that was asked, not whether it is factually correct (that is the Accuracy
Judge's job) or merely on-topic (that is the Relevance Judge's job).
Steps:
1. Identify every distinct requirement or sub-question inside the question. A question
   can ask for one thing or several ("What is X, and how does Y work?" has two).
2. For EACH requirement, using the evidence provided (a reference answer, or otherwise
   retrieved evidence, when given) to know what a complete answer to that requirement
   would mention, classify it as exactly one of:
   - "Addressed": the response clearly covers this requirement.
   - "Partially Addressed": the response touches on this requirement but only thinly
     or incompletely.
   - "Missing": the response does not cover this requirement at all.
3. Use semantic matching, not exact wording - a paraphrase that covers the same ground
   counts as Addressed.
Scoring scale (1-5):
{SCORING_CRITERIA}
Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"score": <int 1-5>, "label": "<short label>", "reasoning": "<2-4 sentences summarizing what was covered and what was not>", "requirements": [{{"requirement": "<requirement text>", "status": "Addressed|Partially Addressed|Missing", "reasoning": "<why>"}}]}}
Include an entry in "requirements" for EVERY requirement you identified.
"""
SCORE_BANDS = (
    (0.95, 5),
    (0.75, 4),
    (0.45, 3),
    (0.20, 2),
)
def _build_user_prompt(question: str, response: str, evidence_label: str, evidence_texts) -> str:
    evidence_block = "\n".join(f"- {e}" for e in evidence_texts) if evidence_texts else "(none provided)"
    return (
        f"Question:\n{question}\n\n"
        f"AI-generated response to check:\n{response}\n\n"
        f"{evidence_label}:\n{evidence_block}\n\n"
        "Identify the requirements, classify each one, and return the JSON object now."
    )
def _select_evidence(payload: CompletenessInput):
    """
    Decide which of the Milestone 3 evidence situations applies. Mirrors
    accuracy_judge._select_evidence exactly, so the two agents pick the
    same evidence for the same input.
    """
    if payload.reference_answer and payload.reference_answer.strip():
        return [payload.reference_answer.strip()], "reference_answer", "Reference answer (ground truth)"
    usable = [e for e in payload.evidence if e and e.strip()]
    if usable:
        return usable, "rag_evidence", "Retrieved evidence from the reference knowledge base (RAG)"
    return [], "none", "No evidence available"
_INSTRUCTION_WORDS = {
    stem(w)
    for w in (
        "explain", "describe", "mention", "list", "tell", "give", "state",
        "discuss", "define", "outline", "summarize", "summarise", "identify",
        "provide", "include", "elaborate", "detail", "show", "say", "answer",
        "please", "briefly", "shortly", "simple", "terms", "word", "words",
        "kindly",
    )
}
_ANSWER_SLOT_NOUNS = {
    stem(w)
    for w in (
        "language", "name", "capital", "color", "colour", "year", "date",
        "number", "amount", "reason", "cause", "author", "location", "place",
        "temperature", "size", "distance", "price", "cost", "currency",
        "population", "species", "title", "author", "inventor", "winner",
        "quantity", "percentage", "age", "height", "weight", "length",
    )
}
EVIDENCE_ADDRESSED_THRESHOLD = 0.40
EVIDENCE_PARTIAL_THRESHOLD = 0.15
ADDRESSED_COVERAGE_THRESHOLD = 0.60
PARTIAL_COVERAGE_THRESHOLD = 0.25
MIN_NOVEL_WORDS_FOR_ADDRESSED = 2
_STATUS_SEVERITY = {"Missing": 0, "Partially Addressed": 1, "Addressed": 2}
_CAPITALIZED_RE = re.compile(r"\b[A-Z][a-z]{1,}\b")
def _best_status(*statuses):
    """The most generous of the supplied statuses (None entries ignored)."""
    usable = [s for s in statuses if s]
    if not usable:
        return "Missing"
    return max(usable, key=lambda s: _STATUS_SEVERITY[s])
def _requirement_terms(requirement: str):
    """
    Split a requirement's wording into:
        topic terms - the subject matter the answer has to engage with;
        slot terms  - the KIND of answer asked for ("language", "year"),
                      which a correct answer normally does NOT repeat.
    """
    words = [w for w in content_words(requirement)]
    slots = {w for w in words if w in _ANSWER_SLOT_NOUNS}
    topic = {w for w in words if w not in slots and w not in _INSTRUCTION_WORDS}
    if not topic and not slots:
        topic = set(words)
    return topic, slots
def _best_response_sentence(target_terms, response: str) -> str:
    """The response sentence that engages most with these terms."""
    sentences = split_into_sentences(response) or ([response] if response else [])
    if not sentences:
        return ""
    if not target_terms:
        return max(sentences, key=len)
    def score(sentence):
        words = set(content_words(sentence))
        matched = sum(1 for t in target_terms if t in words)
        return (matched / len(target_terms), len(words))
    return max(sentences, key=score)
def _novel_words(sentence: str, requirement_words) -> set:
    """Content words the response contributes that the question never used."""
    return set(content_words(sentence)) - set(requirement_words)
SLOT_TOPIC_ENGAGEMENT_THRESHOLD = 0.5
def _slot_is_filled(topic, response: str, requirement_words, allow_short_answer: bool) -> bool:
    """
    Decide whether a response supplies the specific thing an answer-slot
    requirement asked for ("what language...", "in what year...").
    Counted as filled when the response engages this requirement's subject
    matter AND names something the question did not: a named entity, a
    number, or new content on top of the requirement's own wording. A short
    direct answer ("Paris.") to a single-requirement question also counts,
    since such an answer names the thing without repeating the question.
    """
    response_words = set(content_words(response))
    novel = response_words - set(requirement_words)
    if not novel:
        return False
    topic_engaged = (
        not topic
        or sum(1 for t in topic if t in response_words) / len(topic)
        >= SLOT_TOPIC_ENGAGEMENT_THRESHOLD
    )
    if allow_short_answer and len(content_words(response)) <= 3:
        return True
    if not topic_engaged:
        return False
    for sentence in split_into_sentences(response) or [response]:
        tokens = sentence.split()
        for token in tokens[1:]:
            for word in _CAPITALIZED_RE.findall(token):
                if stem(word) not in requirement_words:
                    return True
    if set(extract_numbers(response)) - set(extract_numbers(" ".join(requirement_words))):
        return True
    if topic and all(t in response_words for t in topic):
        return True
    return False
def _status_from_question_wording(requirement: str, response: str, allow_short_answer: bool) -> str:
    """
    Fallback path: with no usable evidence, judge the requirement on its own
    wording. Guarded against question-echo, which used to score highly.
    """
    topic, slots = _requirement_terms(requirement)
    requirement_words = set(content_words(requirement))
    response_words = set(content_words(response))
    if slots:
        filled = _slot_is_filled(topic, response, requirement_words, allow_short_answer)
        if filled:
            return "Addressed"
        if topic and any(t in response_words for t in topic):
            return "Partially Addressed"
        return "Missing"
    if not topic:
        return "Addressed"
    coverage = sum(1 for t in topic if t in response_words) / len(topic)
    best_sentence = _best_response_sentence(topic, response)
    novel = _novel_words(best_sentence, requirement_words)
    if coverage >= ADDRESSED_COVERAGE_THRESHOLD and len(novel) >= MIN_NOVEL_WORDS_FOR_ADDRESSED:
        return "Addressed"
    if coverage >= PARTIAL_COVERAGE_THRESHOLD and novel:
        return "Partially Addressed"
    if coverage >= ADDRESSED_COVERAGE_THRESHOLD and novel:
        return "Partially Addressed"
    return "Missing"
def _expected_answer_terms(requirement: str, evidence_texts):
    """
    What a complete answer to this requirement should actually SAY, derived
    from the evidence sentence that best matches the requirement: the
    evidence's content words minus the words the question already used.
    Returns (answer_terms, evidence_sentence) or (None, None) when the
    evidence has nothing to do with this requirement - in which case the
    evidence must not be used to judge the requirement at all.
    """
    sentences = evidence_sentences(evidence_texts)
    if not sentences:
        return None, None
    match = match_claim_to_evidence(requirement, sentences)
    sentence = match.get("evidence")
    if not sentence or not match.get("coverage"):
        return None, None
    answer_terms = set(content_words(sentence)) - set(content_words(requirement))
    if not answer_terms:
        return None, None
    return answer_terms, sentence
def _status_from_evidence(requirement: str, response: str, evidence_texts):
    """
    Primary path when evidence is available: does the response actually
    contain the answer content the evidence says this requirement calls for?
    Returns None when the evidence cannot speak to this requirement.
    """
    answer_terms, _sentence = _expected_answer_terms(requirement, evidence_texts)
    if not answer_terms:
        return None, 0.0
    response_words = set(content_words(response))
    coverage = sum(1 for t in answer_terms if t in response_words) / len(answer_terms)
    if coverage >= EVIDENCE_ADDRESSED_THRESHOLD:
        return "Addressed", coverage
    if coverage >= EVIDENCE_PARTIAL_THRESHOLD:
        return "Partially Addressed", coverage
    return "Missing", coverage
def _assess_requirement(
    requirement: str, response: str, evidence_texts, single_requirement: bool = True
) -> RequirementAssessment:
    """
    Classify one requirement as Addressed / Partially Addressed / Missing.
    Two independent signals are computed and the MORE GENEROUS one wins:
      1. Evidence path - when a reference answer or retrieved chunk actually
         relates to this requirement, the expected answer content is taken
         from it and looked for in the response. This is what lets a
         correctly-worded paraphrase ("Paris.") count as addressed even
         though it repeats none of the question.
      2. Question-wording path - used when no evidence relates to the
         requirement, and as a floor so that unrelated evidence can never
         make a good response look worse than judging on the question alone.
    """
    topic, slots = _requirement_terms(requirement)
    if not topic and not slots:
        return RequirementAssessment(
            requirement=requirement,
            status="Addressed",
            reasoning="This requirement carries no checkable content words, so it is treated as satisfied.",
        )
    evidence_status, evidence_coverage = _status_from_evidence(requirement, response, evidence_texts)
    has_evidence = evidence_status is not None
    wording_status = _status_from_question_wording(
        requirement, response, allow_short_answer=(not has_evidence) and single_requirement
    )
    status = _best_status(evidence_status, wording_status)
    if has_evidence:
        reasoning = (
            f"The response contains {evidence_coverage:.0%} of the answer content the evidence "
            f"indicates this requirement calls for (evidence path: {evidence_status}; "
            f"question-wording path: {wording_status}). Best of the two: {status}."
        )
    else:
        reasoning = (
            "No evidence relates to this requirement, so it was judged on the question's own "
            f"wording and on whether the response contributes new content: {status}."
        )
    return RequirementAssessment(requirement=requirement, status=status, reasoning=reasoning)
def _score_from_requirements(assessments):
    total = len(assessments)
    if total == 0:
        return 3
    weight = {"Addressed": 1.0, "Partially Addressed": 0.5, "Missing": 0.0}
    fraction = sum(weight[a.status] for a in assessments) / total
    for threshold, score in SCORE_BANDS:
        if fraction >= threshold:
            return score
    return 1
def _heuristic_evaluate(payload: CompletenessInput) -> CompletenessOutput:
    evidence_texts, evidence_mode, _ = _select_evidence(payload)
    requirements = extract_requirements(payload.question)
    assessments = [
        _assess_requirement(
            req, payload.response, evidence_texts, single_requirement=len(requirements) == 1
        )
        for req in requirements
    ]
    score = _score_from_requirements(assessments)
    addressed = [a.requirement for a in assessments if a.status == "Addressed"]
    partial = [a.requirement for a in assessments if a.status == "Partially Addressed"]
    missing = [a.requirement for a in assessments if a.status == "Missing"]
    reasoning = (
        f"Identified {len(assessments)} requirement(s) in the question: {len(addressed)} addressed, "
        f"{len(partial)} partially addressed, {len(missing)} missing. Verdict: {SCORE_LABELS[score]}."
    )
    if missing:
        reasoning += f" Missing: {'; '.join(missing)}."
    return CompletenessOutput(
        score=score,
        label=SCORE_LABELS[score],
        reasoning=reasoning,
        requirement_assessments=assessments,
        addressed_aspects=addressed,
        partial_aspects=partial,
        missing_aspects=missing,
        evidence_mode=evidence_mode,
        mode="heuristic",
    )
def _llm_evaluate(payload: CompletenessInput) -> CompletenessOutput:
    evidence_texts, evidence_mode, evidence_label = _select_evidence(payload)
    user_prompt = _build_user_prompt(payload.question, payload.response, evidence_label, evidence_texts)
    raw = llm_client.call_llm_json(SYSTEM_PROMPT, user_prompt, max_tokens=1200)
    assessments = []
    for item in raw.get("requirements", []):
        if item.get("status") not in ("Addressed", "Partially Addressed", "Missing"):
            continue
        assessments.append(
            RequirementAssessment(
                requirement=str(item.get("requirement", "")).strip() or "(unspecified requirement)",
                status=item["status"],
                reasoning=str(item.get("reasoning", "")).strip() or "No reasoning provided by the model.",
            )
        )
    score = int(raw["score"])
    label = str(raw.get("label") or "").strip() or SCORE_LABELS.get(score, "Completeness")
    reasoning = str(raw.get("reasoning", "")).strip() or "No reasoning provided by the model."
    addressed = [a.requirement for a in assessments if a.status == "Addressed"]
    partial = [a.requirement for a in assessments if a.status == "Partially Addressed"]
    missing = [a.requirement for a in assessments if a.status == "Missing"]
    return CompletenessOutput(
        score=score,
        label=label,
        reasoning=reasoning,
        requirement_assessments=assessments,
        addressed_aspects=addressed,
        partial_aspects=partial,
        missing_aspects=missing,
        evidence_mode=evidence_mode,
        mode="llm",
    )
def evaluate_completeness(payload: CompletenessInput) -> CompletenessOutput:
    """
    Public entry point used by the Evaluation Orchestrator.
    Uses the reference answer when one is supplied (CASE 1), otherwise the
    RAG-retrieved evidence (CASE 2), otherwise the response text alone
    (CASE 3) - exactly the same fallback order as the Accuracy Judge.
    """
    if llm_client.is_llm_available():
        try:
            return _llm_evaluate(payload)
        except Exception as exc:
            logger.warning("Completeness Judge: LLM mode failed (%s); using heuristic fallback.", exc)
    return _heuristic_evaluate(payload)
