
import math
import re
from typing import Dict, List, Optional, Tuple
def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Standard cosine similarity between two equal-length vectors."""
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z]+")
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has", "have",
    "he", "her", "his", "how", "in", "into", "is", "it", "its", "may", "might",
    "more", "most", "much", "must", "of", "on", "or", "our", "out", "over",
    "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "those", "to", "up",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "why", "will", "with", "would", "you", "your", "about", "also", "actually",
    "very", "just", "only", "really", "yes", "no", "not", "any", "all", "if",
}
_SUFFIXES = ("ally", "ing", "ies", "ied", "ed", "es", "ly", "s")
_IRREGULAR_STEMS = {
    "spoken": "speak", "spoke": "speak", "speaks": "speak", "speaking": "speak",
    "written": "write", "wrote": "write", "writes": "write", "writing": "write",
    "made": "make", "makes": "make", "making": "make",
    "given": "give", "gave": "give", "gives": "give", "giving": "give",
    "taken": "take", "took": "take", "takes": "take", "taking": "take",
    "known": "know", "knew": "know", "knows": "know",
    "grown": "grow", "grew": "grow", "grows": "grow", "growing": "grow",
    "held": "hold", "holds": "hold", "holding": "hold",
    "built": "build", "builds": "build", "building": "build",
    "found": "find", "finds": "find", "finding": "find",
    "began": "begin", "begun": "begin", "begins": "begin",
    "children": "child", "people": "person", "men": "man", "women": "woman",
    "feet": "foot", "teeth": "tooth", "leaves": "leaf",
    "produce": "product", "produces": "product", "produced": "product",
    "producing": "product", "products": "product",
    "occurrence": "occur", "occurrences": "occur", "happen": "occur",
    "happens": "occur", "happened": "occur", "happening": "occur",
}
def split_into_sentences(text: str) -> List[str]:
    """Split a block of text into individual sentences."""
    if not text or not text.strip():
        return []
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p and p.strip()]
def stem(word: str) -> str:
    """Very small suffix stripper / lemmatizer so word forms match across texts."""
    w = word.lower()
    if w in _IRREGULAR_STEMS:
        return _IRREGULAR_STEMS[w]
    for suffix in _SUFFIXES:
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w
def extract_numbers(text: str) -> List[str]:
    """Extract numeric tokens (for example '384,400', '1889', '7')."""
    if not text:
        return []
    return [n.replace(",", "") for n in _NUMBER_RE.findall(text)]
def content_words(text: str) -> List[str]:
    """
    The meaningful tokens of a text: non-stopword words (stemmed) plus any
    numbers. These are what "does the evidence back this claim?" is
    measured on.
    Numbers are included deliberately. In a factual claim the number is
    often the single most important token ("384,400 kilometers"), so
    leaving it out would let a response with the wrong figure look as
    well-supported as one with the right figure.
    """
    if not text:
        return []
    words = [w.lower() for w in _WORD_RE.findall(text)]
    kept = [stem(w) for w in words if len(w) >= 3 and w not in STOPWORDS]
    kept.extend(extract_numbers(text))
    return list(dict.fromkeys(kept))
def content_coverage(claim: str, evidence_text: str) -> float:
    """
    Fraction of the claim's content words that actually appear in the
    evidence (after stemming). 1.0 means every meaningful word of the
    claim is present in the evidence; 0.0 means none of them are.
    This is the main "is this claim backed up?" signal. Returns 0.0 when
    the claim has no content words at all, so an empty claim is never
    treated as proven.
    """
    claim_words = content_words(claim)
    if not claim_words:
        return 0.0
    evidence_words = set(content_words(evidence_text))
    if not evidence_words:
        return 0.0
    matched = sum(1 for w in claim_words if w in evidence_words)
    return matched / len(claim_words)
def keyword_overlap_ratio(question: str, response: str) -> float:
    """
    Fraction of the question's content words that appear in the response.
    Used by the Relevance Judge: a response that addresses the question
    normally reuses the things the question asked about.
    """
    q_words = content_words(question)
    if not q_words:
        return 0.0
    r_words = set(content_words(response))
    return sum(1 for w in q_words if w in r_words) / len(q_words)
_MIN_CONTENT_WORDS_FOR_CLAIM = 2
def split_into_claims(text: str) -> List[str]:
    """
    Split an AI response into individual factual claims.
    Step 1: split on sentence boundaries.
    Step 2: merge a sentence that carries fewer than two content words
            (for example "No, that's a myth.") into the next sentence,
            because on its own it states no checkable fact.
    Known limitation, worth mentioning in a walkthrough: a compound
    sentence ("X is true and Y is true") is still treated as one claim.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    claims: List[str] = []
    pending = ""
    for sentence in sentences:
        candidate = (pending + " " + sentence).strip() if pending else sentence
        if len(content_words(candidate)) < _MIN_CONTENT_WORDS_FOR_CLAIM:
            pending = candidate
            continue
        claims.append(candidate)
        pending = ""
    if pending:
        if claims:
            claims[-1] = (claims[-1] + " " + pending).strip()
        else:
            claims.append(pending)
    return claims if claims else [text.strip()]
def numbers_conflict(text_a: str, text_b: str) -> bool:
    """
    True when both texts state numbers and none of text_a's numbers appear
    in text_b, which suggests a numeric fact was changed rather than simply
    left out (for example "1,000,000 kilometers" vs "384,400 kilometers").
    """
    nums_a = set(extract_numbers(text_a))
    nums_b = set(extract_numbers(text_b))
    if not nums_a or not nums_b:
        return False
    return nums_a.isdisjoint(nums_b)
_SUBJECT_RE = re.compile(r"^\s*([A-Z][\w\s.'-]*?)\s+(?:is|was|are|were)\b")
_PRONOUN_SUBJECTS = {"it", "he", "she", "they", "this", "that", "these", "those", "there"}
def extract_subject_and_predicate(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    For "Berlin is the capital of France." return
    ("Berlin", "the capital of France.").
    Only the first sentence after the subject is kept as the predicate,
    because retrieved evidence is often a multi-sentence chunk and
    comparing against the whole chunk would dilute the overlap ratio used
    by subject_conflict().
    """
    if not text:
        return None, None
    match = _SUBJECT_RE.search(text.strip())
    if not match:
        return None, None
    subject = match.group(1).strip()
    tail = text[match.end():].strip()
    sentence_end = re.search(r"[.!?]", tail)
    predicate = tail[: sentence_end.end()] if sentence_end else tail
    return subject, predicate
def subject_conflict(text_a: str, text_b: str, overlap_threshold: float = 0.35) -> bool:
    """
    Catch the classic "Berlin is the capital of France" vs "Paris is the
    capital of France" pattern, where no numbers are involved but the named
    subject has been swapped.
    Fires only when both texts follow "<Subject> is/was ...", the subjects
    differ, and the rest of the two sentences overlaps heavily (so they are
    clearly making a claim about the same thing). That keeps it
    conservative: two merely different statements are not flagged.
    """
    subject_a, predicate_a = extract_subject_and_predicate(text_a)
    subject_b, predicate_b = extract_subject_and_predicate(text_b)
    if not subject_a or not subject_b:
        return False
    norm_a, norm_b = subject_a.lower().strip(), subject_b.lower().strip()
    if norm_a == norm_b:
        return False
    if norm_a in norm_b or norm_b in norm_a:
        return False
    if norm_a in _PRONOUN_SUBJECTS or norm_b in _PRONOUN_SUBJECTS:
        return False
    subject_a_tokens = set(content_words(subject_a))
    if subject_a_tokens and subject_a_tokens & set(content_words(text_b)):
        return False
    words_a = set(content_words(predicate_a or ""))
    words_b = set(content_words(predicate_b or ""))
    if not words_a or not words_b:
        return False
    jaccard = len(words_a & words_b) / len(words_a | words_b)
    return jaccard >= overlap_threshold
def answer_entity_conflict(claim: str, evidence: str, min_shared_context: float = 0.5) -> bool:
    """Detect a changed named answer when surrounding factual context matches.
    This complements ``subject_conflict`` for predicate-first phrasing such as
    "The capital of France is London" versus "The capital of France is Paris".
    """
    claim_subject, claim_predicate = extract_subject_and_predicate(claim)
    evidence_subject, evidence_predicate = extract_subject_and_predicate(evidence)
    if not claim_subject or not evidence_subject or not claim_predicate or not evidence_predicate:
        return False
    def _conflict(c_ctx: str, c_ans: str, e_ctx: str, e_ans: str) -> bool:
        c_ctx_w, e_ctx_w = set(content_words(c_ctx)), set(content_words(e_ctx))
        c_ans_w, e_ans_w = set(content_words(c_ans)), set(content_words(e_ans))
        if not c_ctx_w or not e_ctx_w or not c_ans_w or not e_ans_w:
            return False
        overlap = len(c_ctx_w & e_ctx_w) / len(c_ctx_w | e_ctx_w)
        return overlap >= min_shared_context and c_ans_w.isdisjoint(e_ans_w)
    if _conflict(claim_subject, claim_predicate, evidence_subject, evidence_predicate):
        return True
    for c_ctx, c_ans, e_ans, e_ctx in (
        (claim_subject, claim_predicate, evidence_subject, evidence_predicate),
        (claim_predicate, claim_subject, evidence_predicate, evidence_subject),
    ):
        if len(content_words(c_ans)) <= 3 and len(content_words(e_ans)) <= 3:
            if _conflict(c_ctx, c_ans, e_ctx, e_ans):
                return True
    return False
_REFUTATION_CUES = (
    "myth", "misconception", "exaggerated", "debunked", "is false",
    "not true", "contrary to popular belief", "despite the popular",
    "does not", "do not", "did not", "is not", "are not", "was not",
    "were not", "cannot", "never", "no link", "incorrect",
)
def has_refutation_cue(text: str) -> bool:
    """True when the text explicitly marks something as false or untrue."""
    if not text:
        return False
    return any(cue in text.lower() for cue in _REFUTATION_CUES)
def refutation_conflict(claim: str, evidence_sentence: str, min_coverage: float = 0.4) -> bool:
    """
    True when the evidence explicitly refutes this kind of statement while
    the claim asserts it as true.
    Example - evidence: "the popular claim that eating carrots gives you
    dramatically improved night vision is exaggerated"; claim: "Eating
    carrots dramatically improves your night vision." Same words, opposite
    polarity.
    Only fires in one direction (evidence refutes, claim does not), and
    only when the two texts genuinely overlap. If the claim itself carries
    a refutation cue it is agreeing with the evidence, not contradicting it.
    """
    if not claim or not evidence_sentence:
        return False
    if has_refutation_cue(claim):
        return False
    if not has_refutation_cue(evidence_sentence):
        return False
    return content_coverage(claim, evidence_sentence) >= min_coverage
def evidence_sentences(evidence_texts: List[str]) -> List[str]:
    """
    Flatten retrieved evidence chunks into individual sentences.
    Matching a claim against a whole multi-sentence chunk is misleading:
    an unrelated sentence elsewhere in the chunk can drag the numbers
    around. Sentence-level matching also gives the UI a short, quotable
    piece of evidence instead of a whole paragraph.
    """
    sentences: List[str] = []
    for chunk in evidence_texts or []:
        if chunk and chunk.strip():
            sentences.extend(split_into_sentences(chunk))
    return sentences
def match_claim_to_evidence(claim: str, sentences: List[str], embed_fn=None) -> Dict[str, object]:
    """
    Find the evidence sentence that best matches a claim.
    Ranking is by content coverage first (a stable, checkable signal), with
    embedding similarity as a tie-breaker when an embedding function is
    supplied. Returns the best sentence plus both signals, so the agents can
    explain their verdict with real numbers.
    """
    if not sentences:
        return {"evidence": None, "coverage": 0.0, "similarity": 0.0}
    coverages = [content_coverage(claim, s) for s in sentences]
    similarities = [0.0] * len(sentences)
    if embed_fn is not None:
        try:
            vectors = embed_fn([claim] + list(sentences))
            claim_vec, sentence_vecs = vectors[0], vectors[1:]
            similarities = [cosine_similarity(claim_vec, v) for v in sentence_vecs]
        except Exception:
            similarities = [0.0] * len(sentences)
    best_index = max(
        range(len(sentences)),
        key=lambda i: (round(coverages[i], 4), round(similarities[i], 4)),
    )
    return {
        "evidence": sentences[best_index],
        "coverage": coverages[best_index],
        "similarity": similarities[best_index],
    }
_QUESTION_STARTER_WORDS = (
    "what", "how", "why", "when", "where", "which", "who", "whom", "whose",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "will",
    "would", "should", "has", "have", "had", "am",
)
_QUESTION_STARTERS = "|".join(_QUESTION_STARTER_WORDS)
_CONJUNCTION_SPLIT_RE = re.compile(
    rf"\s*,\s*(?=(?:{_QUESTION_STARTERS})\b)|\s*,?\s+(?:and also|and)\s+|\s*;\s*",
    re.IGNORECASE,
)
_MIN_CONTENT_WORDS_FOR_REQUIREMENT = 2
_COMPOUND_OBJECT_VERBS = (
    "compare", "contrast", "differentiate", "distinguish", "relate",
    "combine", "match", "connect", "link", "weigh",
)
def _opens_with_compound_object_verb(sentence: str) -> bool:
    words = _WORD_RE.findall((sentence or "").lower())
    return bool(words) and words[0] in _COMPOUND_OBJECT_VERBS
def _tidy_requirement(text: str) -> str:
    """
    Normalize a requirement fragment for display and matching: collapse
    whitespace and drop trailing punctuation left behind by the split, so a
    fragment reads "mention its products" rather than "mention its products."
    (which also produced the "products.." double period in the verdict text).
    A question mark is kept, because it is part of how the ask reads.
    """
    cleaned = " ".join((text or "").split()).strip()
    while cleaned and cleaned[-1] in ".,;: ":
        cleaned = cleaned[:-1].rstrip()
    return cleaned
def extract_requirements(question: str) -> List[str]:
    """
    Split a question into its individual requirements / sub-questions.
    Step 1: split on sentence boundaries (a question can literally be
            several sentences: "What is the capital of France? How large
            is its population?").
    Step 2: within each sentence, split on a joining conjunction ("and",
            "and also", ";") IF every resulting piece carries enough
            content of its own to be a standalone ask. Otherwise the
            sentence is kept whole.
    Known limitation, worth noting in a walkthrough: a requirement phrased
    as a single compound clause without a splittable conjunction ("Compare
    the causes and effects of X") is still treated as one requirement,
    exactly like agents.utils.split_into_claims treats compound sentences
    as one claim.
    """
    sentences = split_into_sentences(question)
    if not sentences:
        return [question.strip()] if question and question.strip() else []
    requirements: List[str] = []
    for sentence in sentences:
        candidates = [c.strip() for c in _CONJUNCTION_SPLIT_RE.split(sentence) if c.strip()]
        if (
            len(candidates) > 1
            and not _opens_with_compound_object_verb(sentence)
            and all(
                len(content_words(c)) >= _MIN_CONTENT_WORDS_FOR_REQUIREMENT for c in candidates
            )
        ):
            requirements.extend(candidates)
        else:
            requirements.append(sentence)
    requirements = [r for r in (_tidy_requirement(r) for r in requirements) if r]
    return requirements if requirements else [_tidy_requirement(question)]
_YES_NO_OPENERS = (
    "do", "does", "did", "is", "are", "was", "were", "can", "could", "will",
    "would", "should", "has", "have", "had", "am",
)
_DIRECT_ANSWER_OPENERS = ("yes", "no", "nope", "yep", "correct", "incorrect", "true", "false")
def is_yes_no_question(question: str) -> bool:
    """True when the question is phrased as a yes/no question."""
    words = _WORD_RE.findall((question or "").lower())
    return bool(words) and words[0] in _YES_NO_OPENERS
def starts_with_direct_answer(response: str) -> bool:
    """True when the response opens with an explicit yes/no style answer."""
    words = _WORD_RE.findall((response or "").lower())
    return bool(words) and words[0] in _DIRECT_ANSWER_OPENERS
