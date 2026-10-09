"""
agents package - Milestone 2: Evaluation Judge Agents.
Modules:
    schemas.py             - structured input/output models shared by everything.
    utils.py               - text analysis helpers (claim splitting, coverage, conflicts).
    claim_checker.py       - the shared "is this claim supported?" logic.
    relevance_judge.py     - M2.1: does the response answer the question?
    accuracy_judge.py      - M2.2: is the response factually correct?
    hallucination_judge.py - M2.3: which claims are supported / unsupported / contradicted?
    orchestrator.py        - runs RAG retrieval plus all three agents.
    completeness_judge.py  - M3.1: were all parts of the question addressed?
    verdict_agent.py       - M3.2: weighted overall score and final verdict.
    batch_evaluator.py     - M3.4: CSV batch evaluation (validation + per-row runs).
None of these agents answer the user's question. They only evaluate a
response that already exists.
"""
