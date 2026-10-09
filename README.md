# AI Response Validation System with Hallucination Detection Assistance

A multi-agent validation framework for evaluating AI-generated responses against reference evidence. The system integrates Milestone 1 (RAG Retrieval Foundation), Milestone 2 (Judge Agents and Consistency Validation), Milestone 3 (Completeness Judge, Verdict Agent, and Resilient Batch Evaluation), and Milestone 4 (Analytics Scoring Dashboard and PDF Report Export).

---

## 1. Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Tests

```bash
pytest tests/ -v
```

All 337 automated tests execute offline and deterministically without external API calls or live model downloads.

### Run Consistency Benchmark

```bash
python -m evaluation.run_consistency_check
```

### Launch the Application

```bash
streamlit run app.py
```

The Streamlit web interface starts at `http://localhost:8501`.

The system defaults to deterministic heuristic evaluation mode. To enable LLM-backed evaluation with Anthropic Claude, set `ANTHROPIC_API_KEY` in your environment. If the API key is absent or an API request encounters an issue, the system automatically falls back to deterministic heuristic evaluation.

---

## 2. System Architecture

The pipeline processes inputs through retrieval, multi-agent evaluation, weighted scoring, and presentation layers:

```text
Question + AI Response + Optional Reference Answer / Source Info
                                 |
                                 v
                     Input Validation & SQLite Storage
                                 |
                                 v
                     RAG Evidence Retrieval
          (TruthfulQA / SQuAD -> Chunking -> Embedding -> Vector Search)
                                 |
                                 v
                 Orchestrator Evidence Routing
                                 |
    +-----------------+----------+----------+-----------------+
    |                 |                     |                 |
    v                 v                     v                 v
Relevance         Accuracy            Hallucination      Completeness
  Judge             Judge                 Agent             Judge
    |                 |                     |                 |
    +-----------------+----------+----------+-----------------+
                                 |
                                 v
                           Verdict Agent
               (Normalization -> Weights -> Critical Rules)
                                 |
                                 v
                 Consolidated Evaluation Result
                                 |
            +--------------------+--------------------+
            |                    |                    |
            v                    v                    v
     Streamlit UI          M4 Dashboard          M4 PDF Export
 (Single / Batch Views) (Metrics & Trends)    (Single / Batch PDF)
```

### Component Isolation Principles

- **Relevance Judge** evaluates only the Question and AI Response. It never receives reference evidence, preventing correctness bias from influencing topical relevance.
- **Accuracy Judge** evaluates factual claims against the reference answer when provided (Case 1), falling back to retrieved RAG chunks (Case 2).
- **Hallucination Detection Agent** evaluates individual claims extracted from the AI response against all available evidence (reference answer, user source info, and RAG chunks).
- **Completeness Judge** extracts explicit question requirements and evaluates coverage against the reference answer and retrieved evidence.
- **Verdict Agent** normalizes dimension outputs to a 0-100 scale, calculates weighted composite scores, enforces critical safety thresholds, and emits actionable issue tags.

---

## 3. Project Structure

```text
AI-Response-Validation-System/
|-- app.py                          Streamlit user interface
|-- conftest.py                     Pytest root configuration
|-- requirements.txt                Python package dependencies
|-- README.md                       System documentation
|-- FINAL_PROJECT_REPORT.md         Comprehensive final project report
|
|-- backend/                        Milestone 1 retrieval and storage
|   |-- database.py                 SQLite submission and history storage
|   |-- ingestion.py                TruthfulQA / SQuAD and demo data loader
|   |-- preprocessing.py            Text normalization and chunking
|   |-- embeddings.py               Sentence transformer with hashing fallback
|   |-- vector_store.py             ChromaDB with JSON cosine fallback
|   |-- retrieval.py                RAG pipeline and evidence retrieval
|   |-- llm_client.py               Anthropic API client wrapper
|
|-- agents/                         Milestone 2 and 3 evaluation agents
|   |-- schemas.py                  Dataclasses and strict validation models
|   |-- utils.py                    Sentence splitting and text overlap utilities
|   |-- claim_checker.py            Atomic claim extraction and verification logic
|   |-- relevance_judge.py          Relevance scoring (1-5 scale)
|   |-- accuracy_judge.py           Factual accuracy scoring (1-5 scale)
|   |-- hallucination_judge.py      Claim-level hallucination detection
|   |-- completeness_judge.py       Requirement extraction and coverage (1-5 scale)
|   |-- verdict_agent.py            Weighted scoring and critical decision rules
|   |-- orchestrator.py             Single-evaluation execution pipeline
|   |-- batch_evaluator.py          Resilient CSV batch evaluation engine
|
|-- analytics/                      Milestone 4 analytics engine
|   |-- dashboard.py                Offline metrics, distributions, and trends
|
|-- reporting/                      Milestone 4 reporting engine
|   |-- pdf_report.py               ReportLab document generator and renderer
|
|-- evaluation/                     Validation benchmarks
|   |-- run_consistency_check.py    Automated consistency test runner
|   |-- results/                    Benchmark results and validation artifacts
|
|-- data/
|   |-- demo_dataset.json           Bundled knowledge base records
|   |-- test_dataset.json           15-case evaluation benchmark
|   |-- batch_sample.csv            Sample input file for batch processing
|
|-- tests/                          337 automated unit and integration tests
    |-- test_agents.py
    |-- test_app_smoke.py
    |-- test_basic.py
    |-- test_batch_evaluator.py
    |-- test_claim_checker.py
    |-- test_completeness_judge.py
    |-- test_consistency_analysis.py
    |-- test_dashboard_analytics.py
    |-- test_dashboard_persistence.py
    |-- test_e2e_system_validation.py
    |-- test_embeddings_fallback.py
    |-- test_orchestrator_m3.py
    |-- test_pdf_report.py
    |-- test_regression_grotto.py
    |-- test_regression_m3_final.py
    |-- test_schemas.py
    |-- test_source_information.py
    |-- test_utils.py
    |-- test_vector_store_fallback.py
    |-- test_verdict_agent.py
```

---

## 4. Evaluation Agents

### Relevance Judge (M2.1)
Measures how directly the response addresses the prompt on a 1-5 scale:
- 5: Completely relevant; fully addresses the query.
- 4: Mostly relevant; answers the core question with minor extraneous detail.
- 3: Partially relevant; addresses only a subset of the prompt.
- 2: Mostly irrelevant; minor tangential connection.
- 1: Completely irrelevant or off-topic.

The judge isolates the prompt and response, ensuring an answer that is factually wrong but topical receives a high relevance score.

### Accuracy Judge (M2.2)
Evaluates factual validity against available evidence on a 1-5 scale:
- 5: Completely correct; all claims verified by evidence.
- 4: Mostly correct; core assertions verified, minor unverified detail.
- 3: Partially correct; mix of verified and unverified claims.
- 2: Mostly incorrect; little evidence support.
- 1: Completely incorrect or directly contradicted.

The output records `evidence_mode` (`reference_answer`, `rag_evidence`, or `none`) and extracts verbatim snippets into `supporting_evidence`.

### Hallucination Detection Agent (M2.3)
Splits the AI response into discrete atomic claims and classifies each:
- **Supported**: Directly confirmed by reference evidence.
- **Unsupported**: Not mentioned in the evidence (unverified, not necessarily false).
- **Contradicted**: Directly refuting reference evidence.

The aggregate status is:
- **No hallucination**: All claims supported.
- **Partially hallucinated**: At least one unsupported or contradicted claim alongside supported claims.
- **Hallucinated**: Zero claims supported by evidence.

### Completeness Judge (M3.1)
Extracts key question requirements and evaluates coverage on a 1-5 scale:
- Identifies addressed, partially addressed, and missing requirements.
- Uses reference answers first, then retrieved knowledge chunks.
- Emits explicit lists of missing aspects to inform downstream user feedback.

### Verdict Agent (M3.2)
Combines all four dimensions into an objective final assessment:
- Normalizes individual scores to a 0-100 scale:
  - Accuracy: 35% weight
  - Hallucination: 30% weight
  - Completeness: 20% weight
  - Relevance: 15% weight
- Baseline thresholds on the composite score:
  - Score >= 75: **Pass**
  - Score >= 50: **Needs Improvement**
  - Score < 50: **Fail**
- Critical Safety Rules:
  - If hallucination status is `Hallucinated`, the verdict is capped at `Fail`.
  - If accuracy score is 1, the verdict is capped at `Fail`.
  - If over 50% of claims are `Contradicted`, the verdict is capped at `Fail`.
  - The critical rules can only downgrade a verdict, never upgrade it.

---

## 5. Milestone 4 Features

### Analytics Scoring Dashboard (M4.1)
The dashboard calculation engine resides in `analytics/dashboard.py`. It runs independently of the presentation framework, enabling full offline test coverage.
- **Aggregate Metrics**: Total evaluations, successful evaluations, failed records, average overall score, and average individual dimension scores.
- **Distribution Analysis**: Exact counts and percentages for Pass, Needs Improvement, and Fail verdicts.
- **Hallucination Monitoring**: Overall hallucination frequency, clean response rate, and per-status breakdown.
- **Historical Trends**: Performance tracking across evaluation batches over time.
- **Strict Data Integrity**: Averages are computed strictly from existing dimension values without synthetic defaults or imputed zeros.

### PDF Report Export (M4.2)
The PDF reporting module in `reporting/pdf_report.py` uses a two-stage architecture:
1. `build_report_document()` creates an intermediate `ReportDocument` containing structured metadata, summary tables, score charts, claim breakdowns, and reasoning.
2. `render_pdf()` uses ReportLab to compile the document into clean, professional PDF bytes.

Features supported:
- **Single Evaluation Reports**: Full breakdown including prompt, response, dimension cards, evidence citations, individual claim statuses, and verdict rationale.
- **Batch Evaluation Reports**: Executive summary, dataset statistics, score distributions, and detailed per-record summaries with pagination and text wrapping.

---

## 6. Batch Evaluation Workflow (M3.4)

The batch engine in `agents/batch_evaluator.py` processes CSV files up to 200 rows with field lengths up to 20,000 characters:
- **Accepted Columns**: Required `question` and `ai_response`; optional `reference_answer`, `id`, and `source_information`.
- **Fault-Tolerant Processing**: File-level errors halt execution with clear feedback. Row-level errors (missing values, malformed data) skip individual rows without stopping the batch.
- **Execution Resilience**: Runtime errors during single-row evaluation mark that record as failed and continue processing subsequent records.
- **Output Capabilities**: Generates full aggregate statistics, detailed per-row drilldowns, downloadable processed CSVs, and batch PDF summaries.

---

## 7. RAG Knowledge Base and Fallbacks

Every external dependency includes a verified offline fallback:
- **Judging**: Claude LLM via Anthropic API, falling back to deterministic heuristic rules.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`, falling back to a deterministic 512-dimension hashing vectorizer.
- **Vector Storage**: ChromaDB persistent store, falling back to an in-memory JSON store with exact cosine similarity search.
- **Reference Data**: Live Hugging Face dataset downloads (TruthfulQA / SQuAD), falling back to bundled `data/demo_dataset.json`.

---

## 8. Automated Testing

The automated test suite contains **337 passing tests**:

```bash
pytest
```

Output:
```text
============================= test session starts ==============================
collected 337 items

tests/test_agents.py ...........................                         [  8%]
tests/test_app_smoke.py ................                                 [ 12%]
tests/test_basic.py ..............                                       [ 16%]
tests/test_batch_evaluator.py ............................               [ 25%]
tests/test_claim_checker.py ...........                                  [ 28%]
tests/test_completeness_judge.py ................                        [ 33%]
tests/test_consistency_analysis.py ...................                   [ 38%]
tests/test_dashboard_analytics.py ...................................... [ 50%]
tests/test_dashboard_persistence.py ..........                           [ 54%]
tests/test_e2e_system_validation.py ......................               [ 60%]
tests/test_embeddings_fallback.py ......                                 [ 62%]
tests/test_orchestrator_m3.py ..........                                 [ 65%]
tests/test_pdf_report.py ...........................                     [ 73%]
tests/test_regression_grotto.py ........                                 [ 75%]
tests/test_regression_m3_final.py ..................                     [ 81%]
tests/test_schemas.py ........                                           [ 83%]
tests/test_source_information.py ...........                             [ 86%]
tests/test_utils.py ...................                                  [ 92%]
tests/test_vector_store_fallback.py .....                                [ 93%]
tests/test_verdict_agent.py .....................                        [100%]

============================= 337 passed in 9.32s ==============================
```

Tests cover M1 foundation mechanics, M2 agents, M3 completeness and verdict logic, M3 batch processing, M4 analytics dashboard metrics, M4 PDF document generation, and end-to-end integration.
