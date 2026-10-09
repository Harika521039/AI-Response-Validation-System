# AI Response Validation System - Final Project Report

This is the single overall report for the project. It describes only what is
actually implemented in this codebase. Every number quoted here comes from a
real run of the test suite or the consistency checker on this code.

---

## 1. Project Overview

The system takes an AI-generated answer and decides how trustworthy it is.

Input:

- a question (required),
- an AI-generated response (required),
- a reference answer (optional),
- source information (optional, batch CSV column).

Output:

- reference evidence retrieved from a local knowledge base,
- four independent judge results (Relevance, Accuracy, Hallucination,
  Completeness), each with its own score/status and written reasoning,
- one weighted overall score out of 100 and a final verdict of
  **Pass / Needs Improvement / Fail**,
- the same pipeline applied to an uploaded CSV in batch, with a results
  table, aggregate statistics and per-row detail,
- an analytics dashboard computed from the stored evaluation history,
- downloadable PDF reports for single evaluations and batch runs.

The user interface is a Streamlit app (`app.py`). Everything below the UI is
plain Python and runs without Streamlit, which is why the whole pipeline is
unit-testable offline.

---

## 2. Architecture

```
                     Streamlit UI  (app.py)
        single evaluation | batch CSV upload | dashboard | PDF export
                                |
                                v
                 agents/orchestrator.evaluate_response()
                                |
            +-------------------+--------------------+
            |                                        |
            v                                        v
   backend/retrieval.py  (Milestone 1 RAG)   optional source_information
            |                                        |
            +-------------------+--------------------+
                                |
                        evidence for the judges
                                |
    +-----------+-----------+-----------+-----------+
    |           |           |           |
 Relevance   Accuracy  Hallucination Completeness
   Judge       Judge      Detector       Judge
    |           |           |           |
    +-----------+-----+-----+-----------+
                      |
                      v
              agents/verdict_agent.py
        (normalize -> weight -> critical rules)
                      |
                      v
        EvaluationResult (agents/schemas.py)
                      |
    +----------------+----------------+----------------+
    |                |                |                |
    v                v                v                v
 single-result   agents/batch_     analytics/       reporting/
 UI (M3.3)       evaluator.py      dashboard.py     pdf_report.py
                 (M3.4 batch +     (M4.1 metrics,   (M4.2 single and
                 CSV export)       trends, drill)   batch PDF reports)
```

### Module map

| Path | Responsibility |
| --- | --- |
| `app.py` | Streamlit UI: input form, evidence view, judge panels, final summary, dimension chart, batch upload and batch report, dashboard view, PDF download buttons |
| `backend/ingestion.py` | Loads reference data (TruthfulQA / SQuAD via `datasets`, falling back to `data/demo_dataset.json`) |
| `backend/preprocessing.py` | Cleans text and splits long contexts into chunks |
| `backend/embeddings.py` | `all-MiniLM-L6-v2` embeddings, with a deterministic hashing fallback |
| `backend/vector_store.py` | ChromaDB persistence and search, with a JSON + cosine fallback store |
| `backend/retrieval.py` | Builds the knowledge base once and retrieves top-k evidence per question |
| `backend/database.py` | SQLite storage of submissions, evaluation results, batches and dashboard records |
| `backend/llm_client.py` | Optional Anthropic client; absent key means heuristic mode |
| `agents/schemas.py` | All structured input/output dataclasses with strict validation |
| `agents/relevance_judge.py` | Relevance score 1-5 |
| `agents/accuracy_judge.py` | Accuracy score 1-5 |
| `agents/claim_checker.py` | Splits a response into individual claims |
| `agents/hallucination_judge.py` | Per-claim support check and overall hallucination status |
| `agents/completeness_judge.py` | Requirement extraction and completeness score (M3.1) |
| `agents/verdict_agent.py` | Normalization, weighting, thresholds, critical rules (M3.2) |
| `agents/orchestrator.py` | Runs the whole evaluation for one input |
| `agents/batch_evaluator.py` | CSV parsing, validation, batch run, CSV export (M3.4) |
| `analytics/dashboard.py` | Dashboard calculation layer: counts, averages, distributions, trends, drill-down (M4.1) |
| `reporting/pdf_report.py` | Report document builder and ReportLab renderer (M4.2) |
| `evaluation/run_consistency_check.py` | Benchmarks the judges against `data/test_dataset.json` (M2.4) |
| `tests/` | 337 offline tests |

### Design rule followed throughout

Every component tries the real dependency first and degrades to a documented
fallback rather than crashing: datasets -> bundled demo dataset,
sentence-transformers -> hashing vectorizer, ChromaDB -> JSON vector store,
LLM -> deterministic heuristic judging. The app therefore runs end-to-end with
no network access and no API key.

---

## 3. RAG Pipeline (Milestone 1)

1. **Ingestion** - normalizes every source record to
   `{source, question, answer, context}`.
2. **Preprocessing** - whitespace cleanup and chunking of long context text.
3. **Embedding** - `all-MiniLM-L6-v2`; if the model cannot be downloaded, a
   512-dimensional deterministic hashing vectorizer is used instead. It is a
   real embedding, just lexical rather than semantic.
4. **Storage** - ChromaDB persisted under `chroma_db/`, or the JSON fallback
   store with real cosine similarity search.
5. **Retrieval** - the question is embedded and the top-k (default 3) chunks
   are returned as `EvidenceItem`s with `text`, `source`, `question` and
   `distance`.

The knowledge base is built once per session
(`build_knowledge_base_if_needed`). If retrieval fails entirely, the
orchestrator continues with an empty evidence list and the judges report
claims as unverified rather than guessing.

---

## 4. The Agents

All four judges always run. Each sees only the inputs it is supposed to see.

### Relevance Judge
Inputs: question + response only, deliberately not the evidence, so it cannot
drift into judging correctness. Output: score 1-5, label, reasoning.

### Accuracy Judge
Inputs: question + response, plus a reference answer when one was supplied
(CASE 1, `evidence_mode = reference_answer`), otherwise the retrieved chunks
(CASE 2, `evidence_mode = rag_evidence`), otherwise nothing
(`evidence_mode = none`). Output: score 1-5, label, reasoning, the supporting
snippets it used.

### Hallucination Detection Agent
`claim_checker.py` splits the response into individual claims; each claim is
labelled **Supported / Unsupported / Contradicted** against the evidence
(reference answer first, then retrieved chunks, then any supplied source
information). The overall status is **No hallucination / Partially
hallucinated / Hallucinated**. "Unsupported" means unverified, not proven
false - that distinction is enforced in the UI copy as well.

### Completeness Judge (M3.1)
Extracts the individual requirements from the question, marks each one
**Addressed / Partially Addressed / Missing**, and returns a 1-5 score plus
the addressed, partially addressed and missing aspect lists. It uses the same
reference-answer-then-evidence fallback order as the Accuracy Judge.

### Verdict Agent (M3.2)
Reads the four judge results, never overwrites them. It normalizes each to a
0-100 scale, applies fixed weights:

| Dimension | Weight |
| --- | --- |
| Accuracy | 35% |
| Hallucination | 30% |
| Completeness | 20% |
| Relevance | 15% |

Thresholds on the weighted score: **>= 75 Pass**, **>= 50 Needs
Improvement**, otherwise **Fail**.

Critical rules can only make the verdict more severe, never less:

- hallucination status `Hallucinated` forces at most Fail,
- an accuracy score of 1 forces at most Fail,
- more than half the claims `Contradicted` forces at most Fail.

The agent also emits `major_issues` (low relevance, low accuracy, any
hallucination, low completeness) and a reasoning line naming the score, the
weights and the base verdict.

### Source Information (M3.4)
`source_information` is optional supplied evidence. When present it is placed
ahead of the retrieved chunks in the evidence given to the Accuracy Judge, the
Hallucination Detector and the Completeness Judge. It never replaces the
reference answer and changes no scoring rule. It is preserved on the
`EvaluationResult`, on every batch row (evaluated, skipped or failed), in the
row detail view and in the exported CSV.

---

## 5. Evaluation and Results Display

### Single evaluation (M3.3)
The UI shows, from the real orchestrator result and nothing recomputed:
retrieved evidence, the four judge panels with reasoning, per-claim detail and
requirement detail, a four-bar dimension comparison chart plotted directly
from `verdict.normalized_scores`, the weighted overall score out of 100, the
verdict, the weights used, and the major issues.

### Batch evaluation (M3.4)
CSV contract:

- required: `question`, `ai_response`
- optional: `reference_answer`, `id`, `source_information`
- case-insensitive headers plus documented aliases; unknown columns are
  reported and ignored
- limits: 200 rows per file, 20,000 characters per field

File-level problems (empty, binary, missing required column, duplicate
column, header only, over the row limit) raise a single clear error. Row-level
problems never stop the batch: the row is **skipped** with an exact reason. A
row whose evaluation raises is marked **failed** with the error message and
the batch continues. Progress is driven by a callback fired after each
completed row, so the progress bar reflects real work.

The results view shows:

- counts: rows in file, evaluated, skipped, failed
- the results table: **ID, Relevance, Accuracy, Hallucination, Completeness,
  Overall, Verdict, Status** (one line per input row, in file order)
- averages: relevance, accuracy, completeness, overall score
- verdict counts: Pass / Needs Improvement / Fail
- hallucination frequency as a percentage of evaluated rows, plus the
  breakdown by status
- skipped rows and failed rows with their reasons
- per-row expanders containing the complete evaluation: reasoning, supporting
  evidence, individual claims with status and evidence, requirement
  assessments, missing aspects, the chart and the verdict
- a CSV download of every row, including `source_information`

### Consistency benchmark (M2.4)
`python -m evaluation.run_consistency_check` runs the 15 cases in
`data/test_dataset.json` through the orchestrator and compares each result
against the declared expected behaviour, checks paraphrase pairs against a
tolerance, and counts hallucination false positives and negatives. Latest run
on this code (offline fallbacks active):

```
cases_run: 15          cases_passed: 15        cases_failed: 0
relevance_mismatches: 0   accuracy_mismatches: 0
hallucination_mismatches: 0
hallucination_false_positives: 0   false_negatives: 0
consistency_pairs_checked: 2   consistent: 2   inconsistent: 0
execution_errors: 0
```

---

## 6. Dashboard (M4.1)

`analytics/dashboard.py` is the dashboard calculation layer. It has no
Streamlit import, so every number the UI shows is unit-testable offline and
verifiable directly against the stored evaluation records.

Input: the list of `EvaluationRecord` objects read from the SQLite history
(each wraps one completed `EvaluationResult` plus its provenance: single or
batch, batch id, timestamp).

Output, all read off the stored records and never recomputed or invented:

- totals: evaluations recorded, successful, failed
- averages: relevance, accuracy, hallucination rate, completeness, overall
- verdict counts: Pass / Needs Improvement / Fail, with percentages against
  an explicitly reported denominator
- hallucination frequency: percentage of evaluated records whose status is
  not "No hallucination", plus the breakdown by status and the
  Supported / Unsupported / Contradicted claim counts
- dimension distributions and chart series for the UI plots
- a trend series over batches (average overall score and pass rate)
- a drill-down table of individual records

Integrity rules enforced in the module: an average over zero usable values is
`None`, never 0; a record that does not carry a dimension is excluded from
that average rather than given a placeholder; every percentage names its
denominator.

## 7. PDF Reporting (M4.2)

`reporting/pdf_report.py` runs in two deliberate stages:

1. `build_report_document()` - pure Python, no PDF library. Produces a
   `ReportDocument`: an ordered list of headings, paragraphs, bullet lists,
   tables and bar charts whose every value is read off the stored agent
   results.
2. `render_pdf()` - turns that document into PDF bytes with ReportLab.

Splitting it this way keeps the report contents verifiable directly against
the evaluation data (tests assert on the document structure and
`document.to_text()`, with no PDF parsing and no guesswork), while the
rendering stage stays a thin, replaceable presentation layer.

Single-evaluation reports contain: the question, the AI response, the four
judge results with reasoning, supporting evidence, flagged claims, missing
aspects, the overall score, the verdict, the major issues and the consolidated
reasoning.

Batch reports contain: record count, successful records, failed/invalid
records, average scores per dimension, verdict counts, hallucination
frequency, and one page per result with the same detail as a single report.
Large batches are paginated one result per page, with an optional detail cap
that, when used, is stated in the report. Long text is wrapped by the renderer
and only hard-clipped at a generous per-field limit, and then with an explicit
"truncated" notice so nothing silently disappears.

---

## 8. Testing

**337 tests, all passing** (`python -m pytest -q` -> `337 passed, 0 failed`). Every test
is offline and deterministic: no model download, no dataset download, no API
call.

| Test file | Focus |
| --- | --- |
| `test_basic.py` | Milestone 1 pipeline basics |
| `test_schemas.py` | Structured model validation |
| `test_utils.py` | Shared agent helpers |
| `test_agents.py` | Relevance, accuracy and hallucination behaviour |
| `test_claim_checker.py` | Claim splitting |
| `test_completeness_judge.py` | M3.1 requirement extraction and scoring |
| `test_verdict_agent.py` | M3.2 normalization, weights, thresholds, critical rules |
| `test_orchestrator_m3.py` | Full four-judge orchestration |
| `test_batch_evaluator.py` | M3.4 parsing, validation, batch run, export |
| `test_source_information.py` | The optional `source_information` column end to end |
| `test_app_smoke.py` | Executes `app.py` against a Streamlit stub: single evaluation, chart values, batch run, row detail |
| `test_consistency_analysis.py` | The M2.4 benchmark logic |
| `test_embeddings_fallback.py`, `test_vector_store_fallback.py` | Offline fallbacks |
| `test_dashboard_analytics.py` | M4.1 dashboard calculations against real records (41 tests) |
| `test_dashboard_persistence.py` | M4.1 dashboard records in and out of SQLite (10 tests) |
| `test_pdf_report.py` | M4.2 document structure, content and rendering (27 tests) |
| `test_e2e_system_validation.py` | End-to-end pipeline, batch and PDF verification (22 tests) |
| `test_regression_grotto.py` | Regression case kept from earlier milestones |
| `test_regression_m3_final.py` | Regression tests for the four final Milestone 3 fixes (18 tests) |

Notable checks: the dimension chart's plotted values are asserted to equal the
Verdict Agent's own `normalized_scores` from a real orchestrator run, the
batch row detail is asserted to still contain reasoning, supporting evidence,
individual claims and missing aspects, and the dashboard and PDF tests assert
their numbers against the same stored records that produced them.

---

## 9. Milestones

**Milestone 1 - RAG foundation.** Input validation, SQLite storage, dataset
ingestion, preprocessing, embeddings, vector store, top-k retrieval, Streamlit
UI.

**Milestone 2 - Judge agents.** Structured schemas, Relevance Judge, Accuracy
Judge with the two evidence modes, claim checker and Hallucination Detection
Agent, the Evaluation Orchestrator, and the M2.4 consistency benchmark.

**Milestone 3**
- **M3.1** Completeness Judge with requirement-level assessment.
- **M3.2** Verdict Agent: normalization, 35/30/20/15 weights, thresholds,
  critical rules.
- **M3.3** Results display: all four judges, the dimension comparison chart
  and the verdict block, driven entirely by real orchestrator output.
- **M3.4** Batch evaluation: CSV contract including optional
  `source_information`, file and row validation, resilient per-row execution,
  real progress, the full results table, averages, verdict counts,
  hallucination frequency, complete per-row detail and CSV export.

**Milestone 4**
- **M4.1** Evaluation Scoring Dashboard: SQLite-backed evaluation records,
  the offline dashboard calculation layer, aggregate metrics, verdict and
  hallucination distributions, trends and drill-down in the UI.
- **M4.2** PDF Report Export: the two-stage report builder and renderer,
  single-evaluation and batch PDF reports, pagination and truncation handling.

---

## 10. Challenges and How They Were Handled

- **No network, no API key.** Three separate download/API dependencies would
  each have broken the pipeline. Each got an explicit, documented fallback, so
  the same code path runs in a sandbox and with the real components.
- **Keeping judges independent.** Giving every judge all the evidence made
  them converge. The fix was to scope inputs per judge - the Relevance Judge
  never sees evidence at all.
- **Not inventing numbers.** The Verdict Agent is the only place a combined
  score exists, and the UI, the batch aggregates, the dashboard and the PDF
  reports read values rather than recompute them. The table, the chart, the
  dashboard averages and the report tables all trace back to a single
  `EvaluationResult` per record.
- **One bad row must not kill a batch.** Validation errors and evaluation
  errors are separated: file-level problems raise, row-level problems become
  skipped rows, and a raising evaluation becomes a failed row while the batch
  continues.
- **Testing a Streamlit app.** A Streamlit stub module lets the real `app.py`
  execute top to bottom inside the test suite, so a broken UI fails tests
  rather than only showing up in a browser.
- **Dashboard numbers you can re-derive.** Every dashboard figure is computed
  from stored records with an explicit denominator, and the calculation layer
  is separated from the UI so it can be asserted on directly.
- **PDF content you can verify.** Building the report document before
  rendering means the tests can assert on the exact report contents without
  parsing PDFs, and the renderer stays swappable.

---

## 11. Conclusion

The system does what it set out to do: retrieve real evidence, judge a
response on four independent dimensions, combine those judgements into one
weighted verdict under explicit rules, apply the same pipeline to a whole CSV
with honest validation, honest progress and complete per-row detail, present
the accumulated results on a dashboard computed from the stored records, and
export single and batch reports as PDFs.

All Milestone 1, 2, 3 and 4 requirements listed above are implemented and
covered by the 337-test suite, and the 15-case consistency benchmark passes
with no mismatches. Known limitation, not a defect: in the default offline
mode the judges run deterministic heuristics and the embeddings are lexical,
so scores on nuanced or paraphrased content are cruder than they would be with
the LLM and the real embedding model enabled.
