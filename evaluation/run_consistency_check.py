"""
run_consistency_check.py
------------------------
M2.4 - Agent Evaluation & Consistency Validation.
This is not a fourth judge. It is a plain script that grades the three
judges against a benchmark dataset:
    1. Loads data/test_dataset.json (15 cases: correct, incorrect,
       partially correct, relevant, irrelevant, incomplete, unsupported,
       contradictory and paraphrased responses, in both evidence modes).
    2. Makes sure the Milestone 1 knowledge base is populated.
    3. Runs every case through the Evaluation Orchestrator, so all three
       agents run on identical input.
    4. Compares each actual result against the expected behaviour declared
       in the dataset.
    5. Checks paraphrase consistency: responses that mean the same thing
       must score within CONSISTENCY_TOLERANCE points of each other.
    6. Reports hallucination false positives and false negatives.
    7. Writes evaluation/results/consistency_report.json and prints a
       readable summary.
Nothing here is hardcoded to pass. Every number printed comes from an
actual agent run, and any mismatch is reported rather than hidden.
Run from the project root:
    python -m evaluation.run_consistency_check
"""
import json
import logging
import sys
from pathlib import Path
logging.basicConfig(level=logging.WARNING)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from agents import orchestrator
from backend import embeddings, retrieval, vector_store
DATASET_PATH = PROJECT_ROOT / "data" / "test_dataset.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_PATH = RESULTS_DIR / "consistency_report.json"
CONSISTENCY_TOLERANCE = 1
SCORE_TOLERANCE = 1
def load_test_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)
def run_case(case: dict) -> dict:
    """Run one test case through the orchestrator and record everything."""
    result = orchestrator.evaluate_response(
        question=case["question"],
        ai_response=case["ai_response"],
        reference_answer=case.get("reference_answer"),
    )
    return {
        "id": case["id"],
        "category": case.get("category"),
        "evidence_mode": case.get("evidence_mode"),
        "paraphrase_of": case.get("paraphrase_of"),
        "question": case["question"],
        "ai_response": case["ai_response"],
        "reference_answer": case.get("reference_answer"),
        "retrieved_evidence": [item.text for item in result.retrieved_evidence],
        "expected_behavior": case.get("expected_behavior", {}),
        "actual_output": {
            "relevance": result.relevance.model_dump(),
            "accuracy": result.accuracy.model_dump(),
            "hallucination": result.hallucination.model_dump(),
        },
        "summary": result.summary(),
    }
def analyze_case(record: dict) -> dict:
    """
    Compare one case's actual output against what the dataset expected and
    flag every kind of problem the Milestone 2 brief asks about.
    """
    expected = record.get("expected_behavior") or {}
    actual = record["actual_output"]
    expected_relevance = expected.get("relevance_score")
    expected_accuracy = expected.get("accuracy_score")
    expected_statuses = expected.get("hallucination_status") or []
    excluded_claim_statuses = expected.get("expected_claim_statuses_exclude") or []
    actual_relevance = actual["relevance"]["score"]
    actual_accuracy = actual["accuracy"]["score"]
    actual_status = actual["hallucination"]["hallucination_status"]
    actual_claim_statuses = [c["claim_status"] for c in actual["hallucination"]["flagged_claims"]]
    relevance_mismatch = (
        expected_relevance is not None
        and abs(actual_relevance - expected_relevance) > SCORE_TOLERANCE
    )
    accuracy_mismatch = (
        expected_accuracy is not None
        and abs(actual_accuracy - expected_accuracy) > SCORE_TOLERANCE
    )
    hallucination_mismatch = bool(expected_statuses) and actual_status not in expected_statuses
    false_positive = bool(expected_statuses) and expected_statuses == ["No hallucination"] and actual_status != "No hallucination"
    false_negative = (
        bool(expected_statuses)
        and "No hallucination" not in expected_statuses
        and actual_status == "No hallucination"
    )
    forbidden_claim_status = [s for s in excluded_claim_statuses if s in actual_claim_statuses]
    return {
        "id": record["id"],
        "expected_relevance_score": expected_relevance,
        "actual_relevance_score": actual_relevance,
        "relevance_mismatch": relevance_mismatch,
        "expected_accuracy_score": expected_accuracy,
        "actual_accuracy_score": actual_accuracy,
        "accuracy_mismatch": accuracy_mismatch,
        "expected_hallucination_status": expected_statuses,
        "actual_hallucination_status": actual_status,
        "hallucination_mismatch": hallucination_mismatch,
        "hallucination_false_positive": false_positive,
        "hallucination_false_negative": false_negative,
        "forbidden_claim_statuses_found": forbidden_claim_status,
        "passed": not (
            relevance_mismatch
            or accuracy_mismatch
            or hallucination_mismatch
            or forbidden_claim_status
        ),
    }
def check_paraphrase_consistency(records_by_id: dict) -> list:
    """
    Every case that declares "paraphrase_of" is compared against the case it
    paraphrases. Two responses with the same meaning must receive scores
    within CONSISTENCY_TOLERANCE of each other.
    """
    checks = []
    for record in records_by_id.values():
        origin_id = record.get("paraphrase_of")
        if not origin_id:
            continue
        origin = records_by_id.get(origin_id)
        if not origin:
            checks.append(
                {"pair": [origin_id, record["id"]], "checked": False, "reason": f"{origin_id} not found in results."}
            )
            continue
        relevance_gap = abs(
            origin["actual_output"]["relevance"]["score"] - record["actual_output"]["relevance"]["score"]
        )
        accuracy_gap = abs(
            origin["actual_output"]["accuracy"]["score"] - record["actual_output"]["accuracy"]["score"]
        )
        checks.append(
            {
                "pair": [origin_id, record["id"]],
                "checked": True,
                "evidence_mode": record.get("evidence_mode"),
                "relevance_scores": [
                    origin["actual_output"]["relevance"]["score"],
                    record["actual_output"]["relevance"]["score"],
                ],
                "relevance_gap": relevance_gap,
                "accuracy_scores": [
                    origin["actual_output"]["accuracy"]["score"],
                    record["actual_output"]["accuracy"]["score"],
                ],
                "accuracy_gap": accuracy_gap,
                "consistent": relevance_gap <= CONSISTENCY_TOLERANCE and accuracy_gap <= CONSISTENCY_TOLERANCE,
            }
        )
    return checks
def dataset_coverage(records: list) -> dict:
    """Prove the dataset really does cover both evidence modes and every case type."""
    return {
        "total_cases": len(records),
        "reference_answer_cases": sum(1 for r in records if r.get("evidence_mode") == "reference_answer"),
        "rag_evidence_cases": sum(1 for r in records if r.get("evidence_mode") == "rag"),
        "paraphrase_pairs": sum(1 for r in records if r.get("paraphrase_of")),
        "categories": sorted({r.get("category") for r in records if r.get("category")}),
    }
def print_results_table(records: list):
    header = f"{'ID':<6}{'Mode':<19}{'Rel':<5}{'Acc':<5}{'Hallucination':<24}"
    print(header)
    print("-" * len(header))
    for record in records:
        actual = record["actual_output"]
        print(
            f"{record['id']:<6}"
            f"{(record.get('evidence_mode') or ''):<19}"
            f"{actual['relevance']['score']:<5}"
            f"{actual['accuracy']['score']:<5}"
            f"{actual['hallucination']['hallucination_status']:<24}"
        )
def print_analysis_table(analyses: list):
    header = f"{'ID':<6}{'Relevance':<12}{'Accuracy':<12}{'Hallucination':<16}{'FalsePos':<10}{'FalseNeg':<10}{'Result':<8}"
    print(header)
    print("-" * len(header))
    for item in analyses:
        relevance = "n/a" if item["expected_relevance_score"] is None else ("MISMATCH" if item["relevance_mismatch"] else "ok")
        accuracy = "n/a" if item["expected_accuracy_score"] is None else ("MISMATCH" if item["accuracy_mismatch"] else "ok")
        hallucination = "n/a" if not item["expected_hallucination_status"] else ("MISMATCH" if item["hallucination_mismatch"] else "ok")
        false_positive = "YES" if item["hallucination_false_positive"] else "-"
        false_negative = "YES" if item["hallucination_false_negative"] else "-"
        verdict = "PASS" if item["passed"] else "FAIL"
        print(f"{item['id']:<6}{relevance:<12}{accuracy:<12}{hallucination:<16}{false_positive:<10}{false_negative:<10}{verdict:<8}")
def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.use_namespace("benchmark")
    print("Preparing the benchmark knowledge base (Milestone 1 RAG pipeline)...")
    try:
        added = retrieval.build_knowledge_base_if_needed(use_huggingface=False, limit_per_dataset=50)
        print(
            f"  Vector store backend: {vector_store.backend_name()} | "
            f"{added} new chunks indexed | {vector_store.count_documents()} chunks total\n"
        )
    except Exception as exc:
        print(f"  WARNING: knowledge base build failed ({exc}); evidence-based checks may be empty.\n")
    test_cases = load_test_dataset()
    print(f"Running {len(test_cases)} test cases through the Evaluation Orchestrator...\n")
    records, failures = [], []
    for case in test_cases:
        try:
            records.append(run_case(case))
        except Exception as exc:
            failures.append({"id": case.get("id"), "error": str(exc)})
            print(f"  ERROR: {case.get('id')} could not be evaluated: {exc}")
    records_by_id = {record["id"]: record for record in records}
    analyses = [analyze_case(record) for record in records]
    consistency_checks = check_paraphrase_consistency(records_by_id)
    summary = {
        "cases_run": len(records),
        "cases_passed": sum(1 for a in analyses if a["passed"]),
        "cases_failed": sum(1 for a in analyses if not a["passed"]),
        "relevance_mismatches": sum(1 for a in analyses if a["relevance_mismatch"]),
        "accuracy_mismatches": sum(1 for a in analyses if a["accuracy_mismatch"]),
        "hallucination_mismatches": sum(1 for a in analyses if a["hallucination_mismatch"]),
        "hallucination_false_positives": sum(1 for a in analyses if a["hallucination_false_positive"]),
        "hallucination_false_negatives": sum(1 for a in analyses if a["hallucination_false_negative"]),
        "consistency_pairs_checked": sum(1 for c in consistency_checks if c.get("checked")),
        "consistency_pairs_consistent": sum(1 for c in consistency_checks if c.get("consistent")),
        "consistency_pairs_inconsistent": sum(
            1 for c in consistency_checks if c.get("checked") and not c.get("consistent")
        ),
        "execution_errors": len(failures),
    }
    report = {
        "dataset_coverage": dataset_coverage(records),
        "embedding_mode": embeddings.embedding_mode(),
        "vector_store_backend": vector_store.backend_name(),
        "consistency_tolerance": CONSISTENCY_TOLERANCE,
        "score_tolerance": SCORE_TOLERANCE,
        "summary": summary,
        "results": records,
        "expected_vs_actual_analysis": analyses,
        "paraphrase_consistency_checks": consistency_checks,
        "execution_errors": failures,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("=== Agent results ===\n")
    print_results_table(records)
    print("\n=== Expected vs actual ===\n")
    print_analysis_table(analyses)
    print("\n=== Paraphrase consistency ===\n")
    for check in consistency_checks:
        if not check.get("checked"):
            print(f"{check['pair'][0]} vs {check['pair'][1]}: not checked ({check.get('reason')})")
            continue
        verdict = "CONSISTENT" if check["consistent"] else "INCONSISTENT"
        print(
            f"{check['pair'][0]} vs {check['pair'][1]} ({check['evidence_mode']}): "
            f"relevance {check['relevance_scores'][0]} vs {check['relevance_scores'][1]} "
            f"(gap {check['relevance_gap']}), "
            f"accuracy {check['accuracy_scores'][0]} vs {check['accuracy_scores'][1]} "
            f"(gap {check['accuracy_gap']}) -> {verdict}"
        )
    print("\n=== Summary ===\n")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\n  embedding mode: {embeddings.embedding_mode()}")
    print(f"  vector store backend: {vector_store.backend_name()}")
    print(f"\nFull report written to: {RESULTS_PATH}")
    if summary["cases_failed"] or summary["consistency_pairs_inconsistent"] or failures:
        print("\nSome checks did not pass. They are reported above exactly as they happened.")
        return 1
    print("\nAll expected behaviours matched and all paraphrase pairs were consistent.")
    return 0
if __name__ == "__main__":
    sys.exit(main())
