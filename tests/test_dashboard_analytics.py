"""
test_dashboard_analytics.py
---------------------------
Focused tests for M4.1 - Evaluation Scoring Dashboard.
Every assertion re-derives the expected figure by hand from the same
evaluation records the dashboard was given, so a dashboard number that
stops matching the underlying data fails here.
"""
import sys
import unittest
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from agents import orchestrator, verdict_agent
from agents.schemas import (
    AccuracyOutput,
    CompletenessOutput,
    EvaluationResult,
    FlaggedClaim,
    HallucinationOutput,
    RelevanceOutput,
)
from analytics import dashboard
from tests.support import OfflineTestCase
def make_result(
    question,
    response,
    relevance,
    accuracy,
    completeness,
    hallucination_status,
    claims=(),
    missing_aspects=(),
):
    """A complete EvaluationResult whose verdict comes from the real agent."""
    relevance_output = RelevanceOutput(score=relevance, label="Label", reasoning="Relevance reasoning.")
    accuracy_output = AccuracyOutput(
        score=accuracy,
        label="Label",
        reasoning="Accuracy reasoning.",
        supporting_evidence=["Evidence snippet."],
        evidence_mode="reference_answer",
    )
    completeness_output = CompletenessOutput(
        score=completeness,
        label="Label",
        reasoning="Completeness reasoning.",
        missing_aspects=list(missing_aspects),
    )
    hallucination_output = HallucinationOutput(
        hallucination_status=hallucination_status,
        reasoning="Hallucination reasoning.",
        flagged_claims=[
            FlaggedClaim(claim=text, claim_status=status, reasoning="Claim reasoning.")
            for text, status in claims
        ],
    )
    verdict = verdict_agent.compute_verdict(
        relevance=relevance_output,
        accuracy=accuracy_output,
        completeness=completeness_output,
        hallucination=hallucination_output,
    )
    return EvaluationResult(
        question=question,
        response=response,
        relevance=relevance_output,
        accuracy=accuracy_output,
        hallucination=hallucination_output,
        completeness=completeness_output,
        verdict=verdict,
    )
def make_record(result, record_id, batch_id="", batch_label="", created_at="2026-01-01T00:00:00", sequence=0):
    return dashboard.EvaluationRecord(
        record_id=record_id,
        payload=result.model_dump(),
        created_at=created_at,
        source=dashboard.SOURCE_BATCH if batch_id else dashboard.SOURCE_SINGLE,
        batch_id=batch_id,
        batch_label=batch_label,
        sequence=sequence,
    )
class DashboardFixture(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.strong = make_result(
            "What is the capital of France and what river runs through it?",
            "Paris is the capital of France and the Seine runs through it.",
            relevance=5,
            accuracy=5,
            completeness=5,
            hallucination_status="No hallucination",
            claims=[("Paris is the capital of France.", "Supported")],
        )
        self.middling = make_result(
            "Explain photosynthesis and name its inputs.",
            "Photosynthesis converts light into energy.",
            relevance=4,
            accuracy=3,
            completeness=2,
            hallucination_status="Partially hallucinated",
            claims=[
                ("Photosynthesis converts light into energy.", "Supported"),
                ("It happens only at night.", "Unsupported"),
            ],
            missing_aspects=["the inputs of photosynthesis"],
        )
        self.weak = make_result(
            "Who wrote Hamlet?",
            "Hamlet was written by Charles Dickens in 1990.",
            relevance=3,
            accuracy=1,
            completeness=2,
            hallucination_status="Hallucinated",
            claims=[
                ("Hamlet was written by Charles Dickens.", "Contradicted"),
                ("It was written in 1990.", "Contradicted"),
            ],
            missing_aspects=["the inputs of photosynthesis"],
        )
        self.records = [
            make_record(self.strong, "1", "batch-a", "Batch A", "2026-01-01T10:00:00", 1),
            make_record(self.middling, "2", "batch-a", "Batch A", "2026-01-01T10:00:00", 2),
            make_record(self.weak, "3", "batch-b", "Batch B", "2026-02-01T10:00:00", 1),
        ]
class RecordReadingTests(DashboardFixture):
    def test_record_reads_every_value_off_the_stored_payload(self):
        record = self.records[0]
        self.assertEqual(record.relevance_score, self.strong.relevance.score)
        self.assertEqual(record.accuracy_score, self.strong.accuracy.score)
        self.assertEqual(record.completeness_score, self.strong.completeness.score)
        self.assertEqual(record.hallucination_status, self.strong.hallucination.hallucination_status)
        self.assertEqual(record.overall_score, self.strong.verdict.weighted_overall_score)
        self.assertEqual(record.verdict, self.strong.verdict.verdict)
        self.assertEqual(record.question, self.strong.question)
        self.assertEqual(record.response, self.strong.response)
    def test_record_rejects_a_non_dict_payload(self):
        with self.assertRaises(ValueError):
            dashboard.EvaluationRecord(record_id="x", payload="not a payload")
    def test_missing_dimension_is_absent_rather_than_zero(self):
        payload = self.strong.model_dump()
        payload["completeness"] = None
        record = dashboard.EvaluationRecord(record_id="9", payload=payload)
        self.assertIsNone(record.completeness_score)
        stats = dashboard.compute_statistics([record])
        self.assertIsNone(stats["average_completeness"])
        self.assertNotEqual(stats["average_completeness"], 0)
class TotalsAndVerdictTests(DashboardFixture):
    def test_total_responses_matches_the_record_count(self):
        stats = dashboard.compute_statistics(self.records)
        self.assertEqual(stats["total_responses"], 3)
        self.assertEqual(stats["total_responses"], len(self.records))
    def test_verdict_counts_match_the_individual_verdicts(self):
        stats = dashboard.compute_statistics(self.records)
        expected = {"Pass": 0, "Needs Improvement": 0, "Fail": 0}
        for record in self.records:
            expected[record.verdict] += 1
        self.assertEqual(stats["verdict_counts"], expected)
        self.assertEqual(stats["pass_count"], expected["Pass"])
        self.assertEqual(stats["needs_improvement_count"], expected["Needs Improvement"])
        self.assertEqual(stats["fail_count"], expected["Fail"])
    def test_verdict_percentages_sum_to_one_hundred(self):
        stats = dashboard.compute_statistics(self.records)
        total = sum(value for value in stats["verdict_percentages"].values() if value is not None)
        self.assertAlmostEqual(total, 100.0, delta=0.5)
    def test_verdict_percentage_equals_count_over_total(self):
        stats = dashboard.compute_statistics(self.records)
        for verdict, count in stats["verdict_counts"].items():
            self.assertAlmostEqual(
                stats["verdict_percentages"][verdict],
                round(100.0 * count / 3, 1),
                places=1,
            )
    def test_empty_input_reports_zero_and_no_averages(self):
        stats = dashboard.compute_statistics([])
        self.assertEqual(stats["total_responses"], 0)
        self.assertEqual(stats["verdict_counts"], {"Pass": 0, "Needs Improvement": 0, "Fail": 0})
        self.assertIsNone(stats["average_accuracy"])
        self.assertIsNone(stats["pass_rate_percent"])
        self.assertIsNone(stats["average_overall_score"])
class AverageScoreTests(DashboardFixture):
    def test_average_dimension_scores_match_a_manual_mean(self):
        stats = dashboard.compute_statistics(self.records)
        self.assertAlmostEqual(stats["average_relevance"], round((5 + 4 + 3) / 3, 2))
        self.assertAlmostEqual(stats["average_accuracy"], round((5 + 3 + 1) / 3, 2))
        self.assertAlmostEqual(stats["average_completeness"], round((5 + 2 + 2) / 3, 2))
    def test_average_overall_score_matches_the_verdict_agent_scores(self):
        stats = dashboard.compute_statistics(self.records)
        expected = round(
            sum(record.overall_score for record in self.records) / len(self.records), 2
        )
        self.assertAlmostEqual(stats["average_overall_score"], expected)
    def test_average_hallucination_score_uses_the_verdict_normalization(self):
        stats = dashboard.compute_statistics(self.records)
        expected = round(
            sum(record.hallucination_normalized for record in self.records) / len(self.records), 2
        )
        self.assertAlmostEqual(stats["average_hallucination_score"], expected)
    def test_highest_and_lowest_scores_come_from_the_records(self):
        stats = dashboard.compute_statistics(self.records)
        scores = [record.overall_score for record in self.records]
        self.assertAlmostEqual(stats["highest_overall_score"], round(max(scores), 2))
        self.assertAlmostEqual(stats["lowest_overall_score"], round(min(scores), 2))
class HallucinationStatisticTests(DashboardFixture):
    def test_hallucination_counts_and_frequency(self):
        stats = dashboard.compute_statistics(self.records)
        halluc = stats["hallucination"]
        self.assertEqual(halluc["status_counts"]["No hallucination"], 1)
        self.assertEqual(halluc["status_counts"]["Partially hallucinated"], 1)
        self.assertEqual(halluc["status_counts"]["Hallucinated"], 1)
        self.assertEqual(halluc["hallucinated_responses"], 2)
        self.assertAlmostEqual(halluc["hallucination_frequency_percent"], round(200.0 / 3, 1))
        self.assertEqual(stats["hallucinated_responses"], 2)
    def test_claim_statistics_match_the_flagged_claims(self):
        stats = dashboard.compute_statistics(self.records)
        halluc = stats["hallucination"]
        expected_checked = sum(len(record.flagged_claims) for record in self.records)
        self.assertEqual(halluc["claims_checked"], expected_checked)
        self.assertEqual(halluc["unsupported_claims"], 1)
        self.assertEqual(halluc["contradicted_claims"], 2)
        self.assertEqual(halluc["problem_claims"], 3)
        self.assertAlmostEqual(halluc["problem_claim_percent"], round(300.0 / expected_checked, 1))
class ChartTests(DashboardFixture):
    def test_dimension_chart_values_match_the_averages(self):
        stats = dashboard.compute_statistics(self.records)
        averages = stats["average_normalized_scores"]
        for item in stats["dimension_chart"]:
            key = item["Dimension"].lower()
            self.assertAlmostEqual(item["Average score (0-100)"], averages[key])
        self.assertEqual(len(stats["dimension_chart"]), 4)
    def test_five_point_chart_uses_raw_judge_scores(self):
        stats = dashboard.compute_statistics(self.records)
        values = {item["Dimension"]: item["Average score (1-5)"] for item in stats["five_point_chart"]}
        self.assertAlmostEqual(values["Accuracy"], stats["average_accuracy"])
        self.assertAlmostEqual(values["Relevance"], stats["average_relevance"])
        self.assertAlmostEqual(values["Completeness"], stats["average_completeness"])
    def test_verdict_distribution_chart_matches_the_counts(self):
        stats = dashboard.compute_statistics(self.records)
        for item in stats["verdict_distribution"]:
            self.assertEqual(item["Count"], stats["verdict_counts"][item["Verdict"]])
    def test_score_distribution_covers_every_record_exactly_once(self):
        stats = dashboard.compute_statistics(self.records)
        self.assertEqual(sum(item["Count"] for item in stats["score_distribution"]), 3)
        self.assertEqual(len(stats["score_distribution"]), 10)
class BatchAndTrendTests(DashboardFixture):
    def test_batch_statistics_are_computed_per_batch(self):
        stats = dashboard.compute_statistics(self.records)
        batches = {item["batch_id"]: item for item in stats["batch_statistics"]}
        self.assertEqual(set(batches), {"batch-a", "batch-b"})
        self.assertEqual(batches["batch-a"]["total_responses"], 2)
        self.assertEqual(batches["batch-b"]["total_responses"], 1)
        self.assertAlmostEqual(
            batches["batch-a"]["average_accuracy"],
            round((5 + 3) / 2, 2),
        )
        self.assertAlmostEqual(
            batches["batch-b"]["average_overall_score"],
            round(self.records[2].overall_score, 2),
        )
    def test_combined_statistics_differ_from_a_single_batch(self):
        combined = dashboard.compute_statistics(self.records)
        batch_a = dashboard.compute_statistics(self.records[:2])
        self.assertEqual(combined["total_responses"], 3)
        self.assertEqual(batch_a["total_responses"], 2)
        self.assertNotEqual(combined["average_accuracy"], batch_a["average_accuracy"])
    def test_quality_trend_is_ordered_by_evaluation_time(self):
        stats = dashboard.compute_statistics(self.records)
        trend = stats["quality_trend"]
        self.assertEqual([item["Batch"] for item in trend], ["Batch A", "Batch B"])
        self.assertEqual(sum(item["Responses"] for item in trend), 3)
    def test_batches_covered_counts_distinct_groups(self):
        stats = dashboard.compute_statistics(self.records)
        self.assertEqual(stats["batches_covered"], 2)
class FrequentIssueTests(DashboardFixture):
    def test_repeated_missing_aspect_is_counted_across_records(self):
        issues = dashboard.frequent_issues(self.records)
        missing = [item for item in issues if item["category"] == "Missing aspect"]
        self.assertTrue(missing)
        top = missing[0]
        self.assertEqual(top["issue"], "the inputs of photosynthesis")
        self.assertEqual(top["count"], 2)
        self.assertAlmostEqual(top["affected_percent"], round(200.0 / 3, 1))
    def test_problem_claims_are_reported_as_issues(self):
        issues = dashboard.frequent_issues(self.records)
        categories = {item["category"] for item in issues}
        self.assertIn("Contradicted claim", categories)
        self.assertIn("Unsupported claim", categories)
    def test_issues_are_ranked_by_frequency(self):
        issues = dashboard.frequent_issues(self.records)
        counts = [item["count"] for item in issues]
        self.assertEqual(counts, sorted(counts, reverse=True))
    def test_issue_limit_is_respected(self):
        self.assertLessEqual(len(dashboard.frequent_issues(self.records, limit=2)), 2)
    def test_no_issues_for_a_clean_record(self):
        self.assertEqual(dashboard.frequent_issues([self.records[0]]), [])
class FilterTests(DashboardFixture):
    def test_no_filter_returns_everything(self):
        self.assertEqual(len(dashboard.apply_filters(self.records)), 3)
        self.assertFalse(dashboard.DashboardFilters().is_active())
    def test_batch_filter(self):
        filtered = dashboard.apply_filters(
            self.records, dashboard.DashboardFilters(batch_ids=["batch-b"])
        )
        self.assertEqual([r.record_id for r in filtered], ["3"])
    def test_verdict_filter(self):
        filters = dashboard.DashboardFilters(verdicts=["Fail"])
        filtered = dashboard.apply_filters(self.records, filters)
        self.assertTrue(filtered)
        for record in filtered:
            self.assertEqual(record.verdict, "Fail")
    def test_hallucinated_only_filter(self):
        filtered = dashboard.apply_filters(
            self.records, dashboard.DashboardFilters(hallucinated_only=True)
        )
        self.assertEqual(len(filtered), 2)
        for record in filtered:
            self.assertTrue(record.is_hallucinated)
    def test_score_range_filter(self):
        scores = sorted(record.overall_score for record in self.records)
        filters = dashboard.DashboardFilters(min_overall_score=scores[-1])
        filtered = dashboard.apply_filters(self.records, filters)
        self.assertEqual(len(filtered), 1)
        self.assertAlmostEqual(filtered[0].overall_score, scores[-1])
    def test_search_filter_matches_question_text(self):
        filtered = dashboard.apply_filters(
            self.records, dashboard.DashboardFilters(search_text="Hamlet")
        )
        self.assertEqual([r.record_id for r in filtered], ["3"])
    def test_filters_change_the_statistics(self):
        filters = dashboard.DashboardFilters(batch_ids=["batch-a"])
        filtered = dashboard.apply_filters(self.records, filters)
        stats = dashboard.compute_statistics(filtered)
        self.assertEqual(stats["total_responses"], 2)
        self.assertAlmostEqual(stats["average_relevance"], round((5 + 4) / 2, 2))
    def test_filter_options_only_offer_present_values(self):
        options = dashboard.filter_options(self.records)
        self.assertEqual({group[0] for group in options["groups"]}, {"batch-a", "batch-b"})
        for verdict in options["verdicts"]:
            self.assertTrue(any(record.verdict == verdict for record in self.records))
class DrillDownTests(DashboardFixture):
    def test_results_table_matches_each_record(self):
        table = dashboard.results_table(self.records)
        self.assertEqual(len(table), 3)
        for line, record in zip(table, self.records):
            self.assertEqual(line["ID"], record.record_id)
            self.assertEqual(line["Accuracy"], record.accuracy_score)
            self.assertEqual(line["Verdict"], record.verdict)
            self.assertEqual(line["Overall"], record.overall_score)
    def test_find_record_returns_the_full_underlying_result(self):
        record = dashboard.find_record(self.records, "2")
        self.assertIsNotNone(record)
        self.assertEqual(record.question, self.middling.question)
        self.assertEqual(
            record.hallucination["reasoning"],
            self.middling.hallucination.reasoning,
        )
    def test_find_record_returns_none_for_an_unknown_id(self):
        self.assertIsNone(dashboard.find_record(self.records, "does-not-exist"))
class RecommendationTests(DashboardFixture):
    def test_recommendations_quote_measured_figures(self):
        recommendations = dashboard.improvement_recommendations(self.records)
        self.assertTrue(recommendations)
        joined = " ".join(recommendations)
        self.assertIn("of 3 responses", joined)
    def test_no_recommendations_without_data(self):
        self.assertEqual(dashboard.improvement_recommendations([]), [])
class RealPipelineTests(OfflineTestCase):
    """The dashboard over results produced by the real orchestrator."""
    def test_statistics_match_an_actual_orchestrator_run(self):
        first = orchestrator.evaluate_response(
            "What is the capital of France?",
            "The capital of France is Paris.",
            reference_answer="Paris is the capital of France.",
        )
        second = orchestrator.evaluate_response(
            "What is the capital of France?",
            "The capital of France is Berlin, a city of 40 million people.",
            reference_answer="Paris is the capital of France.",
        )
        records = [
            dashboard.EvaluationRecord(record_id="1", payload=first.model_dump()),
            dashboard.EvaluationRecord(record_id="2", payload=second.model_dump()),
        ]
        stats = dashboard.compute_statistics(records)
        self.assertEqual(stats["total_responses"], 2)
        self.assertAlmostEqual(
            stats["average_accuracy"],
            round((first.accuracy.score + second.accuracy.score) / 2, 2),
        )
        self.assertAlmostEqual(
            stats["average_overall_score"],
            round(
                (first.verdict.weighted_overall_score + second.verdict.weighted_overall_score) / 2, 2
            ),
        )
        self.assertEqual(
            stats["verdict_counts"][first.verdict.verdict]
            + stats["verdict_counts"][second.verdict.verdict],
            2,
        )
if __name__ == "__main__":
    unittest.main()
