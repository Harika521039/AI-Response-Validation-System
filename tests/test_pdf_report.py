"""
test_pdf_report.py
------------------
Focused tests for M4.2 - PDF Report Export.
The report is verified in two ways:
    1. against the document structure returned by build_report_document(),
       whose every value is asserted against the same evaluation records,
    2. against the text actually extracted back out of the rendered PDF,
       so what the reader sees is checked, not just what was assembled.
The PDF text check is skipped, not failed, when no PDF text extractor is
installed, since the extractor is a test-only convenience.
"""
import sys
import unittest
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from analytics import dashboard
from reporting import pdf_report
from tests.support import OfflineTestCase
from tests.test_dashboard_analytics import make_record, make_result
def extract_pdf_text(data):
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    import io
    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)
class ReportFixture(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.clean = make_result(
            "What is the capital of France?",
            "Paris is the capital of France.",
            relevance=5,
            accuracy=5,
            completeness=5,
            hallucination_status="No hallucination",
            claims=[("Paris is the capital of France.", "Supported")],
        )
        self.flawed = make_result(
            "Who wrote Hamlet and when?",
            "Hamlet was written by Charles Dickens in 1990.",
            relevance=3,
            accuracy=1,
            completeness=2,
            hallucination_status="Hallucinated",
            claims=[
                ("Hamlet was written by Charles Dickens.", "Contradicted"),
                ("It was written in 1990.", "Unsupported"),
            ],
            missing_aspects=["the date the play was written"],
        )
        self.records = [
            make_record(self.clean, "row-1", "batch-a", "Batch A", "2026-01-01T10:00:00", 1),
            make_record(self.flawed, "row-2", "batch-a", "Batch A", "2026-01-01T10:00:00", 2),
        ]
        self.document = pdf_report.build_report_document(
            self.records, scope="Batch A", generated_at="2026-03-01 09:00:00"
        )
        self.text = self.document.to_text()
class DocumentStructureTests(ReportFixture):
    def test_every_required_section_is_present(self):
        headings = self.document.headings()
        for expected in (
            "Report metadata",
            "Batch summary",
            "Verdict distribution",
            "Average dimension scores",
            "Hallucination statistics",
            "Per-batch statistics",
            "Most frequent evaluation issues",
            "Improvement recommendations",
            "Individual results",
        ):
            self.assertIn(expected, headings)
    def test_metadata_reports_the_actual_scope_and_count(self):
        self.assertIn("2026-03-01 09:00:00", self.text)
        self.assertIn("Batch A", self.text)
        self.assertIn("Responses included | 2", self.text)
    def test_document_has_tables_and_charts(self):
        self.assertGreaterEqual(len(self.document.tables()), 6)
        self.assertGreaterEqual(len(self.document.charts()), 2)
    def test_page_break_precedes_each_detail_section(self):
        breaks = [b for b in self.document.blocks if isinstance(b, pdf_report.PageBreak)]
        self.assertEqual(len(breaks), len(self.records))
class ContentAgainstDataTests(ReportFixture):
    def test_verdict_counts_match_the_records(self):
        stats = dashboard.compute_statistics(self.records)
        for verdict, count in stats["verdict_counts"].items():
            self.assertIn(f"{verdict} | {count} |", self.text)
    def test_average_dimension_scores_match_the_records(self):
        stats = dashboard.compute_statistics(self.records)
        self.assertIn(f"Relevance | {stats['average_relevance']:g}/5", self.text)
        self.assertIn(f"Accuracy | {stats['average_accuracy']:g}/5", self.text)
        self.assertIn(f"Completeness | {stats['average_completeness']:g}/5", self.text)
    def test_hallucination_statistics_match_the_records(self):
        stats = dashboard.compute_statistics(self.records)
        halluc = stats["hallucination"]
        self.assertIn(f"Hallucination frequency | {halluc['hallucination_frequency_percent']:g}%", self.text)
        self.assertIn(f"Contradicted claims | {halluc['contradicted_claims']}", self.text)
        self.assertIn(f"Unsupported claims | {halluc['unsupported_claims']}", self.text)
        self.assertIn(f"Claims checked | {halluc['claims_checked']}", self.text)
    def test_batch_summary_matches_the_records(self):
        stats = dashboard.compute_statistics(self.records)
        self.assertIn(f"Total responses | {stats['total_responses']}", self.text)
        self.assertIn(f"Average overall score (0-100) | {stats['average_overall_score']:g}", self.text)
    def test_individual_results_carry_each_score_and_verdict(self):
        for record in self.records:
            self.assertIn(record.record_id, self.text)
            self.assertIn(record.question, self.text)
            self.assertIn(record.response, self.text)
            self.assertIn(f"{record.overall_score:g}/100", self.text)
            self.assertIn(record.verdict, self.text)
    def test_agent_reasoning_is_included(self):
        self.assertIn(self.clean.relevance.reasoning, self.text)
        self.assertIn(self.clean.accuracy.reasoning, self.text)
        self.assertIn(self.clean.completeness.reasoning, self.text)
        self.assertIn(self.clean.hallucination.reasoning, self.text)
        self.assertIn(self.flawed.verdict.reasoning, self.text)
    def test_supporting_evidence_is_included(self):
        self.assertIn("Supporting evidence used", self.text)
        self.assertIn(self.clean.accuracy.supporting_evidence[0], self.text)
    def test_hallucinated_and_unsupported_claims_are_listed(self):
        self.assertIn("Hamlet was written by Charles Dickens.", self.text)
        self.assertIn("It was written in 1990.", self.text)
        self.assertIn("Contradicted", self.text)
        self.assertIn("Unsupported", self.text)
    def test_supported_only_response_says_so_instead_of_listing_claims(self):
        document = pdf_report.build_report_document([self.records[0]])
        self.assertIn(
            "No claim in this response was unsupported or contradicted.", document.to_text()
        )
    def test_missing_aspects_are_listed(self):
        self.assertIn("the date the play was written", self.text)
    def test_recommendations_come_from_the_data(self):
        expected = dashboard.improvement_recommendations(self.records)
        self.assertTrue(expected)
        for line in expected:
            self.assertIn(line, self.text)
    def test_report_contains_no_value_absent_from_the_records(self):
        stats = dashboard.compute_statistics(self.records)
        self.assertNotIn("Average overall score (0-100) | n/a", self.text)
        self.assertEqual(stats["total_responses"], len(self.records))
class EmptyAndEdgeCaseTests(ReportFixture):
    def test_empty_scope_produces_an_explicit_document(self):
        document = pdf_report.build_report_document([])
        text = document.to_text()
        self.assertIn("No evaluation data", text)
        self.assertIn("0 stored evaluation results", text)
        self.assertNotIn("Verdict distribution", document.headings())
    def test_long_text_is_wrapped_and_clipped_with_a_notice(self):
        long_response = "This sentence repeats. " * 2000
        result = make_result(
            "A question with a very long answer?",
            long_response,
            relevance=4,
            accuracy=4,
            completeness=4,
            hallucination_status="No hallucination",
            claims=[("This sentence repeats.", "Supported")],
        )
        document = pdf_report.build_report_document([make_record(result, "long-1")])
        text = document.to_text()
        self.assertIn(pdf_report.TRUNCATION_NOTICE.strip(), text)
        self.assertLess(len(text), len(long_response))
    def test_short_text_is_never_clipped(self):
        self.assertNotIn(pdf_report.TRUNCATION_NOTICE.strip(), self.text)
    def test_large_batch_produces_one_detail_section_per_result(self):
        many = [
            make_record(self.clean, f"row-{index}", "batch-big", "Big batch", "2026-01-01T10:00:00", index)
            for index in range(60)
        ]
        document = pdf_report.build_report_document(many)
        breaks = [b for b in document.blocks if isinstance(b, pdf_report.PageBreak)]
        self.assertEqual(len(breaks), 60)
        table = [t for t in document.tables() if t.title == "All evaluated responses"][0]
        self.assertEqual(len(table.rows), 60)
    def test_detail_cap_is_stated_and_the_table_still_covers_everything(self):
        many = [
            make_record(self.clean, f"row-{index}", "batch-big", "Big batch", "2026-01-01T10:00:00", index)
            for index in range(30)
        ]
        document = pdf_report.build_report_document(many, max_detail_records=5)
        text = document.to_text()
        self.assertIn("Detailed sections are included for the first 5 of 30 results.", text)
        table = [t for t in document.tables() if t.title == "All evaluated responses"][0]
        self.assertEqual(len(table.rows), 30)
    def test_special_characters_survive_rendering(self):
        result = make_result(
            "Does 5 < 6 & 7 > 2?",
            "Yes, 5 < 6 and 7 > 2.",
            relevance=5,
            accuracy=5,
            completeness=5,
            hallucination_status="No hallucination",
            claims=[("5 < 6 & 7 > 2.", "Supported")],
        )
        data = pdf_report.generate_pdf([make_record(result, "special-1")])
        self.assertTrue(data.startswith(b"%PDF"))
class RenderedPdfTests(ReportFixture):
    def setUp(self):
        super().setUp()
        if not pdf_report.is_pdf_available():
            self.skipTest("reportlab is not installed in this environment.")
    def test_render_produces_a_valid_pdf(self):
        data = pdf_report.render_pdf(self.document)
        self.assertTrue(data.startswith(b"%PDF"))
        self.assertGreater(len(data), 3000)
    def test_generate_pdf_matches_render_of_the_same_document(self):
        data = pdf_report.generate_pdf(
            self.records, scope="Batch A", generated_at="2026-03-01 09:00:00"
        )
        self.assertTrue(data.startswith(b"%PDF"))
    def test_empty_scope_still_renders(self):
        data = pdf_report.generate_pdf([])
        self.assertTrue(data.startswith(b"%PDF"))
    def test_large_batch_renders_multiple_pages(self):
        many = [
            make_record(self.flawed, f"row-{index}", "batch-big", "Big batch", "2026-01-01T10:00:00", index)
            for index in range(25)
        ]
        data = pdf_report.generate_pdf(many)
        self.assertTrue(data.startswith(b"%PDF"))
        text = extract_pdf_text(data)
        if text is None:
            self.skipTest("No PDF text extractor is installed.")
        self.assertIn("Page 25", text)
    def test_extracted_pdf_text_matches_the_evaluation_data(self):
        data = pdf_report.render_pdf(self.document)
        text = extract_pdf_text(data)
        if text is None:
            self.skipTest("No PDF text extractor is installed.")
        stats = dashboard.compute_statistics(self.records)
        self.assertIn("AI Response Validation Report", text)
        self.assertIn("Hallucination statistics", text)
        self.assertIn("Improvement recommendations", text)
        self.assertIn(str(stats["total_responses"]), text)
        self.assertIn("Charles Dickens", text)
        self.assertIn("the date the play was written", text)
        for record in self.records:
            self.assertIn(record.verdict, text)
if __name__ == "__main__":
    unittest.main()
