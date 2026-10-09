"""
test_app_smoke.py
-----------------
Executes app.py from top to bottom with a stubbed streamlit module.
This is a real run of the UI code - the form handling, the orchestrator
call and every render function actually execute - so a broken UI fails the
test suite instead of only showing up in a browser. It checks two paths:
    1. First page load, before the user submits anything.
    2. After the user fills in the form and clicks the button, with the
       relevance, accuracy, hallucination and final summary blocks all
       rendered.
"""
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from tests.support import OfflineTestCase
from tests import streamlit_stub
class TestStreamlitApp(OfflineTestCase):
    def setUp(self):
        super().setUp()
        patcher_module = mock.patch.dict(sys.modules, {"streamlit": streamlit_stub})
        patcher_module.start()
        self.addCleanup(patcher_module.stop)
        from backend import database, retrieval
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        patchers = [
            mock.patch.object(database, "DB_PATH", Path(self.tmp_dir.name) / "app_test.db"),
            mock.patch.object(retrieval, "build_knowledge_base_if_needed", return_value=0),
            mock.patch.object(
                retrieval,
                "retrieve_evidence",
                side_effect=lambda question, top_k=3: [
                    {
                        "text": "Paris is the capital and most populous city of France.",
                        "source": "demo",
                        "question": question,
                        "distance": 0.05,
                    }
                ],
            ),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        streamlit_stub.recorder.reset()
        self.addCleanup(sys.modules.pop, "app", None)
    def _run_app(self, inputs=None, submit=False):
        streamlit_stub.recorder.reset()
        streamlit_stub.recorder.text_inputs = inputs or []
        streamlit_stub.recorder.submit = submit
        sys.modules.pop("app", None)
        importlib.import_module("app")
        return streamlit_stub.recorder
    def test_page_loads_without_errors(self):
        recorder = self._run_app()
        rendered = recorder.rendered_text()
        self.assertIn("AI Response Validation System", rendered)
        self.assertIn("Evaluation Input", rendered)
        errors = [args for kind, args in recorder.calls if kind == "error"]
        self.assertEqual(errors, [], f"page rendered an error: {errors}")
    def test_submitting_an_empty_form_shows_validation_errors(self):
        recorder = self._run_app(inputs=["", "", ""], submit=True)
        errors = [args[0] for kind, args in recorder.calls if kind == "error"]
        self.assertIn("Question is required.", errors)
        self.assertIn("AI-generated response is required.", errors)
    def test_full_evaluation_renders_all_three_judges_and_the_summary(self):
        recorder = self._run_app(
            inputs=["What is the capital of France?", "Paris is the capital of France.", ""],
            submit=True,
        )
        rendered = recorder.rendered_text()
        self.assertIn("Retrieved Reference Evidence", rendered)
        self.assertIn("Relevance", rendered)
        self.assertIn("Accuracy", rendered)
        self.assertIn("Hallucination Detection", rendered)
        self.assertIn("Final Evaluation Summary", rendered)
        self.assertIn("No hallucination", rendered)
        errors = [args for kind, args in recorder.calls if kind == "error"]
        self.assertEqual(errors, [], f"page rendered an error: {errors}")
    def test_reference_answer_path_is_reported_in_the_ui(self):
        recorder = self._run_app(
            inputs=[
                "How far is the Moon from Earth?",
                "The Moon is about 384,400 kilometers from Earth on average.",
                "The Moon is about 384,400 kilometers from Earth on average.",
            ],
            submit=True,
        )
        rendered = recorder.rendered_text()
        self.assertIn("Compared against the reference answer you supplied.", rendered)
    def test_a_wrong_answer_is_reported_as_relevant_but_inaccurate(self):
        recorder = self._run_app(
            inputs=["What is the capital of France?", "Berlin is the capital of France.", ""],
            submit=True,
        )
        rendered = recorder.rendered_text()
        self.assertIn("Hallucinated", rendered)
        self.assertIn("Contradicted", rendered)
    def test_full_evaluation_renders_completeness_and_verdict(self):
        recorder = self._run_app(
            inputs=["What is the capital of France?", "Paris is the capital of France.", ""],
            submit=True,
        )
        rendered = recorder.rendered_text()
        self.assertIn("Completeness", rendered)
        self.assertIn("Addressed aspects", rendered)
        self.assertIn("Partially addressed aspects", rendered)
        self.assertIn("Missing aspects", rendered)
        self.assertIn("Verdict", rendered)
        self.assertIn("Weighted overall score", rendered)
        errors = [args for kind, args in recorder.calls if kind == "error"]
        self.assertEqual(errors, [], f"page rendered an error: {errors}")
    def test_comparison_chart_plots_all_four_dimensions(self):
        recorder = self._run_app(
            inputs=["What is the capital of France?", "Paris is the capital of France.", ""],
            submit=True,
        )
        rendered = recorder.rendered_text()
        self.assertIn("Dimension Comparison", rendered)
        bar_chart_calls = [args for kind, args in recorder.calls if kind == "bar_chart"]
        self.assertEqual(len(bar_chart_calls), 1, "expected exactly one comparison chart")
        plotted_labels = set(bar_chart_calls[0])
        self.assertEqual(
            plotted_labels, {"Accuracy", "Hallucination", "Completeness", "Relevance"}
        )
        self.assertIsNotNone(recorder.bar_chart_data)
        self.assertEqual(len(recorder.bar_chart_data["labels"]), 4)
        for value in recorder.bar_chart_data["values"]:
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 100)
    def test_the_ui_contains_no_emoji_characters(self):
        source = Path(__file__).resolve().parent.parent / "app.py"
        for character in source.read_text(encoding="utf-8"):
            self.assertLess(
                ord(character), 0x2190, f"app.py contains a non-plain character: {character!r}"
            )
    def _run_batch(self, csv_content):
        """Drive the M3.4 batch section: upload a file and click Run."""
        streamlit_stub.recorder.reset()
        streamlit_stub.recorder.text_inputs = []
        streamlit_stub.recorder.submit = False
        streamlit_stub.recorder.uploaded_file = streamlit_stub._UploadedFile(csv_content)
        streamlit_stub.recorder.buttons = {"Run Batch Evaluation": True}
        sys.modules.pop("app", None)
        importlib.import_module("app")
        return streamlit_stub.recorder
    def _rerun_app(self):
        """
        Rerun the page the way Streamlit does after any widget interaction
        (for example clicking the CSV download button): the script runs again
        from the top with no button pressed, but the SAME session state.
        """
        streamlit_stub.recorder.calls = []
        streamlit_stub.recorder._text_area_index = 0
        streamlit_stub.recorder.uploaded_file = None
        streamlit_stub.recorder.buttons = {}
        streamlit_stub.recorder.downloads = []
        streamlit_stub.recorder.tables = []
        sys.modules.pop("app", None)
        importlib.import_module("app")
        return streamlit_stub.recorder
    def test_batch_results_survive_a_rerun(self):
        self._run_batch(
            "question,ai_response\n"
            "What is the capital of France?,Paris is the capital of France.\n"
        )
        recorder = self._rerun_app()
        rendered = recorder.rendered_text()
        self.assertIn("Batch Summary", rendered)
        self.assertIn("Batch Results Table", rendered)
        self.assertIn("Individual Row Details", rendered)
        self.assertGreaterEqual(len(recorder.tables), 1, "results table was lost on rerun")
        self.assertTrue(
            any("Download batch results" in str(item) for item in recorder.downloads),
            "download button was lost on rerun",
        )
    def test_batch_row_details_do_not_nest_expanders(self):
        recorder = self._run_batch(
            "question,ai_response\n"
            "What is the capital of France?,Paris is the capital of France.\n"
        )
        expander_labels = [args[0] for kind, args in recorder.calls if kind == "expander"]
        for label in expander_labels:
            self.assertFalse(
                label.startswith("Evidence ") or label.startswith("Requirement "),
                f"nested expander rendered inside a batch row: {label}",
            )
        rendered = recorder.rendered_text()
        self.assertIn("Requirement 1", rendered)
    def test_batch_section_renders_on_first_load(self):
        recorder = self._run_app()
        rendered = recorder.rendered_text()
        self.assertIn("Batch Evaluation (CSV upload)", rendered)
        self.assertIn("question, ai_response", rendered)
    def test_batch_run_renders_per_row_results_and_real_progress(self):
        recorder = self._run_batch(
            "question,ai_response,reference_answer\n"
            "What is the capital of France?,Paris is the capital of France.,Paris\n"
            ",This row has no question.,\n"
        )
        rendered = recorder.rendered_text()
        self.assertIn("Batch Summary", rendered)
        self.assertIn("Individual Row Details", rendered)
        self.assertIn("Batch Results Table", rendered)
        self.assertIn("Hallucination frequency", rendered)
        for label in ("Pass", "Needs Improvement", "Fail"):
            self.assertIn(label, rendered)
        table = [
            item for item in streamlit_stub.recorder.tables
            if item and "Status" in item[0]
        ][-1]
        self.assertEqual(len(table), 2)
        self.assertEqual(
            list(table[0].keys()),
            ["ID", "Relevance", "Accuracy", "Hallucination", "Completeness",
             "Overall", "Verdict", "Status"],
        )
        self.assertIn("Skipped Rows", rendered)
        self.assertIn("'question' is empty", rendered)
        self.assertIn("Final Evaluation Summary", rendered)
        self.assertTrue(recorder.progress_values)
        self.assertEqual(recorder.progress_values[-1], 1.0)
        self.assertEqual(len(recorder.downloads), 1)
        self.assertIn("row_number", recorder.downloads[0]["data"])
    def test_batch_run_with_a_bad_file_shows_a_clear_error(self):
        recorder = self._run_batch("notes\nnothing useful here\n")
        errors = [args[0] for kind, args in recorder.calls if kind == "error"]
        self.assertTrue(any("cannot be evaluated" in message for message in errors), errors)
    def test_batch_run_shows_source_information_in_row_details(self):
        recorder = self._run_batch(
            "id,question,ai_response,reference_answer,source_information\n"
            "A1,What is the capital of France?,Paris is the capital of France.,Paris,"
            "Paris has been the capital of France since 987 AD.\n"
        )
        rendered = recorder.rendered_text()
        self.assertIn("Source Information", rendered)
        self.assertIn("987 AD", rendered)
        self.assertIn("source_information", recorder.downloads[0]["data"])
        errors = [args for kind, args in recorder.calls if kind == "error"]
        self.assertEqual(errors, [], f"page rendered an error: {errors}")
    def test_batch_row_details_keep_the_complete_evaluation(self):
        """M3.4 - inspecting one row must still show reasoning, evidence,
        claims and missing aspects, not just the summary line."""
        recorder = self._run_batch(
            "question,ai_response\n"
            "What is the capital of France?,Berlin is the capital of France.\n"
        )
        rendered = recorder.rendered_text()
        for fragment in (
            "Reasoning:",
            "Supporting evidence used:",
            "Individual claims:",
            "Missing aspects",
            "Claim 1:",
            "Weighted overall score",
        ):
            self.assertIn(fragment, rendered)
    def test_m33_display_and_chart_use_the_real_orchestrator_result(self):
        """M3.3 - the numbers shown and plotted are the Verdict Agent's own
        normalized scores from a real Orchestrator run, not UI-side values."""
        from agents import orchestrator
        question = "What is the capital of France?"
        response = "Paris is the capital of France."
        expected = orchestrator.evaluate_response(question=question, ai_response=response)
        recorder = self._run_app(inputs=[question, response, ""], submit=True)
        rendered = recorder.rendered_text()
        self.assertIn(
            f"**Weighted overall score:** {expected.verdict.weighted_overall_score}/100",
            rendered,
        )
        self.assertIn(expected.verdict.verdict, rendered)
        plotted = dict(
            zip(recorder.bar_chart_data["labels"], recorder.bar_chart_data["values"])
        )
        self.assertEqual(
            plotted,
            {
                "Accuracy": expected.verdict.normalized_scores["accuracy"],
                "Hallucination": expected.verdict.normalized_scores["hallucination"],
                "Completeness": expected.verdict.normalized_scores["completeness"],
                "Relevance": expected.verdict.normalized_scores["relevance"],
            },
        )
if __name__ == "__main__":
    unittest.main()
