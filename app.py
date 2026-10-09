
import json
import logging
from pathlib import Path
import streamlit as st
from agents import batch_evaluator, orchestrator
from analytics import dashboard as dashboard_analytics
from reporting import pdf_report
from agents.schemas import BatchValidationError
from backend import database, embeddings, llm_client, retrieval, vector_store
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
st.set_page_config(page_title="AI Response Validation System ", page_icon="AV", layout="wide")
st.markdown(
    """
<style>
:root { --ink:#172033; --muted:#667085; --line:#e4e7ec; --navy:#183153; --teal:#117a72; --soft:#f7f9fc; }
.stApp { background:#f7f9fc; color:#172033; }
[data-testid="stSidebar"] { background:#14243b; }
[data-testid="stSidebar"] * { color:#eef4fb; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding:.52rem .7rem; border-radius:6px; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background:#24466f; }
.block-container { max-width:1440px; padding-top:1.25rem; padding-bottom:3rem; }
.app-header { display:flex; justify-content:space-between; align-items:center; gap:1rem; padding:1.1rem 1.25rem; background:#ffffff; border:1px solid #e4e7ec; border-radius:8px; margin-bottom:1.25rem; }
.app-title { font-size:1.42rem; font-weight:750; color:#172033; }
.app-subtitle { color:#667085; font-size:.86rem; margin-top:.15rem; }
.status-ready, .status-warning { display:inline-flex; align-items:center; gap:.42rem; padding:.38rem .65rem; border-radius:999px; font-size:.78rem; font-weight:700; }
.status-ready { color:#08745e; background:#e8f7f2; border:1px solid #b7e4d8; }
.status-warning { color:#9a6700; background:#fff8df; border:1px solid #f2df99; }
.section-kicker { color:#117a72; font-size:.76rem; font-weight:800; text-transform:uppercase; letter-spacing:.08em; }
.section-title { color:#172033; font-size:1.55rem; font-weight:760; margin:.1rem 0 .2rem; }
.section-copy { color:#667085; margin:0 0 1.15rem; max-width:760px; }
[data-testid="stMetric"] { background:#ffffff; border:1px solid #e4e7ec; border-top:3px solid #117a72; padding:1rem; border-radius:7px; min-height:112px; }
[data-testid="stMetricLabel"] { color:#667085; }
[data-testid="stForm"], [data-testid="stExpander"], [data-testid="stVerticalBlockBorderWrapper"] { background:#ffffff; border-color:#e4e7ec; border-radius:7px; }
.stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button { border-radius:6px; font-weight:700; }
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button { background:#183153; color:#ffffff; border-color:#183153; }
[data-testid="stDataFrame"] { border:1px solid #e4e7ec; border-radius:7px; overflow:hidden; }
.claim-supported { border-left:4px solid #16866f; }
.claim-contradicted { border-left:4px solid #d64545; }
.claim-unsupported { border-left:4px solid #d79a18; }
@media (max-width: 700px) { .app-header { align-items:flex-start; flex-direction:column; } .block-container { padding-left:1rem; padding-right:1rem; } }
</style>
""",
    unsafe_allow_html=True,
)
CLAIM_STATUS_HELP = {
    "Supported": "The evidence confirms this claim.",
    "Unsupported": "Unsupported/Unverified: the available evidence does not confirm or contradict this claim.",
    "Contradicted": "The evidence directly disagrees with this claim.",
}
DIMENSION_LABELS = {
    "accuracy": "Accuracy",
    "hallucination": "Hallucination",
    "completeness": "Completeness",
    "relevance": "Relevance",
}
@st.cache_resource(show_spinner=False)
def initialize_system():
    database.init_db()
    chunks_added = 0
    error = None
    try:
        chunks_added = retrieval.build_knowledge_base_if_needed(use_huggingface=True, limit_per_dataset=50)
    except Exception as exc:
        logger.warning("Knowledge base build failed: %s", exc)
        error = str(exc)
    return {
        "chunks_added": chunks_added,
        "total_chunks": vector_store.count_documents(),
        "vector_backend": vector_store.backend_name(),
        "embedding_mode": embeddings.embedding_mode(),
        "embedding_model": embeddings.MODEL_NAME,
        "error": error,
    }
def page_intro(kicker, title, copy):
    st.markdown(f'<div class="section-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-copy">{copy}</div>', unsafe_allow_html=True)
def render_header(init_info):
    status_class = "status-warning" if init_info["error"] else "status-ready"
    status_text = "Attention required" if init_info["error"] else "System ready"
    st.markdown(
        f"""<div class="app-header"><div><div class="app-title">AI Response Validation System</div>
        <div class="app-subtitle">Evidence-grounded quality assessment</div></div>
        <div style="display:flex;gap:.55rem;align-items:center"><span class="status-ready">Milestone 3</span>
        <span class="{status_class}">{status_text}</span></div></div>""",
        unsafe_allow_html=True,
    )
def render_evidence(evidence_items):
    st.markdown("### Retrieved Reference Evidence")
    if not evidence_items:
        st.warning("No reference evidence was found. Claims remain unverified rather than being guessed.")
        return
    for index, item in enumerate(evidence_items, start=1):
        similarity = max(0.0, 1.0 - item.distance)
        with st.container(border=True):
            st.markdown(f"**Evidence {index}** | {item.source} | similarity {similarity:.1%}")
            if item.question:
                st.caption(f"Related dataset question: {item.question}")
            st.write(item.text)
def render_relevance(relevance):
    st.markdown("### Relevance")
    st.metric("Score", f"{relevance.score} / 5", relevance.label)
    st.write(f"**Reasoning:** {relevance.reasoning}")
    st.caption("Measures whether the response addresses the question, independent of factual accuracy.")
def render_accuracy(accuracy):
    st.markdown("### Accuracy")
    st.metric("Score", f"{accuracy.score} / 5", accuracy.label)
    st.write(f"**Reasoning:** {accuracy.reasoning}")
    evidence_mode_text = {
        "reference_answer": "Compared against the reference answer you supplied.",
        "rag_evidence": "Compared against retrieved knowledge-base evidence.",
        "none": "No reference answer or retrieved evidence was available.",
    }
    st.caption(evidence_mode_text.get(accuracy.evidence_mode, ""))
    st.write("**Supporting evidence used:**")
    if accuracy.supporting_evidence:
        for snippet in accuracy.supporting_evidence:
            st.markdown(f"- {snippet}")
    else:
        st.markdown("- No evidence was available for this question.")
def render_completeness(completeness):
    st.markdown("### Completeness")
    st.metric("Score", f"{completeness.score} / 5", completeness.label)
    st.write(f"**Reasoning:** {completeness.reasoning}")
    if not completeness.requirement_assessments:
        st.info("No individual requirements could be extracted from this question.")
        return
    st.write("**Requirements identified in the question:**")
    for index, assessment in enumerate(completeness.requirement_assessments, start=1):
        with st.container(border=True):
            st.markdown(f"**Requirement {index}: {assessment.status}**")
            st.write(assessment.requirement)
            st.write(f"**Reasoning:** {assessment.reasoning}")
    addressed, partial, missing = st.columns(3)
    with addressed:
        st.markdown("**Addressed aspects**")
        st.write("\n".join(f"- {x}" for x in completeness.addressed_aspects) or "None.")
    with partial:
        st.markdown("**Partially addressed aspects**")
        st.write("\n".join(f"- {x}" for x in completeness.partial_aspects) or "None.")
    with missing:
        st.markdown("**Missing aspects**")
        st.write("\n".join(f"- {x}" for x in completeness.missing_aspects) or "None.")
def render_hallucination(hallucination):
    st.markdown("### Hallucination Detection")
    st.metric("Status", hallucination.hallucination_status)
    st.write(f"**Reasoning:** {hallucination.reasoning}")
    if not hallucination.flagged_claims:
        st.info("No individual claims could be extracted from this response.")
        return
    st.write("**Individual claims:**")
    for index, claim in enumerate(hallucination.flagged_claims, start=1):
        display_status = "Unsupported/Unverified" if claim.claim_status == "Unsupported" else claim.claim_status
        css_status = claim.claim_status.lower()
        with st.container(border=True):
            st.markdown(f'<div class="claim-{css_status}">', unsafe_allow_html=True)
            st.markdown(f"**Claim {index}:** {claim.claim}")
            st.markdown(f"**Status:** {display_status}")
            st.caption(CLAIM_STATUS_HELP.get(claim.claim_status, ""))
            st.markdown(f"**Evidence:** {claim.evidence if claim.evidence else 'None available'}")
            st.markdown(f"**Reasoning:** {claim.reasoning}")
            st.markdown("</div>", unsafe_allow_html=True)
def render_comparison_chart(verdict):
    st.markdown("#### Dimension Comparison")
    order = ["accuracy", "hallucination", "completeness", "relevance"]
    chart_data = [
        {"Dimension": DIMENSION_LABELS[dim], "Normalized score (0-100)": verdict.normalized_scores[dim]}
        for dim in order
    ]
    st.bar_chart(chart_data, x="Dimension", y="Normalized score (0-100)")
def render_final_summary(result):
    st.markdown("---")
    st.subheader("Final Evaluation Summary")
    summary = result.summary()
    rel, acc, comp, hall, overall, final = st.columns(6)
    rel.metric("Relevance", f"{result.relevance.score}/5")
    acc.metric("Accuracy", f"{result.accuracy.score}/5")
    comp.metric("Completeness", f"{result.completeness.score}/5" if result.completeness else "n/a")
    hall.metric("Hallucination", result.hallucination.hallucination_status)
    overall.metric("Final Score", f"{result.verdict.weighted_overall_score}/100" if result.verdict else "n/a")
    final.metric("Final Verdict", result.verdict.verdict if result.verdict else "n/a")
    if result.verdict is not None:
        render_comparison_chart(result.verdict)
        st.markdown("#### Verdict")
        st.markdown(
            f"**Weighted overall score:** {result.verdict.weighted_overall_score}/100  |  "
            f"**Verdict:** {result.verdict.verdict}"
        )
        if result.verdict.major_issues:
            st.write("**Major issues:**")
            for issue in result.verdict.major_issues:
                st.markdown(f"- {issue}")
        st.write(f"**Reasoning:** {result.verdict.reasoning}")
    st.caption(
        f"Claims flagged: {summary['unsupported_or_contradicted_claims']} of "
        f"{summary['total_claims_checked']} checked."
    )
def _avg_text(value, out_of=5):
    return f"{value}/{out_of}" if value is not None else "n/a"
BATCH_REPORT_STATE_KEY = "batch_report"
BATCH_ID_STATE_KEY = "batch_id"
BATCH_LABEL_STATE_KEY = "batch_label"
def _avg_text(value, scale=5):
    if value is None:
        return "No data"
    return f"{value:.2f}/{scale}"
def _percent_text(value):
    if value is None:
        return "No data"
    return f"{value:.1f}%"
def render_batch_report(report):
    stats = report.aggregate()
    st.markdown("#### Batch Summary")
    total, done, skipped, failed = st.columns(4)
    total.metric("Total rows", stats["total_rows_in_file"])
    done.metric("Evaluated", stats["evaluated"])
    skipped.metric("Skipped", stats["skipped"])
    failed.metric("Failed", stats["failed"])
    if report.ignored_columns:
        st.caption("Ignored columns: " + ", ".join(report.ignored_columns))
    if stats["evaluated"]:
        st.markdown("##### Average scores")
        rel, acc, comp, overall = st.columns(4)
        rel.metric("Average Relevance", _avg_text(stats.get("average_relevance")))
        acc.metric("Average Accuracy", _avg_text(stats.get("average_accuracy")))
        comp.metric("Average Completeness", _avg_text(stats.get("average_completeness")))
        overall.metric("Average Overall", f"{stats['average_overall_score']}/100")
        st.markdown("##### Verdict Breakdown")
        verdict_counts = stats.get("verdict_counts") or {}
        st.bar_chart(
            [{"Verdict": key, "Count": verdict_counts.get(key, 0)} for key in ("Pass", "Needs Improvement", "Fail")],
            x="Verdict", y="Count",
        )
        frequency = stats.get("hallucination_frequency_percent")
        st.metric("Hallucination Rate", f"{frequency}%" if frequency is not None else "n/a")
        st.caption(
            f"Hallucination frequency = {stats.get('hallucination_rows', 0)} of {stats['evaluated']} evaluated rows."
        )
    else:
        st.warning("No row in this file could be evaluated.")
    st.markdown("#### Batch Results Table")
    st.dataframe(report.results_table(), use_container_width=True)
    if report.skipped_rows:
        st.markdown("#### Skipped Rows")
        for row in report.skipped_rows:
            st.markdown(f"- Row {row.row_number}: {row.error}")
    if report.failed_rows:
        st.markdown("#### Failed Rows")
        for row in report.failed_rows:
            st.markdown(f"- Row {row.row_number}: {row.error}")
    if report.evaluated_rows:
        st.markdown("#### Individual Row Details")
        for row in report.evaluated_rows:
            label = row.row_id or f"Row {row.row_number}"
            verdict_text = f" - {row.evaluation.verdict.verdict}" if row.evaluation and row.evaluation.verdict else ""
            with st.expander(f"{label}: {row.question[:60]}{verdict_text}"):
                st.write(f"**Question:** {row.question}")
                st.write(f"**AI Response:** {row.ai_response}")
                if row.reference_answer:
                    st.write(f"**Reference Answer:** {row.reference_answer}")
                if row.source_information:
                    st.write(f"**Source Information:** {row.source_information}")
                left, right = st.columns(2)
                with left:
                    render_relevance(row.evaluation.relevance)
                with right:
                    render_accuracy(row.evaluation.accuracy)
                render_completeness(row.evaluation.completeness)
                render_hallucination(row.evaluation.hallucination)
                render_evidence(row.evaluation.retrieved_evidence)
                render_final_summary(row.evaluation)
    st.download_button(
        "Download batch results (CSV)", batch_evaluator.results_to_csv(report),
        "batch_evaluation_results.csv", "text/csv",
    )
    render_pdf_export(
        dashboard_analytics.records_from_batch_report(
            report,
            batch_id=st.session_state.get(BATCH_ID_STATE_KEY, "current-batch"),
            batch_label=st.session_state.get(BATCH_LABEL_STATE_KEY, "Current batch"),
        ),
        scope="Most recent batch run",
        key="batch",
        file_name="batch_evaluation_report.pdf",
    )
def render_pdf_export(records, scope, key, file_name):
    """M4.2 - build a PDF from exactly the records that are on screen."""
    st.markdown("#### PDF Report Export")
    if not records:
        st.info("There is nothing to export yet. Evaluate at least one response first.")
        return
    if not pdf_report.is_pdf_available():
        st.warning(
            "PDF export needs the reportlab package. Install it with: pip install reportlab"
        )
        return
    st.caption(
        f"The report covers the {len(records)} result(s) currently in scope: "
        "metadata, batch summary, verdict counts, average dimension scores, hallucination "
        "statistics, the most frequent issues, improvement recommendations and every "
        "individual result with its agent reasoning, evidence, flagged claims and missing aspects."
    )
    state_key = f"pdf_bytes_{key}"
    if st.button("Generate PDF report", key=f"pdf_button_{key}"):
        try:
            with st.spinner("Building the PDF report from the evaluation results..."):
                st.session_state[state_key] = pdf_report.generate_pdf(records, scope=scope)
        except Exception as exc:
            st.error(f"The PDF report could not be generated: {exc}")
            return
    data = st.session_state.get(state_key)
    if data:
        st.download_button(
            "Download PDF report", data, file_name, "application/pdf", key=f"pdf_download_{key}"
        )
        st.caption(f"Report size: {len(data) / 1024:.0f} KB.")
def load_dashboard_records():
    """Every completed evaluation stored by this application, oldest first."""
    try:
        return dashboard_analytics.records_from_history(database.get_evaluation_records())
    except Exception as exc:
        st.error(f"Stored evaluations could not be read: {exc}")
        return []
def dashboard_filter_controls(records):
    
    options = dashboard_analytics.filter_options(records)
    with st.expander("Filters", expanded=False):
        first, second = st.columns(2)
        with first:
            group_labels = {label: identifier for identifier, label in options["groups"]}
            chosen_groups = st.multiselect(
                "Batch / source", list(group_labels.keys()), default=[]
            )
            chosen_verdicts = st.multiselect("Verdict", options["verdicts"], default=[])
            search_text = st.text_input("Search question, response or ID", value="")
        with second:
            chosen_statuses = st.multiselect(
                "Hallucination status", options["hallucination_statuses"], default=[]
            )
            score_range = st.slider(
                "Weighted overall score range", 0.0, 100.0, (0.0, 100.0), 1.0
            )
            hallucinated_only = st.checkbox("Only responses flagged for hallucination", value=False)
    low, high = score_range
    return dashboard_analytics.DashboardFilters(
        batch_ids=[group_labels[label] for label in chosen_groups] or None,
        verdicts=chosen_verdicts or None,
        hallucination_statuses=chosen_statuses or None,
        min_overall_score=low if low > 0.0 else None,
        max_overall_score=high if high < 100.0 else None,
        search_text=search_text or "",
        hallucinated_only=hallucinated_only,
    )
def render_dashboard_headline(stats):
    total = stats["total_responses"]
    first_row = [
        ("Total Responses", str(total)),
        ("Pass", f"{stats['pass_count']} ({_percent_text(stats['pass_rate_percent'])})"),
        ("Needs Improvement",
         f"{stats['needs_improvement_count']} ({_percent_text(stats['needs_improvement_rate_percent'])})"),
        ("Fail", f"{stats['fail_count']} ({_percent_text(stats['fail_rate_percent'])})"),
    ]
    second_row = [
        ("Average Relevance", _avg_text(stats["average_relevance"])),
        ("Average Accuracy", _avg_text(stats["average_accuracy"])),
        ("Average Completeness", _avg_text(stats["average_completeness"])),
        ("Average Hallucination Score", _avg_text(stats["average_hallucination_score"], 100)),
    ]
    third_row = [
        ("Average Overall Score", _avg_text(stats["average_overall_score"], 100)),
        ("Hallucinated Responses", str(stats["hallucinated_responses"])),
        ("Hallucination Frequency", _percent_text(stats["hallucination_frequency_percent"])),
        ("Batches Covered", str(stats["batches_covered"])),
    ]
    for row in (first_row, second_row, third_row):
        columns = st.columns(4)
        for column, (label, value) in zip(columns, row):
            column.metric(label, value)
def render_dashboard_charts(stats):
    left, right = st.columns(2)
    with left:
        st.markdown("### Dimension Score Chart")
        if stats["dimension_chart"]:
            st.bar_chart(stats["dimension_chart"], x="Dimension", y="Average score (0-100)")
            st.caption("Each dimension on the 0-100 scale the Verdict Agent uses to combine them.")
        else:
            st.info("No dimension scores are available for the selected results.")
    with right:
        st.markdown("### Verdict Distribution")
        st.bar_chart(stats["verdict_distribution"], x="Verdict", y="Count")
        st.caption(
            "Pass "
            f"{_percent_text(stats['pass_rate_percent'])}, Needs Improvement "
            f"{_percent_text(stats['needs_improvement_rate_percent'])}, Fail "
            f"{_percent_text(stats['fail_rate_percent'])}."
        )
    lower_left, lower_right = st.columns(2)
    with lower_left:
        st.markdown("### Judge Score Averages (1-5)")
        if stats["five_point_chart"]:
            st.bar_chart(stats["five_point_chart"], x="Dimension", y="Average score (1-5)")
        else:
            st.info("No judge scores are available for the selected results.")
    with lower_right:
        st.markdown("### Overall Score Distribution")
        st.bar_chart(stats["score_distribution"], x="Score band", y="Count")
def render_quality_trend(stats):
    st.markdown("### Quality Trends Across Batches")
    trend = stats["quality_trend"]
    if len(trend) < 2:
        st.info(
            "A trend needs at least two batches. Run another batch evaluation "
            "to compare quality over time."
        )
    else:
        plottable = [item for item in trend if item["Average overall score"] is not None]
        if plottable:
            st.line_chart(plottable, x="Batch", y="Average overall score")
            st.line_chart(plottable, x="Batch", y="Pass rate (%)")
    st.markdown("### Batch and Combined Statistics")
    table = [
        {
            "Batch": item["batch_label"],
            "Responses": item["total_responses"],
            "Average Overall": item["average_overall_score"],
            "Average Relevance": item["average_relevance"],
            "Average Accuracy": item["average_accuracy"],
            "Average Completeness": item["average_completeness"],
            "Pass Rate (%)": item["pass_rate_percent"],
            "Hallucination Rate (%)": item["hallucination_frequency_percent"],
        }
        for item in stats["batch_statistics"]
    ]
    table.append(
        {
            "Batch": "Combined (all selected)",
            "Responses": stats["total_responses"],
            "Average Overall": stats["average_overall_score"],
            "Average Relevance": stats["average_relevance"],
            "Average Accuracy": stats["average_accuracy"],
            "Average Completeness": stats["average_completeness"],
            "Pass Rate (%)": stats["pass_rate_percent"],
            "Hallucination Rate (%)": stats["hallucination_frequency_percent"],
        }
    )
    st.dataframe(table, use_container_width=True)
def render_frequent_issues(stats):
    st.markdown("### Most Frequent Evaluation Issues")
    issues = stats["frequent_issues"]
    if not issues:
        st.success("No verdict issues, missing aspects or flagged claims were recorded.")
        return
    st.dataframe(
        [
            {
                "Category": item["category"],
                "Issue": item["issue"],
                "Occurrences": item["count"],
                "Share of responses (%)": item["affected_percent"],
                "Example": item["example_record_id"],
            }
            for item in issues
        ],
        use_container_width=True,
    )
def render_record_detail(record):
    st.write(f"**Question:** {record.question}")
    st.write(f"**AI Response:** {record.response}")
    if record.reference_answer:
        st.write(f"**Reference Answer:** {record.reference_answer}")
    if record.source_information:
        st.write(f"**Source Information:** {record.source_information}")
    columns = st.columns(6)
    columns[0].metric("Relevance", _avg_text(record.relevance_score))
    columns[1].metric("Accuracy", _avg_text(record.accuracy_score))
    columns[2].metric("Completeness", _avg_text(record.completeness_score))
    columns[3].metric("Hallucination", record.hallucination_status or "n/a")
    columns[4].metric("Overall", _avg_text(record.overall_score, 100))
    columns[5].metric("Verdict", record.verdict or "n/a")
    for label, section in (
        ("Relevance judge", record.relevance),
        ("Accuracy judge", record.accuracy),
        ("Completeness judge", record.completeness),
        ("Hallucination detector", record.hallucination),
        ("Verdict agent", record.verdict_section),
    ):
        reasoning = section.get("reasoning")
        if reasoning:
            st.write(f"**{label} reasoning:** {reasoning}")
    if record.problem_claims:
        st.write("**Unsupported or contradicted claims:**")
        for claim in record.problem_claims:
            st.markdown(f"- {claim.get('claim_status')}: {claim.get('claim')}")
    if record.missing_aspects:
        st.write("**Missing aspects:**")
        for aspect in record.missing_aspects:
            st.markdown(f"- {aspect}")
    if record.major_issues:
        st.write("**Major issues:**")
        for issue in record.major_issues:
            st.markdown(f"- {issue}")
def render_drill_down(records):
    st.markdown("### Individual Results")
    st.caption("Every line is one stored evaluation. Expand a row to see the agents' own output.")
    st.dataframe(dashboard_analytics.results_table(records), use_container_width=True)
    for record in records:
        verdict_text = f" - {record.verdict}" if record.verdict else ""
        with st.expander(f"{record.record_id}: {record.question[:70]}{verdict_text}"):
            render_record_detail(record)
def render_dashboard():
    page_intro(
        "Overview",
        "Evaluation Scoring Dashboard",
        "Every figure below is calculated from completed evaluation results stored by this "
        "application. Nothing on this page is a placeholder or a fixed value.",
    )
    records = load_dashboard_records()
    if not records:
        st.info(
            "No completed evaluations are stored yet. Run a single evaluation or a batch "
            "evaluation to populate the dashboard."
        )
        return
    filters = dashboard_filter_controls(records)
    filtered = dashboard_analytics.apply_filters(records, filters)
    if filters.is_active():
        st.caption(f"Showing {len(filtered)} of {len(records)} stored results after filtering.")
    if not filtered:
        st.warning("No stored evaluation matches the current filters.")
        return
    stats = dashboard_analytics.compute_statistics(filtered)
    render_dashboard_headline(stats)
    render_dashboard_charts(stats)
    render_quality_trend(stats)
    render_frequent_issues(stats)
    render_drill_down(filtered)
    render_pdf_export(
        filtered,
        scope="Dashboard selection" if filters.is_active() else "All stored evaluations",
        key="dashboard",
        file_name="evaluation_report.pdf",
    )
def render_single_evaluation():
    page_intro("Evaluate", "Single Evaluation", "Assess one AI response with retrieval and all four Milestone 3 judges.")
    st.subheader("Evaluation Input")
    with st.form("evaluation_form"):
        question = st.text_area("Question (required)", placeholder="e.g. What is the capital of France?", height=90)
        ai_response = st.text_area("AI-Generated Response (required)", placeholder="Enter the response to validate.", height=130)
        reference_text = st.text_area("Reference Answer (optional)", placeholder="Leave empty to use retrieved evidence.", height=90)
        submitted = st.form_submit_button("Validate and Evaluate Response")
    if not submitted:
        return
    errors = []
    if not question or not question.strip():
        errors.append("Question is required.")
    if not ai_response or not ai_response.strip():
        errors.append("AI-generated response is required.")
    if errors:
        for message in errors:
            st.error(message)
        return
    reference_answer = reference_text.strip() if reference_text and reference_text.strip() else None
    submission_id = None
    try:
        submission_id = database.save_submission(question, ai_response, reference_answer)
    except Exception as exc:
        st.error(f"Could not store the submission: {exc}")
    try:
        with st.spinner("Retrieving evidence and running all four judges..."):
            evaluation = orchestrator.evaluate_response(question, ai_response, reference_answer)
    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")
        return
    if submission_id is not None:
        try:
            database.save_evaluation_result(submission_id, evaluation)
            st.success(f"Evaluation completed and stored with ID {submission_id}.")
        except Exception as exc:
            st.warning(f"Evaluation completed, but its history record could not be updated: {exc}")
    render_evidence(evaluation.retrieved_evidence)
    st.subheader("Evaluation Judges")
    left, right = st.columns(2)
    with left:
        render_relevance(evaluation.relevance)
    with right:
        render_accuracy(evaluation.accuracy)
    render_completeness(evaluation.completeness)
    render_hallucination(evaluation.hallucination)
    render_final_summary(evaluation)
def render_batch_evaluation():
    page_intro("Evaluate at scale", "Batch Evaluation", "Validate a CSV while preserving every row result, skip reason, and failure reason.")
    st.subheader("Batch Evaluation (CSV upload)")
    st.caption(
        f"Required: {', '.join(batch_evaluator.REQUIRED_COLUMNS)}. Optional: "
        f"{', '.join(batch_evaluator.OPTIONAL_COLUMNS)}. Maximum {batch_evaluator.MAX_ROWS} rows."
    )
    uploaded_csv = st.file_uploader("CSV file", type=["csv"])
    run_batch = st.button("Run Batch Evaluation", type="primary")
    if run_batch and uploaded_csv is None:
        st.error("Please choose a CSV file first.")
    if not (run_batch and uploaded_csv is not None):
        previous = st.session_state.get(BATCH_REPORT_STATE_KEY)
        if previous is not None:
            st.caption("Showing the most recent batch run. Upload a file and run again to replace it.")
            render_batch_report(previous)
        return
    try:
        raw_bytes = uploaded_csv.read()
    except Exception as exc:
        st.error(f"The uploaded file could not be read: {exc}")
        return
    progress_bar = st.progress(0.0)
    progress_note = st.empty()
    def on_progress(completed, total, row_result):
        progress_bar.progress(completed / total if total else 1.0)
        progress_note.caption(f"Evaluated {completed} of {total} valid rows (row {row_result.row_number}: {row_result.status}).")
    try:
        report = batch_evaluator.evaluate_batch(raw_bytes, progress_callback=on_progress)
    except BatchValidationError as exc:
        st.error(f"This file cannot be evaluated: {exc}")
        return
    except Exception as exc:
        st.error(f"Batch evaluation failed: {exc}")
        return
    st.session_state[BATCH_REPORT_STATE_KEY] = report
    label = f"Batch {len(database.get_batches()) + 1}"
    try:
        st.session_state[BATCH_ID_STATE_KEY] = database.save_batch_report(report, label=label)
        st.session_state[BATCH_LABEL_STATE_KEY] = label
    except Exception as exc:
        st.warning(f"The batch results could not be saved to the history: {exc}")
    render_batch_report(report)
def render_history():
    page_intro("Records", "Evaluation History", "Completed evaluations stored locally by the application.")
    rows = database.get_evaluation_history(limit=250)
    if not rows:
        st.info("No completed evaluations are stored yet.")
        return
    st.dataframe([
        {"ID": r["id"], "Question": r["question"], "Score": r["weighted_overall_score"],
         "Verdict": r["verdict"], "Date/time": r["created_at"]} for r in rows
    ], use_container_width=True)
    st.markdown("### View Details")
    for row in rows:
        with st.expander(f"#{row['id']} | {row['verdict']} | {row['question'][:70]}"):
            payload = json.loads(row["evaluation_json"])
            st.write(f"**Question:** {payload['question']}")
            st.write(f"**AI Response:** {payload['response']}")
            st.write(f"**Accuracy:** {payload['accuracy']['score']}/5")
            st.write(f"**Relevance:** {payload['relevance']['score']}/5")
            st.write(f"**Completeness:** {payload['completeness']['score']}/5")
            st.write(f"**Hallucination:** {payload['hallucination']['hallucination_status']}")
            st.write(f"**Detailed reasoning:** {payload['verdict']['reasoning']}")


