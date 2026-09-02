"""
app.py
------
Milestone 1 Streamlit demo for the "AI Response Validation System with
Hallucination Detection Assistance" project.

WHAT THIS DEMO SHOWS (Milestone 1 scope only):
    USER INPUT -> INPUT VALIDATION -> SUBMISSION STORAGE ->
    QUESTION EMBEDDING -> CHROMADB SEMANTIC SEARCH ->
    RELEVANT REFERENCE EVIDENCE -> DISPLAY EVIDENCE

WHAT THIS DEMO DOES NOT DO YET (future milestones):
    Relevance/Accuracy/Completeness/Hallucination judge agents, a verdict
    agent, final scoring, batch CSV evaluation, or analytics dashboards.
"""

import logging
import streamlit as st

from backend import database, retrieval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Response Validation System - Milestone 1",
    page_icon="🔎",
    layout="centered",
)


# ---------------------------------------------------------------------------
# One-time setup (cached across reruns within a session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def initialize_system():
    """Initialize the SQLite database and populate the vector store once."""
    database.init_db()
    try:
        added = retrieval.build_knowledge_base_if_needed(use_huggingface=True, limit_per_dataset=50)
        used_fallback = False
    except Exception as exc:  # noqa: BLE001
        logger.warning("Knowledge base build failed, will rely on demo fallback: %s", exc)
        added = 0
        used_fallback = True
    return {"chunks_added": added, "used_fallback": used_fallback}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🔎 AI Response Validation System")
st.caption("Milestone 1 Demo — Evaluation Input, Retrieval Foundation & Storage")

with st.spinner("Preparing knowledge base (first run may take a moment)..."):
    try:
        init_info = initialize_system()
        st.success(
            f"System ready. Knowledge base check complete "
            f"({init_info['chunks_added']} new chunks indexed this session)."
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to initialize the system: {exc}")
        st.stop()

st.markdown("---")
st.subheader("1️⃣ Evaluation Input")

with st.form("evaluation_form"):
    question = st.text_area(
        "Question *",
        placeholder="e.g. What is the capital of France?",
        height=80,
    )
    ai_response = st.text_area(
        "AI-Generated Response *",
        placeholder="e.g. The capital of France is Paris.",
        height=100,
    )
    reference_text = st.text_area(
        "Optional Reference Answer / Source Document",
        placeholder="(Optional) Paste a known-correct answer or source text here.",
        height=80,
    )
    submitted = st.form_submit_button("✅ Validate Response")

if submitted:
    # --- INPUT VALIDATION ---------------------------------------------------
    errors = []
    if not question or not question.strip():
        errors.append("Question is required.")
    if not ai_response or not ai_response.strip():
        errors.append("AI-generated response is required.")

    if errors:
        for err in errors:
            st.error(f"❌ {err}")
    else:
        # --- SUBMISSION STORAGE ---------------------------------------------
        try:
            submission_id = database.save_submission(question, ai_response, reference_text)
            st.success(f"✅ Submission stored (ID: {submission_id}).")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to store submission: {exc}")
            st.stop()

        # --- QUESTION EMBEDDING + CHROMADB SEMANTIC SEARCH -------------------
        st.markdown("---")
        st.subheader("2️⃣ Retrieved Reference Evidence")
        try:
            with st.spinner("Embedding question and searching the knowledge base..."):
                evidence = retrieval.retrieve_evidence(question, top_k=3)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Retrieval failed: {exc}")
            evidence = []

        if not evidence:
            st.warning("No relevant reference evidence was found for this question.")
        else:
            for i, item in enumerate(evidence, start=1):
                with st.container(border=True):
                    st.markdown(f"**Evidence #{i}**  ·  Source: `{item['source']}`  ·  Distance: `{item['distance']:.4f}`")
                    if item.get("question"):
                        st.caption(f"Related dataset question: {item['question']}")
                    st.write(item["text"])

        st.markdown("---")
        st.info(
            "ℹ️ **Milestone 1 scope note:** This demo retrieves relevant reference "
            "evidence only. It does NOT yet judge relevance/accuracy/completeness, "
            "detect hallucinations, or produce a verdict/score. Those are planned "
            "for later milestones."
        )

# ---------------------------------------------------------------------------
# Sidebar: recent submissions (handy for the demo / debugging)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📋 Recent Submissions")
    try:
        recent = database.get_recent_submissions(limit=5)
        st.caption(f"Total stored: {database.count_submissions()}")
        if not recent:
            st.write("No submissions yet.")
        else:
            for row in recent:
                with st.expander(f"#{row['id']} — {row['question'][:40]}..."):
                    st.write(f"**Question:** {row['question']}")
                    st.write(f"**AI Response:** {row['ai_response']}")
                    if row.get("reference_text"):
                        st.write(f"**Reference:** {row['reference_text']}")
                    st.caption(f"Stored at: {row['created_at']} UTC")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load recent submissions: {exc}")

    st.markdown("---")
    st.caption("Milestone 1 · Foundation Demo")
