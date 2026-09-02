# AI Response Validation System with Hallucination Detection Assistance

**Infosys Springboard Project — Milestone 1: Foundation Demo**

## 1. Project Objective

This project aims to build a system that helps validate AI-generated responses
against trustworthy reference information and eventually assist in detecting
hallucinations.

### Milestone 1 Goal

Milestone 1 focuses on building the foundation of the system:

- Collecting user input
- Validating input
- Storing submissions
- Retrieving relevant reference evidence using semantic search

**Note:** Accuracy scoring, judging, and hallucination detection are not part
of Milestone 1. They are planned for future milestones.

---

## 2. System Architecture

```text
Question + AI-Generated Response
              ↓
Evaluation Input Module
              ↓
Backend / API Layer
              ↓
Reference Knowledge Base
              ↓
Data Cleaning & Standardization
              ↓
Chunking
              ↓
Embedding Generation
              ↓
Vector Database (ChromaDB)
              ↓
Semantic Similarity Search / RAG Retrieval
              ↓
Relevant Reference Evidence
              ↓
Display Evidence to User
```

### Knowledge Base Creation

```text
TruthfulQA + SQuAD
        ↓
Data Cleaning
        ↓
Chunking
        ↓
Embedding Generation
        ↓
ChromaDB
```

If the external datasets are unavailable, the system uses the local
`data/demo_dataset.json` fallback dataset.

---

## 3. Project Structure

```text
AI-Response-Validation-System/
│
├── app.py
├── requirements.txt
├── README.md
│
├── backend/
│   ├── database.py
│   ├── ingestion.py
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── retrieval.py
│
├── data/
│   ├── demo_dataset.json
│   └── submissions.db
│
├── chroma_db/
│
└── tests/
    └── test_basic.py
```

---

## 4. Technologies Used

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Backend / API | Python + FastAPI |
| Submission Storage | SQLite |
| Reference Datasets | Hugging Face `datasets` |
| Data Processing | Python + LangChain Text Splitter |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Database | ChromaDB |
| Testing | pytest |

---

## 5. Milestone 1 Functionality

- ✅ Streamlit evaluation interface
- ✅ Input validation
- ✅ Submission storage using SQLite
- ✅ Reference knowledge base
- ✅ Data cleaning and chunking
- ✅ Embedding generation
- ✅ ChromaDB vector storage
- ✅ Semantic similarity search
- ✅ Relevant reference evidence retrieval
- ✅ Evidence display in the UI
- ✅ Modular backend structure
- ✅ Basic unit tests

---

## 6. Future Milestones

The following features are planned for future milestones:

- ❌ Relevance evaluation
- ❌ Accuracy evaluation
- ❌ Completeness evaluation
- ❌ Hallucination detection
- ❌ Final verdict and scoring
- ❌ Advanced analytics
- ❌ Batch evaluation
- ❌ Multi-agent orchestration

---

## 7. Installation

```bash
pip install -r requirements.txt
```

---

## 8. Run the Application

```bash
streamlit run app.py
```

Open the URL shown by Streamlit, usually:

```text
http://localhost:8501
```

---

## 9. Run Tests

```bash
pytest tests/ -v
```

---

## 10. Milestone 1 Summary

Milestone 1 demonstrates the complete foundation:

```text
Input
  ↓
Validation
  ↓
Storage
  ↓
Reference Retrieval
  ↓
Evidence Display
```

The system currently retrieves relevant reference evidence but does not yet
judge the accuracy or detect hallucinations in the AI response.