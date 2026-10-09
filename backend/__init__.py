"""
Backend package for the AI Response Validation System.
Milestone 1 modules (unchanged, still the foundation for everything else):
- database.py     : SQLite storage for user submissions
- ingestion.py     : Loading reference datasets (TruthfulQA, SQuAD, or demo fallback)
- preprocessing.py : Cleaning, standardizing, and chunking text
- embeddings.py     : Converting text into vector embeddings
- vector_store.py   : Storing and searching embeddings in ChromaDB
- retrieval.py       : Tying embeddings + vector store together for semantic search
Milestone 2 addition:
- llm_client.py     : Shared LLM wrapper used by the judge agents in the
                      top-level `agents/` package (relevance, accuracy,
                      hallucination). Falls back gracefully to heuristic
                      evaluation when no LLM API key is configured.
"""
