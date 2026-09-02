"""
Backend package for the AI Response Validation System (Milestone 1).

This package contains the modular building blocks of the project:
- database.py     : SQLite storage for user submissions
- ingestion.py     : Loading reference datasets (TruthfulQA, SQuAD, or demo fallback)
- preprocessing.py : Cleaning, standardizing, and chunking text
- embeddings.py     : Converting text into vector embeddings
- vector_store.py   : Storing and searching embeddings in ChromaDB
- retrieval.py       : Tying embeddings + vector store together for semantic search
"""
