"""
rag/retriever.py — Convenience re-export.

The canonical runtime retrieval implementation lives in
tools/medical_rag_search.py, not here, by design: the pipeline's
runtime query path (Step 5, MedicalRAGSearch) intentionally has zero
import-time dependency on the rag/ build package, so a broken or
mid-rebuild rag/ module can never break request-time retrieval.

This module exists only so `from rag.retriever import ...` works for
anyone exploring the rag/ package expecting a retriever module here.
It re-exports the real implementation unchanged.
"""

from __future__ import annotations

from tools.medical_rag_search import (
    MedicalRAGSearch,
    is_index_ready,
    search,
)

__all__ = ["MedicalRAGSearch", "is_index_ready", "search"]
