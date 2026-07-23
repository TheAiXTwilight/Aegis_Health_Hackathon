"""
tests/tools/test_medical_rag_search.py — MedicalRAGSearch.

Mocking strategy:
    The ONNX singleton (_onnx_session) is reset to None before each
    test via autouse fixture, preventing a failed load from poisoning
    subsequent tests with a False sentinel.

    _embed() is patched to return a fixed numpy vector. Since _embed
    is a module-level function called directly by run(), this intercepts
    the real call path.

    _chroma_query() and _faiss_query() are patched at the module level
    to return controlled passage lists without requiring real indexes.

    Tests verify: query construction, retrieval logic, result shaping,
    ChromaDB→FAISS fallback, and error handling.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch

import tools.medical_rag_search as rag_module
from schemas.errors import ToolError
from schemas.rag import RAGPassage, RAGSearchResult
from schemas.state import AegisState
from schemas.symptom import SymptomExtractionResult
from tools.medical_rag_search import MedicalRAGSearch


# ── Singleton reset ────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_rag_singletons():
    """
    Reset module-level singletons before each test.

    Resets both the _attempted flag and the cached value for each
    singleton. Without resetting _attempted, a failed load in one test
    would cause all subsequent tests to skip the load entirely — even
    when the load function is patched to succeed.

    Phase 4 pattern: _attempted: bool + _value: T | None
    (Decision 94 — replaces the T | None | bool sentinel anti-pattern).
    """
    original = (
        rag_module._onnx_attempted,
        rag_module._onnx_session,
        rag_module._chroma_attempted,
        rag_module._chroma_collection,
        rag_module._faiss_attempted,
        rag_module._faiss_index,
        rag_module._faiss_docs,
    )

    rag_module._onnx_attempted    = False
    rag_module._onnx_session      = None
    rag_module._chroma_attempted  = False
    rag_module._chroma_collection = None
    rag_module._faiss_attempted   = False
    rag_module._faiss_index       = None
    rag_module._faiss_docs        = None

    yield

    (
        rag_module._onnx_attempted,
        rag_module._onnx_session,
        rag_module._chroma_attempted,
        rag_module._chroma_collection,
        rag_module._faiss_attempted,
        rag_module._faiss_index,
        rag_module._faiss_docs,
    ) = original


# ── Helpers ────────────────────────────────────────────────────────

def _fake_embedding() -> np.ndarray:
    vec = np.ones(384, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def _fake_passages() -> list[RAGPassage]:
    return [
        RAGPassage(
            text="Chest pain with elevated troponin may indicate ACS.",
            source="AHA Guidelines",
            citation="AHA-ACS-2024",
        ),
        RAGPassage(
            text="Troponin elevation is a marker of myocardial injury.",
            source="MedlinePlus",
            citation="MEDLINE-TROP-001",
        ),
    ]


async def _search(state: AegisState) -> RAGSearchResult | ToolError:
    return await MedicalRAGSearch().run(state)


# ── Empty state — no embedding needed ─────────────────────────────

async def test_empty_state_returns_rag_result():
    """Empty query short-circuits before embedding."""
    state = AegisState()
    result = await _search(state)
    assert isinstance(result, RAGSearchResult)


async def test_empty_state_retrieval_successful():
    state = AegisState()
    result = await _search(state)
    assert result.retrieval_successful is True


async def test_empty_state_query_used_is_string():
    state = AegisState()
    result = await _search(state)
    assert isinstance(result.query_used, str)


async def test_empty_state_returns_empty_passages():
    state = AegisState()
    result = await _search(state)
    assert result.passages == []


# ── Relevant query — embedding + retrieval ─────────────────────────

async def test_chest_pain_query_retrieves_passages():
    state = AegisState(raw_symptoms_text="chest pain troponin")
    with patch("tools.medical_rag_search._embed", return_value=_fake_embedding()), \
         patch("tools.medical_rag_search._chroma_query", return_value=_fake_passages()):
        result = await _search(state)
    assert isinstance(result, RAGSearchResult)
    assert len(result.passages) > 0


async def test_relevant_query_sets_retrieval_successful():
    state = AegisState(raw_symptoms_text="chest pain")
    with patch("tools.medical_rag_search._embed", return_value=_fake_embedding()), \
         patch("tools.medical_rag_search._chroma_query", return_value=_fake_passages()):
        result = await _search(state)
    assert result.retrieval_successful is True


async def test_retrieved_passages_have_text_source_citation():
    state = AegisState(raw_symptoms_text="chest pain troponin")
    with patch("tools.medical_rag_search._embed", return_value=_fake_embedding()), \
         patch("tools.medical_rag_search._chroma_query", return_value=_fake_passages()):
        result = await _search(state)
    for passage in result.passages:
        assert passage.text != ""
        assert passage.source != ""
        assert passage.citation != ""


async def test_citations_list_matches_passage_citations():
    state = AegisState(raw_symptoms_text="chest pain troponin")
    with patch("tools.medical_rag_search._embed", return_value=_fake_embedding()), \
         patch("tools.medical_rag_search._chroma_query", return_value=_fake_passages()):
        result = await _search(state)
    for citation in result.citations:
        assert any(p.citation == citation for p in result.passages)


async def test_irrelevant_query_retrieval_successful():
    """Zero passages returned — mechanism succeeded, found nothing."""
    state = AegisState(raw_symptoms_text="zzz_no_match_zzz_xyz")
    with patch("tools.medical_rag_search._embed", return_value=_fake_embedding()), \
         patch("tools.medical_rag_search._chroma_query", return_value=[]):
        result = await _search(state)
    assert isinstance(result, RAGSearchResult)
    assert result.retrieval_successful is True


# ── ToolError guards ───────────────────────────────────────────────

async def test_tool_error_symptom_result_does_not_crash():
    state = AegisState(raw_symptoms_text="chest pain")
    state.symptom_result = ToolError(tool="SymptomExtractor", reason="fail")
    with patch("tools.medical_rag_search._embed", return_value=_fake_embedding()), \
         patch("tools.medical_rag_search._chroma_query", return_value=_fake_passages()):
        result = await _search(state)
    assert isinstance(result, RAGSearchResult)


async def test_tool_error_lab_result_does_not_crash():
    state = AegisState(raw_symptoms_text="elevated troponin")
    state.lab_result = ToolError(tool="LabReportParser", reason="fail")
    with patch("tools.medical_rag_search._embed", return_value=_fake_embedding()), \
         patch("tools.medical_rag_search._chroma_query", return_value=_fake_passages()):
        result = await _search(state)
    assert isinstance(result, RAGSearchResult)


# ── Structured result expands query ───────────────────────────────

async def test_symptom_result_symptoms_expand_query():
    state = AegisState()
    state.symptom_result = SymptomExtractionResult(
        symptoms=["chest pain", "troponin elevated"],
    )
    with patch("tools.medical_rag_search._embed", return_value=_fake_embedding()), \
         patch("tools.medical_rag_search._chroma_query", return_value=_fake_passages()):
        result = await _search(state)
    assert isinstance(result, RAGSearchResult)
    assert len(result.passages) >= 1


# ── Embedding failure ──────────────────────────────────────────────

async def test_embed_failure_returns_tool_error():
    state = AegisState(raw_symptoms_text="chest pain")
    with patch("tools.medical_rag_search._embed", return_value=None):
        result = await _search(state)
    assert isinstance(result, ToolError)
    assert result.fatal is False
    assert result.tool == "MedicalRAGSearch"


# ── ChromaDB → FAISS fallback ──────────────────────────────────────

async def test_chroma_failure_tries_faiss():
    state = AegisState(raw_symptoms_text="chest pain")
    faiss_passages = [
        RAGPassage(text="FAISS result", source="FAISS", citation="FAISS-001")
    ]
    with patch("tools.medical_rag_search._embed", return_value=_fake_embedding()), \
         patch("tools.medical_rag_search._chroma_query",
               side_effect=RuntimeError("chroma down")), \
         patch("tools.medical_rag_search._faiss_query",
               return_value=faiss_passages):
        result = await _search(state)
    assert isinstance(result, RAGSearchResult)
    assert result.retrieval_successful is True
    assert result.passages[0].source == "FAISS"


async def test_both_retrievers_fail_returns_tool_error():
    state = AegisState(raw_symptoms_text="chest pain")
    with patch("tools.medical_rag_search._embed", return_value=_fake_embedding()), \
         patch("tools.medical_rag_search._chroma_query",
               side_effect=RuntimeError("chroma down")), \
         patch("tools.medical_rag_search._faiss_query",
               side_effect=RuntimeError("faiss down")):
        result = await _search(state)
    assert isinstance(result, ToolError)
    assert result.fatal is False


# ── Query content ──────────────────────────────────────────────────

async def test_query_used_reflects_raw_symptoms():
    state = AegisState(raw_symptoms_text="severe headache")
    with patch("tools.medical_rag_search._embed", return_value=_fake_embedding()), \
         patch("tools.medical_rag_search._chroma_query", return_value=[]):
        result = await _search(state)
    assert "severe headache" in result.query_used


# ── Schema compliance ──────────────────────────────────────────────

async def test_rag_result_schema_version():
    state = AegisState()
    result = await _search(state)
    assert result.schema_version == "1.0"


# ── Functional entrypoint ──────────────────────────────────────────

async def test_search_functional_entrypoint():
    from tools.medical_rag_search import search
    state = AegisState(raw_symptoms_text="fever cough")
    with patch("tools.medical_rag_search._embed", return_value=_fake_embedding()), \
         patch("tools.medical_rag_search._chroma_query", return_value=[]):
        result = await search(state)
    assert isinstance(result, RAGSearchResult)