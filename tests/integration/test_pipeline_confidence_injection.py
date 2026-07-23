"""
tests/integration/test_pipeline_confidence_injection.py

Verifies confidence injection after ReportGenerator completes.

Mocking strategy:
    The test has one external HTTP dependency — httpx.AsyncClient —
    with two distinct usage patterns:
        - ReportGenerator    → client.stream(...)
        - ExecutionPlanner   → client.post(...)
        - SymptomExtractor   → uses _call_ollama (mocked separately)

    _UnifiedClient implements both stream() and post() so a single
    httpx.AsyncClient patch satisfies the interface required by both
    pipeline callers.

    Other patches:
        - tools.symptom_extractor._call_ollama → AsyncMock
        - tools.medical_rag_search._embed      → fixed numpy vector
        - tools.medical_rag_search._chroma_query → []
        - populated_state fake file paths cleared so LabReportParser
          and XRayProcessor are not invoked against non-existent files.

    All production modules import httpx via `import httpx`, so a single
    patch on httpx.AsyncClient propagates to all three call sites.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

import tools.medical_rag_search as rag_module
from agents.pipeline import AegisPipeline
from tools.report_generator import REQUIRED_SECTIONS


# ── Helpers ────────────────────────────────────────────────────────

def _full_report_text() -> str:
    return "\n".join(
        f"{header}\nsome content here"
        for header in REQUIRED_SECTIONS
    )


def _ollama_ndjson(text: str) -> list[str]:
    return [
        json.dumps({"response": text, "done": False}),
        json.dumps({"response": "", "done": True}),
    ]


def _fake_embedding() -> np.ndarray:
    vec = np.ones(384, dtype=np.float32)
    return vec / np.linalg.norm(vec)


# ── Fake httpx components ──────────────────────────────────────────

class _FakeStream:
    """Async context manager mimicking httpx streaming response."""

    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakePostResponse:
    """Mimics httpx response for non-streaming post()."""

    def raise_for_status(self):
        pass

    def json(self):
        return {
            "response": json.dumps({
                "use_rag":   True,
                "reasoning": "test plan",
            })
        }


class _UnifiedClient:
    """
    Single fake httpx.AsyncClient implementing the full interface
    required by all pipeline callers:

        ReportGenerator   uses .stream(...)
        ExecutionPlanner  uses .post(...)

    A single patch on httpx.AsyncClient returns this client for every
    httpx.AsyncClient(...) call across the pipeline.
    """

    def __init__(self, report_lines: list[str]):
        self._report_lines = report_lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, *args, **kwargs):
        return _FakeStream(self._report_lines)

    async def post(self, *args, **kwargs):
        return _FakePostResponse()


# ── Fixture preparation ────────────────────────────────────────────

def _prepare_state(populated_state):
    """
    Clear fake file paths so LabReportParser and XRayProcessor are
    not invoked. The pre-injected *_result fields remain on state.
    """
    populated_state.lab_pdf_path    = None
    populated_state.xray_image_path = None
    return populated_state


# ── Patch context ──────────────────────────────────────────────────

class _AllPatched:
    """
    Single entry point for all external dependency patches.

    One patch covers httpx.AsyncClient via a unified client that
    satisfies both stream() and post() usage patterns. All three
    production modules (report_generator, execution_planner,
    symptom_extractor) import httpx via `import httpx`, so a single
    patch propagates to all call sites.
    """

    def __init__(self, report_text: str):
        self._text = report_text
        self._patches: list = []

    def __enter__(self):
        report_lines = _ollama_ndjson(self._text)

        def _client_factory(*args, **kwargs):
            return _UnifiedClient(report_lines)

        self._patches = [
            # Single shared httpx.AsyncClient patch
            patch("httpx.AsyncClient", new=_client_factory),

            # SymptomExtractor internal helper
            patch(
                "tools.symptom_extractor._call_ollama",
                new=AsyncMock(return_value=json.dumps({
                    "symptoms":            ["chest pain", "shortness of breath"],
                    "duration":            "3 days",
                    "severity_indicators": ["severe"],
                    "medical_entities":    ["chest"],
                    "negations":           [],
                })),
            ),

            # MedicalRAGSearch embedding + retrieval
            patch(
                "tools.medical_rag_search._embed",
                return_value=_fake_embedding(),
            ),
            patch(
                "tools.medical_rag_search._chroma_query",
                return_value=[],
            ),
        ]
        for p in self._patches:
            p.__enter__()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.__exit__(*exc)


# ── RAG singleton reset ────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_rag_singletons():
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


# ── Tests ──────────────────────────────────────────────────────────

async def test_confidence_injected_into_report(populated_state):
    state = _prepare_state(populated_state)
    with _AllPatched(_full_report_text()):
        pipeline = AegisPipeline()
        async for _ in pipeline.run(state):
            pass
    assert state.report is not None
    assert state.report.confidence > 0.0


async def test_confidence_is_float_in_range(populated_state):
    state = _prepare_state(populated_state)
    with _AllPatched(_full_report_text()):
        async for _ in AegisPipeline().run(state):
            pass
    assert state.report is not None
    c = state.report.confidence
    assert isinstance(c, float)
    assert 0.0 <= c <= 1.0


async def test_confidence_above_zero_for_populated_state(populated_state):
    """
    With all external dependencies mocked and fake paths cleared,
    all tools succeed → coverage=1.0, success_rate=1.0 → confidence >= 0.5.
    """
    state = _prepare_state(populated_state)
    with _AllPatched(_full_report_text()):
        async for _ in AegisPipeline().run(state):
            pass
    assert state.report is not None
    assert state.report.confidence >= 0.5


async def test_pipeline_complete_set_after_run(populated_state):
    state = _prepare_state(populated_state)
    with _AllPatched(_full_report_text()):
        async for _ in AegisPipeline().run(state):
            pass
    assert state.pipeline_complete is True


async def test_pipeline_timing_recorded(populated_state):
    state = _prepare_state(populated_state)
    with _AllPatched(_full_report_text()):
        async for _ in AegisPipeline().run(state):
            pass
    assert state.pipeline_start_ms is not None
    assert state.pipeline_end_ms is not None
    assert state.pipeline_end_ms >= state.pipeline_start_ms


async def test_current_tool_is_none_after_run(populated_state):
    state = _prepare_state(populated_state)
    with _AllPatched(_full_report_text()):
        async for _ in AegisPipeline().run(state):
            pass
    assert state.current_tool is None


async def test_report_generator_in_tools_run(populated_state):
    state = _prepare_state(populated_state)
    with _AllPatched(_full_report_text()):
        async for _ in AegisPipeline().run(state):
            pass
    from tools.tool_names import TOOL_REPORT_GENERATOR
    assert TOOL_REPORT_GENERATOR in state.tools_run


async def test_tools_run_and_tools_failed_disjoint(populated_state):
    state = _prepare_state(populated_state)
    with _AllPatched(_full_report_text()):
        async for _ in AegisPipeline().run(state):
            pass
    overlap = set(state.tools_run) & set(state.tools_failed)
    assert overlap == set()