"""
tests/tools/test_execution_planner.py — ExecutionPlanner.

Tests LLM-adjacent infrastructure only. No real Ollama calls.
httpx.AsyncClient mocked at HTTP boundary (same pattern as
test_report_generator.py).

The planner decides ONLY use_rag. Tests reflect this — no assertions
on mandatory tool fields (which do not exist in ExecutionPlan).

Coverage:
    - Valid JSON → ExecutionPlan with use_rag respected
    - use_rag=True and use_rag=False both propagated correctly
    - Malformed JSON attempt 1 → attempt 2 tried
    - Both attempts fail → ToolError(fatal=False)
    - HTTP failure attempt 1 → attempt 2 tried
    - Both HTTP failures → ToolError(fatal=False)
    - stream=False in Ollama request
    - temperature=0.0 in Ollama request
    - _make_fallback_plan: use_rag=True always
    - _make_fallback_plan: is_fallback=True, was_repaired=False
    - _make_fallback_plan: satisfies model_validator invariant
    - Returned plan: was_repaired=False, is_fallback=False on success
    - Schema version = 1
    - ToolError tool attribution
"""

from __future__ import annotations

import json


from schemas.errors import ToolError
from schemas.plan import ExecutionPlan
from schemas.state import AegisState
from tools import execution_planner as ep
from tools.execution_planner import ExecutionPlanner, _make_fallback_plan
from tools.tool_names import TOOL_EXECUTION_PLANNER


# ── Fake HTTP infrastructure ──────────────────────────────────────

class _FakeResponse:
    def __init__(self, body: dict):
        self._body = body

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._body


class _FakeClient:
    def __init__(self, response_body: dict):
        self._body = response_body
        self.last_post_json: dict = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url: str, json: dict | None = None, **kwargs):
        self.last_post_json = json or {}
        return _FakeResponse(self._body)


class _FailingClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, *args, **kwargs):
        raise RuntimeError("connection refused")


def _valid_response(use_rag: bool = True) -> dict:
    return {
        "response": json.dumps({
            "use_rag":   use_rag,
            "reasoning": "test reasoning",
        })
    }


def _patch(monkeypatch, factory):
    monkeypatch.setattr(ep.httpx, "AsyncClient", factory)


# ── Happy path ────────────────────────────────────────────────────

async def test_valid_json_returns_execution_plan(monkeypatch):
    _patch(monkeypatch, lambda **kw: _FakeClient(_valid_response()))
    result = await ExecutionPlanner().run(AegisState(raw_symptoms_text="chest pain"))
    assert isinstance(result, ExecutionPlan)


async def test_use_rag_true_respected(monkeypatch):
    _patch(monkeypatch, lambda **kw: _FakeClient(_valid_response(use_rag=True)))
    result = await ExecutionPlanner().run(AegisState(raw_symptoms_text="chest pain"))
    assert isinstance(result, ExecutionPlan)
    assert result.use_rag is True


async def test_use_rag_false_respected(monkeypatch):
    _patch(monkeypatch, lambda **kw: _FakeClient(_valid_response(use_rag=False)))
    result = await ExecutionPlanner().run(AegisState(raw_symptoms_text="mild headache"))
    assert isinstance(result, ExecutionPlan)
    assert result.use_rag is False


async def test_reasoning_propagated(monkeypatch):
    _patch(monkeypatch, lambda **kw: _FakeClient(_valid_response()))
    result = await ExecutionPlanner().run(AegisState(raw_symptoms_text="chest pain"))
    assert isinstance(result, ExecutionPlan)
    assert result.reasoning == "test reasoning"


async def test_success_plan_not_fallback(monkeypatch):
    _patch(monkeypatch, lambda **kw: _FakeClient(_valid_response()))
    result = await ExecutionPlanner().run(AegisState(raw_symptoms_text="chest pain"))
    assert isinstance(result, ExecutionPlan)
    assert result.is_fallback is False


async def test_success_plan_not_repaired(monkeypatch):
    _patch(monkeypatch, lambda **kw: _FakeClient(_valid_response()))
    result = await ExecutionPlanner().run(AegisState(raw_symptoms_text="chest pain"))
    assert isinstance(result, ExecutionPlan)
    assert result.was_repaired is False


async def test_schema_version_is_one(monkeypatch):
    _patch(monkeypatch, lambda **kw: _FakeClient(_valid_response()))
    result = await ExecutionPlanner().run(AegisState(raw_symptoms_text="chest pain"))
    assert isinstance(result, ExecutionPlan)
    assert result.schema_version == 1


# ── Retry on malformed JSON ───────────────────────────────────────

async def test_malformed_attempt1_retries_attempt2(monkeypatch):
    call_count = 0

    class _Selective:
        async def __aenter__(self): return self
        async def __aexit__(self, *e): return False
        async def post(self, *a, **kw):
            nonlocal call_count
            call_count += 1
            body = (
                {"response": "not json {"}
                if call_count == 1
                else _valid_response()
            )
            return _FakeResponse(body)

    _patch(monkeypatch, lambda **kw: _Selective())
    result = await ExecutionPlanner().run(AegisState(raw_symptoms_text="chest pain"))
    assert isinstance(result, ExecutionPlan)
    assert call_count == 2


async def test_both_malformed_returns_tool_error(monkeypatch):
    _patch(monkeypatch, lambda **kw: _FakeClient({"response": "not json"}))
    result = await ExecutionPlanner().run(AegisState(raw_symptoms_text="chest pain"))
    assert isinstance(result, ToolError)


# ── HTTP failures ─────────────────────────────────────────────────

async def test_http_failure_attempt1_retries_attempt2(monkeypatch):
    call_count = 0

    class _Selective:
        async def __aenter__(self): return self
        async def __aexit__(self, *e): return False
        async def post(self, *a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("refused")
            return _FakeResponse(_valid_response())

    _patch(monkeypatch, lambda **kw: _Selective())
    result = await ExecutionPlanner().run(AegisState(raw_symptoms_text="chest pain"))
    assert isinstance(result, ExecutionPlan)
    assert call_count == 2


async def test_both_http_failures_returns_tool_error(monkeypatch):
    _patch(monkeypatch, lambda **kw: _FailingClient())
    result = await ExecutionPlanner().run(AegisState(raw_symptoms_text="chest pain"))
    assert isinstance(result, ToolError)


# ── ToolError properties ──────────────────────────────────────────

async def test_tool_error_nonfatal(monkeypatch):
    _patch(monkeypatch, lambda **kw: _FailingClient())
    result = await ExecutionPlanner().run(AegisState(raw_symptoms_text="chest pain"))
    assert isinstance(result, ToolError)
    assert result.fatal is False


async def test_tool_error_tool_attribution(monkeypatch):
    _patch(monkeypatch, lambda **kw: _FailingClient())
    result = await ExecutionPlanner().run(AegisState(raw_symptoms_text="chest pain"))
    assert isinstance(result, ToolError)
    assert result.tool == TOOL_EXECUTION_PLANNER


# ── Ollama request properties ─────────────────────────────────────

async def test_request_non_streaming(monkeypatch):
    captured: dict = {}

    class _Cap:
        async def __aenter__(self): return self
        async def __aexit__(self, *e): return False
        async def post(self, url, json=None, **kw):
            captured["json"] = json
            return _FakeResponse(_valid_response())

    _patch(monkeypatch, lambda **kw: _Cap())
    await ExecutionPlanner().run(AegisState(raw_symptoms_text="chest pain"))
    assert captured["json"]["stream"] is False


async def test_request_temperature_zero(monkeypatch):
    captured: dict = {}

    class _Cap:
        async def __aenter__(self): return self
        async def __aexit__(self, *e): return False
        async def post(self, url, json=None, **kw):
            captured["json"] = json
            return _FakeResponse(_valid_response())

    _patch(monkeypatch, lambda **kw: _Cap())
    await ExecutionPlanner().run(AegisState(raw_symptoms_text="chest pain"))
    assert captured["json"]["options"]["temperature"] == 0.0


# ── _make_fallback_plan ───────────────────────────────────────────

def test_fallback_use_rag_always_true():
    assert _make_fallback_plan(AegisState()).use_rag is True


def test_fallback_use_rag_true_even_no_inputs():
    assert _make_fallback_plan(AegisState()).use_rag is True


def test_fallback_is_fallback_true():
    assert _make_fallback_plan(AegisState()).is_fallback is True


def test_fallback_was_repaired_false():
    assert _make_fallback_plan(AegisState()).was_repaired is False


def test_fallback_satisfies_model_validator():
    """is_fallback=True + was_repaired=False must not raise."""
    plan = _make_fallback_plan(AegisState())
    assert plan.is_fallback is True
    assert plan.was_repaired is False


def test_fallback_reasoning_nonempty():
    assert len(_make_fallback_plan(AegisState()).reasoning) > 0


def test_fallback_schema_version():
    assert _make_fallback_plan(AegisState()).schema_version == 1