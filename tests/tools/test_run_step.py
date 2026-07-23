"""
tests/tools/test_run_step.py — AegisPipeline._run_step orchestration.

Tests the central step helper in isolation. AegisPipeline is instantiated
real (no mocks), but tool_fn callables are lightweight async lambdas or
small coroutine factories — no actual tools are invoked.

Coverage targets:
    - Success path: result returned, tools_run appended, duration recorded
    - Non-fatal ToolError: tools_failed appended, result returned, no raise
    - Fatal ToolError: tools_failed appended, FatalPipelineError raised
    - Unhandled exception: tools_failed appended, None returned, no raise
    - current_tool lifecycle: set before call, cleared in finally
    - step_durations_ms: always populated regardless of outcome
    - tools_run ∩ tools_failed = ∅ invariant across all paths
"""

from __future__ import annotations

import pytest

from agents.pipeline import AegisPipeline
from schemas.errors import FatalPipelineError, ToolError
from schemas.state import AegisState


# ── Factories ──────────────────────────────────────────────────────

def _make_pipeline() -> AegisPipeline:
    return AegisPipeline()


def _tool_fn_ok(value: object):
    async def _fn(state: AegisState):
        return value
    return _fn


def _tool_fn_error(error: ToolError):
    async def _fn(state: AegisState):
        return error
    return _fn


def _tool_fn_raises(exc: Exception):
    async def _fn(state: AegisState):
        raise exc
    return _fn


# ── Success path ───────────────────────────────────────────────────

async def test_run_step_success_path(empty_state):
    """
    On success: result returned, tools_run updated, duration recorded,
    current_tool cleared, tools_failed untouched.
    """
    pipeline = _make_pipeline()
    result = await pipeline._run_step("TestTool", _tool_fn_ok("hello"), empty_state)

    assert result == "hello"
    assert "TestTool" in empty_state.tools_run
    assert "TestTool" not in empty_state.tools_failed
    assert "TestTool" in empty_state.step_durations_ms
    assert empty_state.step_durations_ms["TestTool"] >= 0.0
    assert empty_state.current_tool is None


# ── Non-fatal ToolError ────────────────────────────────────────────

async def test_run_step_nonfatal_error_path(empty_state, tool_error_nonfatal):
    """
    On non-fatal ToolError: error returned, tools_failed updated,
    no raise, duration recorded, current_tool cleared, tools_run untouched.
    """
    pipeline = _make_pipeline()
    result = await pipeline._run_step(
        "TestTool", _tool_fn_error(tool_error_nonfatal), empty_state
    )

    assert isinstance(result, ToolError)
    assert "TestTool" in empty_state.tools_failed
    assert "TestTool" not in empty_state.tools_run
    assert "TestTool" in empty_state.step_durations_ms
    assert empty_state.current_tool is None


# ── Fatal ToolError ────────────────────────────────────────────────

async def test_run_step_fatal_error_path(empty_state, tool_error_fatal):
    """
    On fatal ToolError: FatalPipelineError raised, tools_failed updated,
    duration recorded, current_tool cleared, tools_run untouched.
    The FatalPipelineError carries the originating ToolError.
    """
    pipeline = _make_pipeline()

    with pytest.raises(FatalPipelineError) as exc_info:
        await pipeline._run_step(
            "TestTool", _tool_fn_error(tool_error_fatal), empty_state
        )

    assert exc_info.value.tool_error is tool_error_fatal
    assert "TestTool" in empty_state.tools_failed
    assert "TestTool" not in empty_state.tools_run
    assert "TestTool" in empty_state.step_durations_ms
    assert empty_state.current_tool is None


# ── Unhandled exception ────────────────────────────────────────────

async def test_run_step_unhandled_exception_path(empty_state):
    """
    On unhandled exception: None returned, tools_failed updated,
    no re-raise, duration recorded, current_tool cleared, tools_run untouched.
    """
    pipeline = _make_pipeline()
    result = await pipeline._run_step(
        "TestTool",
        _tool_fn_raises(RuntimeError("boom")),
        empty_state,
    )

    assert result is None
    assert "TestTool" in empty_state.tools_failed
    assert "TestTool" not in empty_state.tools_run
    assert "TestTool" in empty_state.step_durations_ms
    assert empty_state.current_tool is None


# ── Invariant: tools_run ∩ tools_failed = ∅ ───────────────────────

async def test_run_step_invariant_disjoint_on_success(empty_state):
    pipeline = _make_pipeline()
    await pipeline._run_step("A", _tool_fn_ok(1), empty_state)
    assert set(empty_state.tools_run) & set(empty_state.tools_failed) == set()


async def test_run_step_invariant_disjoint_on_nonfatal(
    empty_state, tool_error_nonfatal
):
    pipeline = _make_pipeline()
    await pipeline._run_step(
        "A", _tool_fn_error(tool_error_nonfatal), empty_state
    )
    assert set(empty_state.tools_run) & set(empty_state.tools_failed) == set()


async def test_run_step_invariant_disjoint_on_exception(empty_state):
    pipeline = _make_pipeline()
    await pipeline._run_step(
        "A", _tool_fn_raises(RuntimeError()), empty_state
    )
    assert set(empty_state.tools_run) & set(empty_state.tools_failed) == set()


# ── current_tool set during execution ─────────────────────────────

async def test_run_step_sets_current_tool_during_execution(empty_state):
    """
    current_tool is set to the step name while tool_fn executes.
    Captured at call time by the tool_fn itself.
    """
    pipeline = _make_pipeline()
    captured: list[str | None] = []

    async def _capturing_fn(state: AegisState):
        captured.append(state.current_tool)
        return "done"

    await pipeline._run_step("MyTool", _capturing_fn, empty_state)
    assert captured == ["MyTool"]


# ── Multiple steps accumulate correctly ───────────────────────────

async def test_run_step_multiple_steps_accumulate(empty_state):
    """
    Sequential steps accumulate into tools_run, tools_failed,
    and step_durations_ms independently without cross-contamination.
    """
    pipeline = _make_pipeline()
    await pipeline._run_step("ToolA", _tool_fn_ok(1), empty_state)
    await pipeline._run_step("ToolB", _tool_fn_ok(2), empty_state)
    await pipeline._run_step(
        "ToolC",
        _tool_fn_error(ToolError(tool="c", reason="fail", fatal=False)),
        empty_state,
    )

    assert empty_state.tools_run == ["ToolA", "ToolB"]
    assert empty_state.tools_failed == ["ToolC"]
    assert set(empty_state.step_durations_ms.keys()) == {
        "ToolA", "ToolB", "ToolC"
    }