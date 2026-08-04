"""Tests for the current deterministic ReportGenerator.

The generator no longer calls Ollama/HTTP for report text. It builds a stable
multi-section report from AegisState and streams its finished sections.
"""
from __future__ import annotations

import pytest

from schemas.errors import FatalPipelineError, ToolError
from schemas.state import AegisState
from tools import report_generator as rg
from tools.report_generator import (
    DISCLAIMER,
    REQUIRED_SECTIONS,
    ReportGenerator,
    _assemble_prompt,
    _estimate_tokens,
    _validate_sections,
)


def all_sections_text() -> str:
    return "\n".join(f"{section}\ncontent" for section in REQUIRED_SECTIONS)


def test_estimate_tokens_never_returns_less_than_one_and_rounds_up():
    assert _estimate_tokens("") == 1
    assert _estimate_tokens("a") == 1
    assert _estimate_tokens("hello") == 2
    assert _estimate_tokens("12345678") == 2


def test_section_validator_uses_current_seven_section_contract():
    assert _validate_sections(all_sections_text()) == []
    missing = _validate_sections("")
    assert set(missing) == set(REQUIRED_SECTIONS)
    assert len(missing) == len(REQUIRED_SECTIONS) == 7


def test_prompt_assembly_preserves_brace_laden_context_without_formatting_error():
    context = "Patient typed {unsafe-looking} text and {0}."
    prompt = _assemble_prompt("Synthetic disclaimer", context)
    assert context in prompt
    assert "Synthetic disclaimer" in prompt
    assert "%%CONTEXT%%" not in prompt


async def test_run_raises_fatal_error_when_severity_is_missing():
    with pytest.raises(FatalPipelineError):
        async for _ in ReportGenerator().run(AegisState()):
            pass


async def test_run_raises_fatal_error_when_severity_failed(tool_error_fatal):
    state = AegisState()
    state.severity_result = tool_error_fatal
    with pytest.raises(FatalPipelineError):
        async for _ in ReportGenerator().run(state):
            pass


async def test_run_builds_current_required_sections_and_streams_exact_report(monkeypatch, populated_state):
    monkeypatch.setattr(rg, "SECTION_REVEAL_PAUSE_SECONDS", 0)

    chunks = [chunk async for chunk in ReportGenerator().run(populated_state)]

    assert populated_state.report is not None
    report = populated_state.report
    assert "".join(chunks) == report.text
    assert report.severity == populated_state.severity_result.level
    # Current ReportGenerator carries the deterministic scorer confidence;
    # pipeline calibration occurs later and may overwrite it.
    assert report.confidence == populated_state.severity_result.confidence
    assert report.disclaimer == DISCLAIMER
    assert all(section in report.text for section in REQUIRED_SECTIONS)


async def test_run_propagates_citations_only_from_successful_rag(monkeypatch, populated_state, tool_error_nonfatal):
    monkeypatch.setattr(rg, "SECTION_REVEAL_PAUSE_SECONDS", 0)

    async for _ in ReportGenerator().run(populated_state):
        pass
    assert populated_state.report is not None
    assert populated_state.report.citations == populated_state.rag_result.citations

    populated_state.rag_result = tool_error_nonfatal
    populated_state.report = None
    async for _ in ReportGenerator().run(populated_state):
        pass
    assert populated_state.report is not None
    assert populated_state.report.citations == []


async def test_run_keeps_user_text_as_text_not_as_template_code(monkeypatch, severity_low):
    monkeypatch.setattr(rg, "SECTION_REVEAL_PAUSE_SECONDS", 0)
    state = AegisState(raw_symptoms_text="Pain when typing { or } and <tag>.")
    state.severity_result = severity_low

    async for _ in ReportGenerator().run(state):
        pass
    assert state.report is not None
    assert "Pain when typing { or } and <tag>." in state.report.text


async def test_run_continues_when_optional_tool_results_are_nonfatal_errors(monkeypatch, severity_low):
    monkeypatch.setattr(rg, "SECTION_REVEAL_PAUSE_SECONDS", 0)
    state = AegisState(raw_symptoms_text="headache")
    state.severity_result = severity_low
    state.lab_result = ToolError(tool="LabReportParser", reason="unavailable")
    state.xray_result = ToolError(tool="XRayProcessor", reason="unavailable")
    state.rag_result = ToolError(tool="MedicalRAGSearch", reason="unavailable")

    async for _ in ReportGenerator().run(state):
        pass
    assert state.report is not None
    assert all(section in state.report.text for section in REQUIRED_SECTIONS)
