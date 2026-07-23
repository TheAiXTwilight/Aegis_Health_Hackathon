"""
tests/scenarios/test_scenario_a.py — Scenario A: Full modality clean run.

Uses text inputs only (no audio, no PDF — placeholder tools reject binary).
Exercises the full pipeline against a real Ollama instance.

The pipeline executes exactly once via a module-scoped async fixture.
All test functions read from the single resulting (state, report_text)
tuple — one Ollama call for the entire module.

Inputs:
    raw_symptoms_text  — minimal symptom phrase, no clinical risk terms
    medications_raw    — two medications with no known interaction
    xray_findings_raw  — normal findings

Expected:
    severity                = LOW
    highest_priority_rule   = RULE_DEFAULT_LOW
    triggered_rules         = [RULE_DEFAULT_LOW]
    confidence              > 0.9
    report present          (sections validated by ReportGenerator)
    pipeline_complete       = True
    pipeline_duration_ms    > 0 and < 60 000
    no tool failures
    no truncation

Marked @pytest.mark.ollama — requires live Ollama with aegis-llama loaded.
Run with: pytest tests/scenarios/test_scenario_a.py -v -m ollama
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from agents.pipeline import AegisPipeline
from schemas.state import AegisState
from tools.severity_scorer import RULE_DEFAULT_LOW
from tools.tool_names import TOOL_REPORT_GENERATOR


# ── Module-scoped fixture — one pipeline execution ─────────────────

@pytest_asyncio.fixture(scope="module")
async def scenario_a():
    """
    Run Scenario A once. All tests in this module share the result.

    Symptom text is deliberately minimal — no chest pain, no shortness
    of breath, no duration in weeks or months, no terms that map to
    any MEDIUM or HIGH severity rule. RULE_DEFAULT_LOW must fire with
    certainty regardless of placeholder extractor behaviour.

    Returns (state, report_text).
    """
    state = AegisState(
        raw_symptoms_text="Mild headache since yesterday.",
        medications_raw=["Metformin 500mg", "Lisinopril 10mg"],
        xray_findings_raw=["Normal / No significant findings"],
    )
    pipeline = AegisPipeline()
    tokens: list[str] = []
    async for token in pipeline.run(state):
        tokens.append(token)
    return state, "".join(tokens)


# ── Severity ───────────────────────────────────────────────────────

@pytest.mark.ollama
def test_a_severity_is_low(scenario_a):
    state, _ = scenario_a
    assert state.severity_result is not None
    assert state.severity_result.level == "LOW"


@pytest.mark.ollama
def test_a_highest_priority_rule_is_default_low(scenario_a):
    state, _ = scenario_a
    assert state.severity_result is not None
    assert state.severity_result.highest_priority_rule == RULE_DEFAULT_LOW


@pytest.mark.ollama
def test_a_only_default_low_triggered(scenario_a):
    state, _ = scenario_a
    assert state.severity_result is not None
    assert state.severity_result.triggered_rules == [RULE_DEFAULT_LOW]


# ── Report ─────────────────────────────────────────────────────────

@pytest.mark.ollama
def test_a_confidence_above_threshold(scenario_a):
    """
    Confidence threshold for text-only Scenario A.

    The spec target of > 0.9 assumes full modality with audio and PDF.
    Placeholder tools reject binary input, so Scenario A maxes at 3/5
    modalities (symptoms, medications, rag). Formula:
        0.4 × (3/5) + 0.4 × 1.0 + 0.2 × 1.0 = 0.84

    Phase 3 will restore the > 0.9 threshold once Faster-Whisper and
    real PDF parsing land.
    """
    state, _ = scenario_a
    assert state.report is not None
    assert state.report.confidence >= 0.84


@pytest.mark.ollama
def test_a_report_present(scenario_a):
    """
    Report existence is the authoritative section check.
    ReportGenerator raises FatalPipelineError if any required section
    is missing — state.report is not None implies all six sections
    passed validation.
    """
    state, _ = scenario_a
    assert state.report is not None


@pytest.mark.ollama
def test_a_report_severity_matches_scorer(scenario_a):
    state, _ = scenario_a
    assert state.report is not None
    assert state.severity_result is not None
    assert state.report.severity == state.severity_result.level


@pytest.mark.ollama
def test_a_report_text_contains_summary(scenario_a):
    """Smoke-check on streamed token content."""
    _, report_text = scenario_a
    assert "### Summary" in report_text


# ── Pipeline lifecycle ─────────────────────────────────────────────

@pytest.mark.ollama
def test_a_pipeline_complete(scenario_a):
    state, _ = scenario_a
    assert state.pipeline_complete is True


@pytest.mark.ollama
def test_a_pipeline_duration_recorded(scenario_a):
    state, _ = scenario_a
    assert state.pipeline_start_ms is not None
    assert state.pipeline_end_ms is not None
    duration_ms = state.pipeline_end_ms - state.pipeline_start_ms
    assert duration_ms > 0
    assert duration_ms < 60_000, (
        f"Pipeline took {duration_ms:.0f}ms — exceeded 60s budget"
    )


@pytest.mark.ollama
def test_a_no_tool_failures(scenario_a):
    state, _ = scenario_a
    assert state.tools_failed == [], (
        f"Unexpected tool failures: {state.tools_failed}"
    )


@pytest.mark.ollama
def test_a_report_generator_in_tools_run(scenario_a):
    state, _ = scenario_a
    assert TOOL_REPORT_GENERATOR in state.tools_run


@pytest.mark.ollama
def test_a_tools_run_and_failed_disjoint(scenario_a):
    state, _ = scenario_a
    overlap = set(state.tools_run) & set(state.tools_failed)
    assert overlap == set()


# ── Truncation ─────────────────────────────────────────────────────

@pytest.mark.ollama
def test_a_no_truncation(scenario_a):
    """Minimal input must not trigger truncation flags."""
    state, _ = scenario_a
    assert state.core_fields_truncated is False
    assert state.enrichment_fields_truncated is False