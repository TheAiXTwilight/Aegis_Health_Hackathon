"""
tests/scenarios/test_scenario_b.py — Scenario B: Partial modality degraded run.

Uses text inputs only. Exercises the high-severity symptom path and
an unresolvable medication.

The pipeline executes exactly once via a module-scoped async fixture.
All test functions read from the single resulting (state, report_text)
tuple — one Ollama call for the entire module.

Inputs:
    raw_symptoms_text  — chest pain with shortness of breath
    medications_raw    — one unresolvable test drug

Expected:
    severity                = HIGH
    highest_priority_rule   = RULE_CHEST_PAIN_AND_SOB
    RULE_CHEST_PAIN_AND_SOB in triggered_rules
    drug_result.unresolved  contains the test drug (lowercased)
    drug_result.confidence  = 0.0
    rag_result              is RAGSearchResult (not ToolError)
    report.confidence       < 1.0
    report present

Marked @pytest.mark.ollama — requires live Ollama with aegis-llama loaded.
Run with: pytest tests/scenarios/test_scenario_b.py -v -m ollama
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from agents.pipeline import AegisPipeline
from schemas.errors import ToolError
from schemas.rag import RAGSearchResult
from schemas.state import AegisState
from tools.severity_scorer import RULE_CHEST_PAIN_AND_SOB

_UNRESOLVABLE_DRUG = "AEGIS_TEST_UNRESOLVABLE_DRUG_XYZ"


# ── Module-scoped fixture — one pipeline execution ─────────────────

@pytest_asyncio.fixture(scope="module")
async def scenario_b():
    """
    Run Scenario B once. All tests in this module share the result.
    Returns (state, report_text).
    """
    state = AegisState(
        raw_symptoms_text="Chest pain for 3 days with shortness of breath.",
        medications_raw=[_UNRESOLVABLE_DRUG],
    )
    pipeline = AegisPipeline()
    tokens: list[str] = []
    async for token in pipeline.run(state):
        tokens.append(token)
    return state, "".join(tokens)


# ── Severity ───────────────────────────────────────────────────────

@pytest.mark.ollama
def test_b_severity_is_high(scenario_b):
    state, _ = scenario_b
    assert state.severity_result is not None
    assert state.severity_result.level == "HIGH"


@pytest.mark.ollama
def test_b_highest_priority_rule(scenario_b):
    state, _ = scenario_b
    assert state.severity_result is not None
    assert state.severity_result.highest_priority_rule == RULE_CHEST_PAIN_AND_SOB


@pytest.mark.ollama
def test_b_chest_pain_sob_in_triggered_rules(scenario_b):
    state, _ = scenario_b
    assert state.severity_result is not None
    assert RULE_CHEST_PAIN_AND_SOB in state.severity_result.triggered_rules


# ── Drug result ────────────────────────────────────────────────────

@pytest.mark.ollama
def test_b_unresolvable_drug_in_unresolved(scenario_b):
    state, _ = scenario_b
    assert state.drug_result is not None
    assert not isinstance(state.drug_result, ToolError)
    assert _UNRESOLVABLE_DRUG.lower() in state.drug_result.unresolved


@pytest.mark.ollama
def test_b_drug_confidence_is_zero(scenario_b):
    state, _ = scenario_b
    assert state.drug_result is not None
    assert not isinstance(state.drug_result, ToolError)
    assert state.drug_result.confidence == 0.0


# ── RAG result ─────────────────────────────────────────────────────

@pytest.mark.ollama
def test_b_rag_result_is_not_tool_error(scenario_b):
    """RAG must succeed (even with zero passages) — not a ToolError."""
    state, _ = scenario_b
    assert isinstance(state.rag_result, RAGSearchResult)


# ── Report ─────────────────────────────────────────────────────────

@pytest.mark.ollama
def test_b_report_present(scenario_b):
    state, _ = scenario_b
    assert state.report is not None


@pytest.mark.ollama
def test_b_confidence_is_one(scenario_b):
    """
    Scenario B confidence under the corrected formula.

    The corrected confidence formula (handled / submitted) does not
    penalise unsubmitted modalities. Scenario B submits symptoms +
    meds + RAG, and all three are handled successfully — the
    unresolvable drug is correctly *handled* by DrugInteractionChecker
    (it produces a structured DrugInteractionResult with unresolved
    drugs), not a ToolError. So coverage = 3/3 = 1.0 and final
    confidence reaches 1.0.

    Drug-level confidence (DrugInteractionResult.confidence = 0.0) is
    surfaced separately and asserted by test_b_drug_confidence_is_zero.
    """
    state, _ = scenario_b
    assert state.report is not None
    assert state.report.confidence == 1.0


@pytest.mark.ollama
def test_b_report_severity_matches_scorer(scenario_b):
    state, _ = scenario_b
    assert state.report is not None
    assert state.severity_result is not None
    assert state.report.severity == state.severity_result.level


@pytest.mark.ollama
def test_b_report_text_contains_summary(scenario_b):
    _, report_text = scenario_b
    assert "### Summary" in report_text


# ── Pipeline lifecycle ─────────────────────────────────────────────

@pytest.mark.ollama
def test_b_pipeline_complete(scenario_b):
    state, _ = scenario_b
    assert state.pipeline_complete is True


@pytest.mark.ollama
def test_b_tools_run_and_failed_disjoint(scenario_b):
    state, _ = scenario_b
    overlap = set(state.tools_run) & set(state.tools_failed)
    assert overlap == set()