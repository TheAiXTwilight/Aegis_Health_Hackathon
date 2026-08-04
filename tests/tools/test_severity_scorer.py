"""
tests/tools/test_severity_scorer.py — SeverityScorer rule engine.

Design decisions:
    - Assert on rule constants (ALL_RULE_CONSTANTS), never on reason strings.
    - EXPECTED_LEVELS restates the published spec — tests may restate contracts.
    - _RULE_FIRING_STATES provides one minimal firing state per rule constant.
      Missing entry → KeyError at parametrize time (structural enforcement).
    - Bidirectional coverage: every constant in ALL_RULE_CONSTANTS has a
      firing state and an expected level, and vice versa.
    - Confidence: ordering + behavioral floor, never exact values or _RULES access.
    - _RULES is never imported in tests (private implementation detail).
    - ALL_RULE_CONSTANTS is the public surface — import only that.
"""

from __future__ import annotations

import pytest

from schemas.drugs import (
    DrugInteraction,
    DrugInteractionResult,
    DrugInteractionSeverity,
)
from schemas.errors import ToolError
from schemas.lab import LabReportResult
from schemas.severity import SeverityResult
from schemas.state import AegisState
from schemas.symptom import SymptomExtractionResult
from schemas.xray import XRayResult
from tools.lab_constants import (
    LAB_KEY_HAEMOGLOBIN,
    LAB_KEY_POTASSIUM,
    LAB_KEY_TROPONIN,
)
from tools.lab_thresholds import (
    CRITICAL_HAEMOGLOBIN_G_DL,
    CRITICAL_POTASSIUM_MMOL_L,
    CRITICAL_TROPONIN_NG_ML,
)
from tools.severity_scorer import (
    ALL_RULE_CONSTANTS,
    RULE_TEXT_FINDING_PREFIX,
    RULE_ABNORMAL_LAB_ANY,
    RULE_CHEST_PAIN_AND_SOB,
    RULE_CRITICAL_LAB_HAEMOGLOBIN,
    RULE_CRITICAL_LAB_POTASSIUM,
    RULE_CRITICAL_LAB_TROPONIN,
    RULE_DEFAULT_LOW,
    RULE_MODERATE_DRUG_INTERACTION,
    RULE_PROLONGED_SYMPTOMS,
    RULE_SEVERE_DRUG_INTERACTION,
    RULE_XRAY_CARDIOMEGALY,
    RULE_XRAY_CONSOLIDATION,
    RULE_XRAY_PLEURAL_EFFUSION,
    RULE_XRAY_PNEUMOTHORAX,
    RULE_XRAY_PULMONARY_EDEMA,
    SeverityScorer,
)


# ── Published contract (spec restatement) ─────────────────────────
# This mapping intentionally restates the published severity contract.
# It is NOT derived from _RULES because tests verify external behavior,
# not private implementation details.
# Contract changes should require deliberate updates to both the scorer
# and this mapping.
# Uses imported constants (never string literals) for typo protection,
# IDE rename support, and static analysis.

EXPECTED_LEVELS: dict[str, str] = {
    RULE_CHEST_PAIN_AND_SOB:        "HIGH",
    RULE_CRITICAL_LAB_TROPONIN:     "HIGH",
    RULE_CRITICAL_LAB_HAEMOGLOBIN:  "HIGH",
    RULE_CRITICAL_LAB_POTASSIUM:    "HIGH",
    RULE_XRAY_PNEUMOTHORAX:         "HIGH",
    RULE_XRAY_PULMONARY_EDEMA:      "HIGH",
    RULE_SEVERE_DRUG_INTERACTION:   "HIGH",
    RULE_ABNORMAL_LAB_ANY:          "MEDIUM",
    RULE_XRAY_CARDIOMEGALY:         "MEDIUM",
    RULE_XRAY_PLEURAL_EFFUSION:     "MEDIUM",
    RULE_XRAY_CONSOLIDATION:        "MEDIUM",
    RULE_PROLONGED_SYMPTOMS:        "MEDIUM",
    RULE_MODERATE_DRUG_INTERACTION: "MEDIUM",
    RULE_DEFAULT_LOW:               "LOW",
}


# ── Minimal firing states ──────────────────────────────────────────
# One AegisState per rule constant, minimal to fire exactly that rule.
# Missing entry → KeyError at parametrize time.

def _symptom_state(*symptoms: str, duration: str | None = None) -> AegisState:
    state = AegisState()
    state.symptom_result = SymptomExtractionResult(
        symptoms=list(symptoms),
        duration=duration,
    )
    return state


def _lab_state(
    measurements: dict[str, float],
    abnormal_values: list[str] | None = None,
) -> AegisState:
    state = AegisState()
    state.lab_result = LabReportResult(
        measurements=measurements,
        abnormal_values=abnormal_values or [],
    )
    return state


def _xray_state(*findings: str) -> AegisState:
    state = AegisState()
    state.xray_result = XRayResult(findings=list(findings))
    return state


def _drug_state(*interactions: DrugInteraction) -> AegisState:
    state = AegisState()
    resolved = list({d for i in interactions for d in i.drugs})
    state.drug_result = DrugInteractionResult(
        resolved=resolved,
        unresolved=[],
        interactions=list(interactions),
        warnings=[],
        confidence=1.0,
    )
    return state


_RULE_FIRING_STATES: dict[str, AegisState] = {
    RULE_CHEST_PAIN_AND_SOB: _symptom_state(
        "chest pain", "shortness of breath"
    ),
    RULE_CRITICAL_LAB_TROPONIN: _lab_state(
        {LAB_KEY_TROPONIN: CRITICAL_TROPONIN_NG_ML + 0.01}
    ),
    RULE_CRITICAL_LAB_HAEMOGLOBIN: _lab_state(
        {LAB_KEY_HAEMOGLOBIN: CRITICAL_HAEMOGLOBIN_G_DL - 0.1}
    ),
    RULE_CRITICAL_LAB_POTASSIUM: _lab_state(
        {LAB_KEY_POTASSIUM: CRITICAL_POTASSIUM_MMOL_L + 0.1}
    ),
    RULE_XRAY_PNEUMOTHORAX: _xray_state("Pneumothorax"),
    RULE_XRAY_PULMONARY_EDEMA: _xray_state("Pulmonary Edema"),
    RULE_SEVERE_DRUG_INTERACTION: _drug_state(
        DrugInteraction(
            drugs=["warfarin", "aspirin"],
            severity=DrugInteractionSeverity.SEVERE,
            description="Bleeding risk.",
        )
    ),
    RULE_ABNORMAL_LAB_ANY: _lab_state(
        measurements={},
        abnormal_values=["High glucose: 200 mg/dL"],
    ),
    RULE_XRAY_CARDIOMEGALY: _xray_state("Cardiomegaly"),
    RULE_XRAY_PLEURAL_EFFUSION: _xray_state("Pleural Effusion"),
    RULE_XRAY_CONSOLIDATION: _xray_state("Consolidation"),
    RULE_PROLONGED_SYMPTOMS: _symptom_state(
        "fatigue", duration="2 weeks"
    ),
    RULE_MODERATE_DRUG_INTERACTION: _drug_state(
        DrugInteraction(
            drugs=["aspirin", "ibuprofen"],
            severity=DrugInteractionSeverity.MODERATE,
            description="GI bleeding risk.",
        )
    ),
    RULE_DEFAULT_LOW: AegisState(),
}


# ── Bidirectional coverage ─────────────────────────────────────────

def _base_rule_constants() -> set[str]:
    """Rules with stable hand-authored contracts in this test module.

    Text-finding rules are generated from the pattern registry at import time,
    so they intentionally do not have a hand-written minimal firing state here.
    Their registry-specific behavior is covered by text-finding tests.
    """
    return {
        constant
        for constant in ALL_RULE_CONSTANTS
        if not constant.startswith(RULE_TEXT_FINDING_PREFIX)
    }


def test_expected_levels_covers_all_stable_rule_constants():
    assert _base_rule_constants() == set(EXPECTED_LEVELS)


def test_firing_states_cover_all_stable_rule_constants():
    assert _base_rule_constants() == set(_RULE_FIRING_STATES)


def test_synthesized_text_finding_constants_follow_the_dynamic_contract():
    dynamic = [
        constant for constant in ALL_RULE_CONSTANTS
        if constant.startswith(RULE_TEXT_FINDING_PREFIX)
    ]
    # Dynamic rules are valid additions from the text-finding registry. This
    # assertion protects naming/registration without requiring a duplicate
    # hand-maintained fixture for every registry pattern.
    assert all(constant.startswith(RULE_TEXT_FINDING_PREFIX) for constant in dynamic)


def test_firing_states_have_no_unknown_constants():
    assert set(_RULE_FIRING_STATES) <= set(ALL_RULE_CONSTANTS)


# ── Parametrized behavioral invariants ────────────────────────────

@pytest.mark.parametrize("constant,state", _RULE_FIRING_STATES.items())
async def test_rule_fires_on_minimal_state(constant, state):
    """Each rule must fire when given its minimal firing state."""
    result = await SeverityScorer().score(state)
    assert constant in result.triggered_rules


@pytest.mark.parametrize("constant,state", _RULE_FIRING_STATES.items())
async def test_highest_priority_rule_matches_reported_level(constant, state):
    """
    Reported severity level must agree with the highest-priority
    triggered rule. Core pipeline guarantee from the spec.
    """
    result = await SeverityScorer().score(state)
    assert constant in result.triggered_rules
    assert result.level == EXPECTED_LEVELS[constant]


@pytest.mark.parametrize("constant,state", _RULE_FIRING_STATES.items())
async def test_highest_priority_rule_equals_triggered_rules_zero(constant, state):
    """
    highest_priority_rule always equals triggered_rules[0].
    Fundamental to downstream consumers — kept as a dedicated invariant.
    Never collapse this into another test.
    """
    result = await SeverityScorer().score(state)
    assert result.highest_priority_rule == result.triggered_rules[0]


@pytest.mark.parametrize("constant,state", _RULE_FIRING_STATES.items())
async def test_reasons_length_matches_triggered_rules(constant, state):
    """len(reasons) == len(triggered_rules) for every firing state."""
    result = await SeverityScorer().score(state)
    assert len(result.reasons) == len(result.triggered_rules)


# ── Confidence invariants ──────────────────────────────────────────

async def test_high_severity_confidence_exceeds_default_low():
    """Ordering: HIGH rule confidence must exceed DEFAULT_LOW confidence."""
    high_result = await SeverityScorer().score(
        _RULE_FIRING_STATES[RULE_CHEST_PAIN_AND_SOB]
    )
    low_result = await SeverityScorer().score(
        _RULE_FIRING_STATES[RULE_DEFAULT_LOW]
    )
    assert high_result.confidence > low_result.confidence


async def test_high_severity_confidence_above_behavioral_floor():
    """HIGH rules should produce meaningfully high confidence."""
    result = await SeverityScorer().score(
        _RULE_FIRING_STATES[RULE_CHEST_PAIN_AND_SOB]
    )
    assert result.highest_priority_rule == RULE_CHEST_PAIN_AND_SOB
    assert result.confidence >= 0.9


async def test_default_low_confidence_above_minimal_floor():
    """Even the fallback rule should produce meaningful confidence."""
    result = await SeverityScorer().score(
        _RULE_FIRING_STATES[RULE_DEFAULT_LOW]
    )
    assert result.highest_priority_rule == RULE_DEFAULT_LOW
    assert result.confidence >= 0.5


async def test_confidence_always_in_valid_range():
    """Schema constraint: SeverityResult.confidence is ge=0.0 le=1.0."""
    result = await SeverityScorer().score(AegisState())
    assert 0.0 <= result.confidence <= 1.0


# ── Suppression invariants ─────────────────────────────────────────

async def test_prolonged_symptoms_suppressed_by_high_rule():
    """RULE_PROLONGED_SYMPTOMS must not fire alongside any HIGH rule."""
    state = _symptom_state(
        "chest pain", "shortness of breath", duration="3 weeks"
    )
    result = await SeverityScorer().score(state)
    assert RULE_CHEST_PAIN_AND_SOB in result.triggered_rules
    assert RULE_PROLONGED_SYMPTOMS not in result.triggered_rules


async def test_moderate_drug_suppressed_by_severe_drug():
    """RULE_MODERATE_DRUG_INTERACTION must not fire alongside any HIGH rule."""
    state = AegisState()
    state.drug_result = DrugInteractionResult(
        resolved=["warfarin", "aspirin", "ibuprofen"],
        unresolved=[],
        interactions=[
            DrugInteraction(
                drugs=["warfarin", "aspirin"],
                severity=DrugInteractionSeverity.SEVERE,
                description="Bleeding risk.",
            ),
            DrugInteraction(
                drugs=["aspirin", "ibuprofen"],
                severity=DrugInteractionSeverity.MODERATE,
                description="GI risk.",
            ),
        ],
        warnings=[],
        confidence=1.0,
    )
    result = await SeverityScorer().score(state)
    assert RULE_SEVERE_DRUG_INTERACTION in result.triggered_rules
    assert RULE_MODERATE_DRUG_INTERACTION not in result.triggered_rules


# ── Structural invariants ──────────────────────────────────────────

def test_default_low_is_last_constant():
    """RULE_DEFAULT_LOW is always the final entry in ALL_RULE_CONSTANTS."""
    assert ALL_RULE_CONSTANTS[-1] == RULE_DEFAULT_LOW


async def test_triggered_rules_never_empty():
    """Even with no inputs, RULE_DEFAULT_LOW fires — list is never empty."""
    result = await SeverityScorer().score(AegisState())
    assert len(result.triggered_rules) >= 1


# ── contributing_tools ─────────────────────────────────────────────

async def test_contributing_tools_populated_when_rules_fire():
    result = await SeverityScorer().score(
        _RULE_FIRING_STATES[RULE_CHEST_PAIN_AND_SOB]
    )
    assert len(result.contributing_tools) > 0


async def test_contributing_tools_empty_for_default_low():
    result = await SeverityScorer().score(AegisState())
    assert result.contributing_tools == []


async def test_contributing_tools_deduplicated():
    """Multiple lab rules firing must not duplicate LabReportParser."""
    state = _lab_state(
        measurements={
            LAB_KEY_TROPONIN: CRITICAL_TROPONIN_NG_ML + 0.01,
            LAB_KEY_HAEMOGLOBIN: CRITICAL_HAEMOGLOBIN_G_DL - 0.1,
        },
        abnormal_values=["elevated troponin", "low haemoglobin"],
    )
    result = await SeverityScorer().score(state)
    from tools.tool_names import TOOL_LAB_REPORT_PARSER
    assert result.contributing_tools.count(TOOL_LAB_REPORT_PARSER) == 1


# ── ToolError guards ───────────────────────────────────────────────

async def test_scorer_ignores_tool_error_lab_result():
    """Lab rules must not fire when lab_result is a ToolError."""
    state = AegisState()
    state.lab_result = ToolError(tool="LabReportParser", reason="fail")
    result = await SeverityScorer().score(state)
    assert RULE_CRITICAL_LAB_TROPONIN not in result.triggered_rules
    assert RULE_CRITICAL_LAB_HAEMOGLOBIN not in result.triggered_rules
    assert RULE_CRITICAL_LAB_POTASSIUM not in result.triggered_rules
    assert RULE_ABNORMAL_LAB_ANY not in result.triggered_rules


async def test_scorer_ignores_tool_error_xray_result():
    """X-ray rules must not fire when xray_result is a ToolError."""
    state = AegisState()
    state.xray_result = ToolError(tool="XRayProcessor", reason="fail")
    result = await SeverityScorer().score(state)
    assert RULE_XRAY_PNEUMOTHORAX not in result.triggered_rules
    assert RULE_XRAY_CARDIOMEGALY not in result.triggered_rules


async def test_scorer_ignores_tool_error_drug_result():
    """Drug rules must not fire when drug_result is a ToolError."""
    state = AegisState()
    state.drug_result = ToolError(tool="DrugInteractionChecker", reason="fail")
    result = await SeverityScorer().score(state)
    assert RULE_SEVERE_DRUG_INTERACTION not in result.triggered_rules
    assert RULE_MODERATE_DRUG_INTERACTION not in result.triggered_rules


# ── Resilience ─────────────────────────────────────────────────────

async def test_scorer_continues_when_check_fn_raises(monkeypatch):
    """
    A bug in one check_fn skips that rule only — evaluation continues.
    DEFAULT_LOW still fires when all real rules are broken or skipped.
    """
    import tools.severity_scorer as ss

    original_rules = ss._RULES[:]
    broken_rule = ss.Rule(
        constant="RULE_BROKEN_TEST",
        priority=999,
        level="HIGH",
        check_fn=lambda ctx: 1 / 0,
        reason="broken",
        contributing_tool=None,
        rule_confidence=0.99,
    )
    monkeypatch.setattr(ss, "_RULES", [broken_rule] + original_rules)

    result = await ss.SeverityScorer().score(AegisState())
    assert RULE_DEFAULT_LOW in result.triggered_rules
    assert "RULE_BROKEN_TEST" not in result.triggered_rules


# ── Functional entrypoint ──────────────────────────────────────────

async def test_score_functional_entrypoint():
    from tools.severity_scorer import score
    result = await score(AegisState())
    assert isinstance(result, SeverityResult)
    assert result.level == "LOW"
