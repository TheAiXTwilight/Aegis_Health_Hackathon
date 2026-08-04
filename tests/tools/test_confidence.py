"""Tests for the current calibrated pipeline-confidence model.

The previous suite asserted the retired coverage/success/truncation formula.
Confidence now combines rule-validator agreement, deterministic-rule strength,
evidence richness, and pipeline health, with a medical ceiling below 1.0.
"""
from __future__ import annotations

import pytest

from schemas.drugs import DrugInteractionResult
from schemas.errors import ToolError
from schemas.lab import LabReportResult
from schemas.severity import SeverityResult
from schemas.state import AegisState
from schemas.symptom import SymptomExtractionResult
from schemas.validation import RuleValidatorResult, ValidationStatus
from schemas.xray import XRayResult
from tools.confidence import calculate_confidence


def build_state(
    *,
    agreement: ValidationStatus | None = ValidationStatus.AGREEMENT,
    rule_confidence: float = 0.8,
    symptoms: bool = False,
    labs: bool = False,
    xray: bool = False,
    medications: bool = False,
    handled: bool = True,
    tools_run: list[str] | None = None,
    tools_failed: list[str] | None = None,
    core_truncated: bool = False,
    enrichment_truncated: bool = False,
) -> AegisState:
    state = AegisState(
        raw_symptoms_text="synthetic symptoms" if symptoms else None,
        lab_pdf_path="/tmp/lab.pdf" if labs else None,
        xray_image_path="/tmp/xray.png" if xray else None,
        medications_raw=["aspirin"] if medications else [],
    )

    if symptoms:
        state.symptom_result = (
            SymptomExtractionResult(symptoms=["headache"])
            if handled else ToolError(tool="SymptomExtractor", reason="failed")
        )
    if labs:
        state.lab_result = (
            LabReportResult(measurements={"glucose": 100.0})
            if handled else ToolError(tool="LabReportParser", reason="failed")
        )
    if xray:
        state.xray_result = (
            XRayResult(findings=["Normal / No significant findings"])
            if handled else ToolError(tool="XRayProcessor", reason="failed")
        )
    if medications:
        state.drug_result = (
            DrugInteractionResult(resolved=["aspirin"], confidence=1.0)
            if handled else ToolError(tool="DrugInteractionChecker", reason="failed")
        )

    state.severity_result = SeverityResult(
        level="LOW",
        confidence=rule_confidence,
        triggered_rules=["RULE_DEFAULT_LOW"],
        highest_priority_rule="RULE_DEFAULT_LOW",
        reasons=["Synthetic test rule."],
        contributing_tools=[],
    )
    if agreement is not None:
        state.rule_validator_result = RuleValidatorResult(
            status=agreement,
            deterministic_level="LOW",
            slm_narrative_level="LOW",
            overridden=agreement == ValidationStatus.OVERRIDE,
        )
    state.tools_run = tools_run if tools_run is not None else ["SyntheticTool"]
    state.tools_failed = tools_failed if tools_failed is not None else []
    state.core_fields_truncated = core_truncated
    state.enrichment_fields_truncated = enrichment_truncated
    return state


def test_confidence_is_bounded_by_zero_and_medical_ceiling():
    empty = build_state(agreement=None, tools_run=[])
    full = build_state(
        agreement=ValidationStatus.AGREEMENT,
        rule_confidence=1.0,
        symptoms=True,
        labs=True,
        xray=True,
        medications=True,
        tools_run=["A", "B", "C", "D"],
    )
    assert 0.0 <= calculate_confidence(empty) <= 0.97
    assert calculate_confidence(full) == pytest.approx(0.97)


def test_agreement_status_orders_confidence_by_validation_trust():
    agreement = calculate_confidence(build_state(agreement=ValidationStatus.AGREEMENT))
    warning = calculate_confidence(build_state(agreement=ValidationStatus.WARNING))
    override = calculate_confidence(build_state(agreement=ValidationStatus.OVERRIDE))
    unavailable = calculate_confidence(build_state(agreement=None))

    assert agreement > unavailable > warning > override


def test_rule_confidence_changes_the_rule_strength_component():
    low_rule = calculate_confidence(build_state(rule_confidence=0.4))
    high_rule = calculate_confidence(build_state(rule_confidence=0.95))
    assert high_rule > low_rule


def test_each_handled_user_submitted_modality_increases_evidence_score():
    no_evidence = calculate_confidence(build_state())
    symptom_only = calculate_confidence(build_state(symptoms=True))
    symptoms_and_labs = calculate_confidence(build_state(symptoms=True, labs=True))
    all_modalities = calculate_confidence(
        build_state(symptoms=True, labs=True, xray=True, medications=True)
    )

    assert no_evidence < symptom_only < symptoms_and_labs < all_modalities


def test_rag_result_does_not_count_as_a_user_evidence_modality():
    state_without_rag = build_state(symptoms=True)
    state_with_rag = build_state(symptoms=True)
    # RAG may enrich a report but it is not one of the four user-submitted
    # evidence modalities in the current confidence contract.
    state_with_rag.rag_result = object()
    assert calculate_confidence(state_with_rag) == calculate_confidence(state_without_rag)


def test_unhandled_submitted_modality_does_not_receive_evidence_credit():
    handled = calculate_confidence(build_state(symptoms=True, handled=True))
    failed = calculate_confidence(
        build_state(
            symptoms=True,
            handled=False,
            tools_run=[],
            tools_failed=["SymptomExtractor"],
        )
    )
    assert handled > failed


def test_tool_failure_reduces_pipeline_health_component():
    clean = calculate_confidence(build_state(symptoms=True, tools_run=["A", "B"]))
    partial_failure = calculate_confidence(
        build_state(symptoms=True, tools_run=["A"], tools_failed=["B"])
    )
    all_failed = calculate_confidence(
        build_state(symptoms=True, tools_run=[], tools_failed=["A", "B"])
    )
    assert clean > partial_failure > all_failed


def test_enrichment_and_core_truncation_apply_increasing_penalties():
    clean = calculate_confidence(build_state(symptoms=True))
    enrichment = calculate_confidence(build_state(symptoms=True, enrichment_truncated=True))
    core = calculate_confidence(build_state(symptoms=True, core_truncated=True))
    both = calculate_confidence(
        build_state(symptoms=True, core_truncated=True, enrichment_truncated=True)
    )

    assert clean > enrichment > core
    assert both == core


def test_confidence_uses_neutral_defaults_when_validator_or_scorer_is_absent():
    state = AegisState(raw_symptoms_text="synthetic")
    state.symptom_result = SymptomExtractionResult(symptoms=["headache"])
    state.tools_run = ["SymptomExtractor"]
    value = calculate_confidence(state)
    assert 0.0 < value < 0.97


def test_nan_rule_confidence_falls_back_safely():
    # Pydantic correctly rejects NaN in a real SeverityResult. Exercise the
    # defensive helper path with a deliberately malformed legacy-like object.
    from types import SimpleNamespace

    state = build_state()
    state.severity_result = SimpleNamespace(confidence=float("nan"))
    value = calculate_confidence(state)
    assert 0.0 <= value <= 0.97
