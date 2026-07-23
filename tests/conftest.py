"""
tests/conftest.py — Shared fixtures for the Aegis Health test suite.

Lean by design. Only fixtures with proven cross-module need live here.
Specialised fixtures are added incrementally as later modules require them.

All fixtures build real AegisState and real schema instances — no mocks.
British spelling throughout (haemoglobin), matching tools/lab_constants.py.

Fixture categories
──────────────────
Bare state         : empty_state
ToolError blocks   : tool_error_fatal · tool_error_nonfatal
Schema blocks      : severity_low · severity_high · symptom_result
                     lab_result · xray_result · rag_with_passages
                     rag_empty · drug_severe · voice_result
Composite          : populated_state (all results present, no errors)

Note on fixture confidence values
──────────────────────────────────
severity_low and severity_high use representative confidence values
that are intentionally independent of production calibration constants.
Tests that need to verify exact calibration should use the scorer
directly, not read from fixtures.

Note on voice_result
──────────────────────────────────
voice_result is shared across multiple test modules
(test_voice_transcriber.py, test_symptom_extractor.py).
It is not used in test_report_generator.py but lives here rather than
in a module-local fixture to avoid duplication across test files.
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
from schemas.rag import RAGPassage, RAGSearchResult
from schemas.severity import SeverityResult
from schemas.state import AegisState
from schemas.symptom import SymptomExtractionResult
from schemas.voice import VoiceTranscriptionResult
from schemas.xray import XRayResult
from tools.lab_constants import LAB_KEY_HAEMOGLOBIN, LAB_KEY_POTASSIUM


# ── Bare state ────────────────────────────────────────────────────

@pytest.fixture
def empty_state() -> AegisState:
    """A fresh AegisState with no inputs and no results."""
    return AegisState()


# ── ToolError building blocks ─────────────────────────────────────

@pytest.fixture
def tool_error_fatal() -> ToolError:
    """A fatal ToolError for testing pipeline halt paths."""
    return ToolError(
        tool="test_tool",
        reason="fatal test error",
        fatal=True,
    )


@pytest.fixture
def tool_error_nonfatal() -> ToolError:
    """A non-fatal ToolError for testing pipeline continuation paths."""
    return ToolError(
        tool="test_tool",
        reason="non-fatal test error",
        fatal=False,
    )


# ── Individual schema building blocks ─────────────────────────────

@pytest.fixture
def severity_low() -> SeverityResult:
    """
    A valid LOW SeverityResult (the DEFAULT_LOW fallback shape produced
    by SeverityScorer when no real rule fires).

    Representative fixture value.
    Intentionally independent of production calibration.
    """
    return SeverityResult(
        level="LOW",
        confidence=0.75,
        triggered_rules=["RULE_DEFAULT_LOW"],
        highest_priority_rule="RULE_DEFAULT_LOW",
        reasons=["No high-risk rules triggered."],
        contributing_tools=[],
    )


@pytest.fixture
def severity_high() -> SeverityResult:
    """
    A valid HIGH SeverityResult with one fired rule.

    Representative fixture value.
    Intentionally independent of production calibration.
    """
    return SeverityResult(
        level="HIGH",
        confidence=0.95,
        triggered_rules=["RULE_CHEST_PAIN_AND_SOB"],
        highest_priority_rule="RULE_CHEST_PAIN_AND_SOB",
        reasons=["Chest pain with shortness of breath detected."],
        contributing_tools=["SymptomExtractor"],
    )


@pytest.fixture
def symptom_result() -> SymptomExtractionResult:
    """
    A populated symptom result.

    duration is a string per the actual schema — there is no
    duration_days field. SeverityScorer's _check_prolonged_symptoms
    reads sym.duration as a substring search for "week"/"month".
    """
    return SymptomExtractionResult(
        symptoms=["chest pain", "shortness of breath"],
        duration="3 days",
        severity_indicators=["severe"],
        medical_entities=["chest", "heart"],
        negations=["no fever"],
    )


@pytest.fixture
def lab_result() -> LabReportResult:
    """
    A populated lab result using British canonical keys.

    Contains one ABNORMAL value (potassium 5.8 > 5.5 mmol/L threshold).
    extra_measurements contains "crp" which is intentionally unrecognised
    by the canonical key set — exercises the extra_measurements path in
    LabReportParser and ensures unknown keys are preserved, not dropped.
    """
    return LabReportResult(
        abnormal_values=[
            "High potassium: 5.8 mmol/L (threshold > 5.5)",
        ],
        measurements={
            LAB_KEY_HAEMOGLOBIN: 13.5,
            LAB_KEY_POTASSIUM: 5.8,
        },
        extra_measurements={
            "crp": 12.0,  # intentionally unrecognised — exercises extra_measurements
        },
    )


@pytest.fixture
def xray_result() -> XRayResult:
    """A populated X-ray result with one positive checklist finding."""
    return XRayResult(
        findings=["Cardiomegaly"],
        free_text="Mild enlargement noted.",
    )


@pytest.fixture
def rag_with_passages() -> RAGSearchResult:
    """A RAG result with one retrieved passage."""
    return RAGSearchResult(
        passages=[
            RAGPassage(
                text=(
                    "Chest pain with elevated troponin may indicate "
                    "acute coronary syndrome."
                ),
                source="AHA Guidelines",
                citation="AHA-ACS-2024",
            )
        ],
        citations=["AHA-ACS-2024"],
        query_used="chest pain",
        retrieval_successful=True,
    )


@pytest.fixture
def rag_empty() -> RAGSearchResult:
    """
    A RAG result that ran successfully but retrieved nothing.

    retrieval_successful=True because the mechanism worked correctly —
    it simply found no relevant passages. This is distinct from a
    ToolError, which indicates the mechanism itself failed.
    """
    return RAGSearchResult(
        passages=[],
        citations=[],
        query_used="obscure query",
        retrieval_successful=True,
    )


@pytest.fixture
def drug_severe() -> DrugInteractionResult:
    """A drug result containing one severe interaction."""
    return DrugInteractionResult(
        resolved=["warfarin", "aspirin"],
        unresolved=[],
        interactions=[
            DrugInteraction(
                drugs=["warfarin", "aspirin"],
                severity=DrugInteractionSeverity.SEVERE,
                description=(
                    "Warfarin + Aspirin significantly increases bleeding risk."
                ),
            )
        ],
        warnings=["1 potential drug interaction(s) detected."],
        confidence=1.0,
    )


@pytest.fixture
def voice_result() -> VoiceTranscriptionResult:
    """
    A populated voice transcription result.

    Shared across test_voice_transcriber.py and test_symptom_extractor.py.
    Not used in test_report_generator.py but defined here to avoid
    duplication across modules — consistent with the cross-module sharing
    principle for conftest.py fixtures.
    """
    return VoiceTranscriptionResult(
        transcript="chest pain for three days",
    )


# ── Composite populated state ─────────────────────────────────────

@pytest.fixture
def populated_state(
    severity_low: SeverityResult,
    symptom_result: SymptomExtractionResult,
    lab_result: LabReportResult,
    xray_result: XRayResult,
    rag_with_passages: RAGSearchResult,
    drug_severe: DrugInteractionResult,
) -> AegisState:
    """
    An AegisState with every tool result populated, no ToolError,
    no truncation. Represents a clean full-modality run ready for
    ReportGenerator.

    Input fields are set to non-None values so that confidence
    calculations (active_modalities / 5) reflect a full-modality run.

    severity_result uses severity_low (RULE_DEFAULT_LOW) because
    populated_state represents a clean baseline run — not a HIGH
    severity clinical presentation. Tests that need HIGH severity
    should build their own state or override severity_result directly:

        populated_state.severity_result = severity_high
    """
    state = AegisState(
        raw_symptoms_text="chest pain and shortness of breath for 3 days",
        lab_pdf_path="/tmp/fake_lab.txt",
        xray_image_path="/tmp/fake_xray.jpg",
        medications_raw=["warfarin", "aspirin"],
    )
    state.symptom_result = symptom_result
    state.lab_result = lab_result
    state.xray_result = xray_result
    state.rag_result = rag_with_passages
    state.drug_result = drug_severe
    state.severity_result = severity_low
    return state