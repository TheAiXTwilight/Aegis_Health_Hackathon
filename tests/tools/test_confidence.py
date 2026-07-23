"""
tests/tools/test_confidence.py — calculate_confidence formula.

Formula: confidence = 0.4 * coverage + 0.4 * success_rate + 0.2 * truncation

Coverage signal   (weight 0.4): handled / submitted
                                Only counts submitted modalities.
                                Unsubmitted modalities are not penalised.
                                RAG is always submitted (pipeline always
                                attempts retrieval), guaranteeing
                                submitted >= 1.
Success signal    (weight 0.4): tools_run / (tools_run + tools_failed)
Truncation signal (weight 0.2): 1.0 / 0.7 / 0.5

Severity level does NOT affect confidence.

Coverage semantics:
    A modality is "submitted" when the user provided the corresponding
    input. A modality is "handled" when the corresponding tool produced
    a non-error structured result. Coverage = handled / submitted.

    Tests use _state_with() to control which modalities are submitted
    and inject pre-built results to control which are handled.
"""

from __future__ import annotations

from schemas.drugs import DrugInteractionResult
from schemas.errors import ToolError
from schemas.lab import LabReportResult
from schemas.rag import RAGSearchResult
from schemas.state import AegisState
from schemas.symptom import SymptomExtractionResult
from schemas.xray import XRayResult
from tools.confidence import calculate_confidence


# ── Helpers ────────────────────────────────────────────────────────

def _state_with(
    *,
    symptoms: bool = False,
    lab: bool = False,
    xray: bool = False,
    meds: bool = False,
    rag: bool = False,
    symptoms_handled: bool = True,
    lab_handled: bool = True,
    xray_handled: bool = True,
    meds_handled: bool = True,
    tools_run: list[str] | None = None,
    tools_failed: list[str] | None = None,
    core_truncated: bool = False,
    enrichment_truncated: bool = False,
) -> AegisState:
    """
    Build a state with explicit control over submitted vs handled
    for each modality.

    Submitted is controlled by setting the input field.
    Handled is controlled by injecting a successful result (default)
    or a ToolError (when *_handled=False).

    RAG is always submitted by the pipeline. The `rag` flag controls
    whether the result is a successful RAGSearchResult (rag=True) or
    absent / failed (rag=False).
    """
    state = AegisState(
        raw_symptoms_text="text" if symptoms else None,
        lab_pdf_path="/tmp/lab.pdf" if lab else None,
        xray_image_path="/tmp/xray.jpg" if xray else None,
        medications_raw=["drug"] if meds else [],
    )

    if symptoms:
        state.symptom_result = (
            SymptomExtractionResult(symptoms=["x"])
            if symptoms_handled
            else ToolError(tool="SymptomExtractor", reason="fail")
        )
    if lab:
        state.lab_result = (
            LabReportResult()
            if lab_handled
            else ToolError(tool="LabReportParser", reason="fail")
        )
    if xray:
        state.xray_result = (
            XRayResult(findings=["Cardiomegaly"])
            if xray_handled
            else ToolError(tool="XRayProcessor", reason="fail")
        )
    if meds:
        state.drug_result = (
            DrugInteractionResult(confidence=1.0)
            if meds_handled
            else ToolError(tool="DrugInteractionChecker", reason="fail")
        )

    if rag:
        state.rag_result = RAGSearchResult(
            passages=[],
            citations=[],
            query_used="test",
            retrieval_successful=True,
        )

    state.tools_run = tools_run or []
    state.tools_failed = tools_failed or []
    state.core_fields_truncated = core_truncated
    state.enrichment_fields_truncated = enrichment_truncated
    return state


# ── Output range ───────────────────────────────────────────────────

def test_confidence_in_range_zero_to_one():
    state = _state_with()
    c = calculate_confidence(state)
    assert 0.0 <= c <= 1.0


def test_confidence_maximum_on_full_clean_run():
    """
    All 5 modalities submitted and handled, all tools succeeded,
    no truncation → 1.0.
    """
    state = _state_with(
        symptoms=True, lab=True, xray=True, meds=True, rag=True,
        tools_run=["A", "B", "C"],
    )
    c = calculate_confidence(state)
    assert abs(c - 1.0) < 1e-9


# ── Coverage signal — submitted vs handled ───────────────────────

def test_coverage_only_rag_submitted_and_handled():
    """
    Nothing submitted by user. RAG always submitted by pipeline.
    submitted=1, handled=1 → coverage = 1.0.
    """
    state = _state_with(rag=True, tools_run=["A"])
    # 0.4 * 1.0 + 0.4 * 1.0 + 0.2 * 1.0 = 1.0
    c = calculate_confidence(state)
    assert abs(c - 1.0) < 1e-9


def test_coverage_symptoms_only_fully_handled():
    """
    User submits symptoms only. RAG always submitted.
    Both handled → coverage = 2/2 = 1.0.
    """
    state = _state_with(symptoms=True, rag=True, tools_run=["A"])
    # 0.4 * 1.0 + 0.4 * 1.0 + 0.2 * 1.0 = 1.0
    c = calculate_confidence(state)
    assert abs(c - 1.0) < 1e-9


def test_coverage_symptoms_submitted_but_failed():
    """
    Symptoms submitted but extractor failed. RAG succeeded.
    submitted=2, handled=1 → coverage = 0.5.
    """
    state = _state_with(
        symptoms=True, symptoms_handled=False,
        rag=True,
        tools_run=["A"],
    )
    # 0.4 * 0.5 + 0.4 * 1.0 + 0.2 * 1.0 = 0.8
    c = calculate_confidence(state)
    assert abs(c - 0.8) < 1e-9


def test_coverage_all_modalities_submitted_all_handled():
    """All 5 submitted, all handled → coverage = 1.0."""
    state = _state_with(
        symptoms=True, lab=True, xray=True, meds=True, rag=True,
        tools_run=["A"],
    )
    c = calculate_confidence(state)
    assert abs(c - 1.0) < 1e-9


def test_coverage_partial_submission_all_handled():
    """
    User submits symptoms + meds only. RAG always submitted.
    All 3 handled → coverage = 3/3 = 1.0.
    """
    state = _state_with(
        symptoms=True, meds=True, rag=True,
        tools_run=["A"],
    )
    c = calculate_confidence(state)
    assert abs(c - 1.0) < 1e-9


def test_coverage_unsubmitted_modalities_do_not_penalise():
    """
    Demonstrates the corrected semantics: submitting fewer modalities
    does not reduce confidence if everything submitted was handled.
    """
    state_minimal = _state_with(symptoms=True, rag=True, tools_run=["A"])
    state_full = _state_with(
        symptoms=True, lab=True, xray=True, meds=True, rag=True,
        tools_run=["A"],
    )
    assert calculate_confidence(state_minimal) == calculate_confidence(state_full)


def test_coverage_rag_tool_error_reduces_coverage():
    """
    RAG is always submitted. If rag_result is a ToolError, RAG
    counts as submitted-but-not-handled.
    submitted=2 (symptoms+rag), handled=1 (symptoms) → 0.5.
    """
    state = _state_with(symptoms=True, tools_run=["A"])
    state.rag_result = ToolError(tool="MedicalRAGSearch", reason="fail")
    # 0.4 * 0.5 + 0.4 * 1.0 + 0.2 * 1.0 = 0.8
    c = calculate_confidence(state)
    assert abs(c - 0.8) < 1e-9


def test_coverage_xray_findings_count_as_submitted():
    """
    X-ray modality counts as submitted when clinician findings are
    provided even without an image upload.
    """
    state = AegisState(
        raw_symptoms_text="text",
        xray_findings_raw=["Cardiomegaly"],
    )
    state.symptom_result = SymptomExtractionResult(symptoms=["x"])
    state.xray_result = XRayResult(findings=["Cardiomegaly"])
    state.rag_result = RAGSearchResult(
        passages=[], citations=[], query_used="t", retrieval_successful=True,
    )
    state.tools_run = ["A"]
    # submitted: symptoms, xray, rag = 3; all handled = 3 → 1.0
    c = calculate_confidence(state)
    assert abs(c - 1.0) < 1e-9


def test_coverage_xray_free_text_counts_as_submitted():
    """
    X-ray modality counts as submitted when free text is provided
    even without image or checklist findings.
    """
    state = AegisState(
        raw_symptoms_text="text",
        xray_free_text_raw="mild interstitial markings",
    )
    state.symptom_result = SymptomExtractionResult(symptoms=["x"])
    state.xray_result = XRayResult(free_text="ok")
    state.rag_result = RAGSearchResult(
        passages=[], citations=[], query_used="t", retrieval_successful=True,
    )
    state.tools_run = ["A"]
    c = calculate_confidence(state)
    assert abs(c - 1.0) < 1e-9


def test_coverage_audio_path_counts_as_symptoms_submitted():
    """Audio file path counts as symptoms modality submitted."""
    state = AegisState(audio_file_path="/tmp/audio.wav")
    state.symptom_result = SymptomExtractionResult(symptoms=["x"])
    state.rag_result = RAGSearchResult(
        passages=[], citations=[], query_used="t", retrieval_successful=True,
    )
    state.tools_run = ["A"]
    # submitted: symptoms, rag = 2; handled: 2 → 1.0
    c = calculate_confidence(state)
    assert abs(c - 1.0) < 1e-9


# ── Success rate signal ────────────────────────────────────────────

def test_success_rate_all_succeeded():
    """All tools succeeded, no submissions beyond default RAG."""
    state = _state_with(rag=True, tools_run=["A", "B", "C"])
    # coverage = 1.0, success = 3/3 = 1.0, truncation = 1.0 → 1.0
    c = calculate_confidence(state)
    assert abs(c - 1.0) < 1e-9


def test_success_rate_partial_failure():
    state = _state_with(
        symptoms=True, rag=True,
        tools_run=["A", "B"],
        tools_failed=["C"],
    )
    # coverage = 1.0, success = 2/3, truncation = 1.0
    expected = 0.4 * 1.0 + 0.4 * (2 / 3) + 0.2 * 1.0
    c = calculate_confidence(state)
    assert abs(c - expected) < 1e-6


def test_success_rate_all_failed():
    """
    No successful tools.

    RAG is always considered submitted by the pipeline, but no
    RAGSearchResult was produced, so coverage = 0/1 = 0.0.
    """
    state = _state_with(tools_failed=["A", "B"])
    # coverage = handled/submitted: submitted=1 (rag), handled=0 → 0.0
    # success = 0/2 = 0.0
    # truncation = 1.0
    expected = 0.4 * 0.0 + 0.4 * 0.0 + 0.2 * 1.0
    c = calculate_confidence(state)
    assert abs(c - expected) < 1e-9


def test_success_rate_no_tools_at_all():
    """Floor of 1 prevents ZeroDivisionError."""
    state = _state_with()
    c = calculate_confidence(state)
    assert 0.0 <= c <= 1.0


# ── Truncation signal ──────────────────────────────────────────────

def test_no_truncation_score_is_1_0():
    state = _state_with(symptoms=True, rag=True, tools_run=["A"])
    # coverage = 1.0, success = 1.0, truncation = 1.0 → 1.0
    c = calculate_confidence(state)
    assert abs(c - 1.0) < 1e-9


def test_enrichment_truncation_score_is_0_7():
    state = _state_with(
        symptoms=True, rag=True, tools_run=["A"],
        enrichment_truncated=True,
    )
    expected = 0.4 * 1.0 + 0.4 * 1.0 + 0.2 * 0.7
    c = calculate_confidence(state)
    assert abs(c - expected) < 1e-9


def test_core_truncation_score_is_0_5():
    state = _state_with(
        symptoms=True, rag=True, tools_run=["A"],
        core_truncated=True,
    )
    expected = 0.4 * 1.0 + 0.4 * 1.0 + 0.2 * 0.5
    c = calculate_confidence(state)
    assert abs(c - expected) < 1e-9


def test_core_truncation_dominates_enrichment():
    state = _state_with(
        symptoms=True, rag=True, tools_run=["A"],
        core_truncated=True, enrichment_truncated=True,
    )
    expected = 0.4 * 1.0 + 0.4 * 1.0 + 0.2 * 0.5
    c = calculate_confidence(state)
    assert abs(c - expected) < 1e-9


# ── Severity does not affect confidence ───────────────────────────

def test_severity_level_does_not_affect_confidence(
    severity_high, severity_low
):
    """LOW and HIGH severity on identical state produce identical confidence."""
    state_high = _state_with(symptoms=True, rag=True, tools_run=["A", "B"])
    state_high.severity_result = severity_high

    state_low = _state_with(symptoms=True, rag=True, tools_run=["A", "B"])
    state_low.severity_result = severity_low

    assert calculate_confidence(state_high) == calculate_confidence(state_low)


# ── Clamping ───────────────────────────────────────────────────────

def test_confidence_never_below_zero():
    state = _state_with(tools_failed=["A", "B", "C"])
    assert calculate_confidence(state) >= 0.0


def test_confidence_never_above_one():
    state = _state_with(
        symptoms=True, lab=True, xray=True, meds=True, rag=True,
        tools_run=["A", "B", "C", "D", "E"],
    )
    assert calculate_confidence(state) <= 1.0