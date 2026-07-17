"""
tools/confidence.py — Pipeline confidence calculation.

Single authoritative implementation of the confidence formula.

Formula:
    confidence = 0.4 * coverage + 0.4 * success_rate + 0.2 * truncation

Severity level does NOT reduce confidence.

This function computes TriageReport.confidence (pipeline-level confidence).
It is distinct from SeverityResult.confidence (rule-level confidence),
which comes from the highest-priority fired rule's rule_confidence value.

Called in agents/pipeline.py after tools_run.append("ReportGenerator")
and after truncation flags have been set by _build_context().

Coverage semantics:
    Coverage measures how well the pipeline handled what the user
    submitted, not whether they submitted everything possible.

    Modalities the user did not submit are NOT counted against confidence.
    A session with only typed symptoms can reach coverage = 1.0 if the
    symptom path succeeded.

    coverage = handled / submitted

    where:
        submitted = count of modalities the user actually provided
        handled   = count of submitted modalities that produced a
                    non-error structured result

    RAG is treated as an always-submitted pipeline modality because
    every report attempts retrieval of supporting medical knowledge.
    Failure of retrieval therefore reduces overall pipeline confidence.

    Because RAG is always submitted, `submitted` is guaranteed to be
    at least 1 by construction — no ZeroDivisionError guard is needed.
"""

from __future__ import annotations

from schemas.errors import ToolError
from schemas.rag import RAGSearchResult
from schemas.state import AegisState


_WEIGHT_COVERAGE = 0.4
_WEIGHT_SUCCESS_RATE = 0.4
_WEIGHT_TRUNCATION = 0.2

_TRUNCATION_NONE = 1.0
_TRUNCATION_ENRICHMENT = 0.7
_TRUNCATION_CORE = 0.5


def calculate_confidence(state: AegisState) -> float:
    """
    Compute pipeline confidence from completed AegisState.

    Precondition: all tools have run, truncation flags are set.
    Returns float in [0.0, 1.0].
    """
    coverage = _modality_coverage(state)
    success_rate = _tool_success_rate(state)
    truncation = _truncation_score(state)

    raw = (
        _WEIGHT_COVERAGE * coverage
        + _WEIGHT_SUCCESS_RATE * success_rate
        + _WEIGHT_TRUNCATION * truncation
    )

    return max(0.0, min(1.0, raw))


def _handled(result: object) -> bool:
    """
    Return True when a modality/tool result exists and is not a ToolError.

    Centralised so the definition of "handled successfully" lives in
    one place. If Phase 3 introduces richer failure wrappers, only this
    helper should need updating.
    """
    return result is not None and not isinstance(result, ToolError)


def _modality_coverage(state: AegisState) -> float:
    """
    Fraction of submitted modalities that produced a successful result.

    Modalities the user did not submit are not counted against the score.

    Submission rules:
        symptoms — raw_symptoms_text OR audio_file_path
        lab      — lab_pdf_path
        xray     — xray_image_path OR xray_findings_raw OR xray_free_text_raw
        meds     — medications_raw non-empty
        rag      — always submitted (pipeline always attempts retrieval)

    Handled rules:
        *_result is the corresponding structured Result type
        (not ToolError, not None)

    Because RAG is always submitted, `submitted` is guaranteed to be
    at least 1 — division is safe without a guard.

    Phase 3 note: while XRayProcessor remains a stub returning None,
    X-ray will never be handled even when submitted. This resolves
    when the real XRayProcessor lands.
    """
    submitted_handled_pairs: list[tuple[bool, bool]] = [
        # Symptoms (text or audio)
        (
            bool(state.raw_symptoms_text) or bool(state.audio_file_path),
            _handled(state.symptom_result),
        ),
        # Lab (PDF upload)
        (
            bool(state.lab_pdf_path),
            _handled(state.lab_result),
        ),
        # X-ray (image OR clinician findings OR free text)
        (
            bool(state.xray_image_path)
            or bool(state.xray_findings_raw)
            or bool(state.xray_free_text_raw),
            _handled(state.xray_result),
        ),
        # Medications
        (
            bool(state.medications_raw),
            _handled(state.drug_result),
        ),
        # RAG — always attempted by pipeline, so always submitted.
        # This guarantees `submitted >= 1` below.
        (
            True,
            isinstance(state.rag_result, RAGSearchResult),
        ),
    ]

    submitted = sum(1 for s, _ in submitted_handled_pairs if s)
    handled = sum(1 for s, h in submitted_handled_pairs if s and h)

    # submitted is guaranteed >= 1 because the RAG entry is always True.
    return handled / submitted


def _tool_success_rate(state: AegisState) -> float:
    """
    Fraction of tools that completed successfully.

    tools_run ∩ tools_failed = ∅ enforced by AegisPipeline.
    Floor of 1 prevents ZeroDivisionError on empty pipelines.
    """
    total = len(state.tools_run) + len(state.tools_failed)
    return len(state.tools_run) / max(total, 1)


def _truncation_score(state: AegisState) -> float:
    """
    Penalty score for context truncation.

    Each flag can only lower the score, never raise it.
    Core truncation dominates enrichment truncation.
    """
    score = _TRUNCATION_NONE

    if state.enrichment_fields_truncated:
        score = min(score, _TRUNCATION_ENRICHMENT)

    if state.core_fields_truncated:
        score = min(score, _TRUNCATION_CORE)

    return score