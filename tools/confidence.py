"""
tools/confidence.py — Pipeline confidence calculation.

Single authoritative implementation of the confidence formula defined
in the technical spec.

Formula:
    confidence = 0.4 * coverage + 0.4 * success_rate + 0.2 * truncation

Severity level does NOT reduce confidence.

This function computes TriageReport.confidence (pipeline-level confidence).
It is distinct from SeverityResult.confidence (rule-level confidence),
which comes from the highest-priority fired rule's rule_confidence value.

Called in agents/pipeline.py after tools_run.append("ReportGenerator")
and after truncation flags have been set by _build_context().
"""

from __future__ import annotations

from schemas.rag import RAGSearchResult
from schemas.state import AegisState


_WEIGHT_COVERAGE     = 0.4
_WEIGHT_SUCCESS_RATE = 0.4
_WEIGHT_TRUNCATION   = 0.2

_TRUNCATION_NONE       = 1.0
_TRUNCATION_ENRICHMENT = 0.7
_TRUNCATION_CORE       = 0.5


def calculate_confidence(state: AegisState) -> float:
    """
    Compute pipeline confidence from completed AegisState.

    Precondition: all tools have run, truncation flags are set.
    Returns float in [0.0, 1.0].
    """
    coverage     = _modality_coverage(state)
    success_rate = _tool_success_rate(state)
    truncation   = _truncation_score(state)

    raw = (
        _WEIGHT_COVERAGE     * coverage
        + _WEIGHT_SUCCESS_RATE * success_rate
        + _WEIGHT_TRUNCATION   * truncation
    )

    return max(0.0, min(1.0, raw))


def _modality_coverage(state: AegisState) -> float:
    """
    Fraction of five clinical modalities that were active.

    Modalities: symptoms · lab · xray · medications · rag

    RAG counted only when it ran successfully (RAGSearchResult instance).
    ToolError does NOT count as coverage — the mechanism failed.
    """
    active = sum([
        bool(state.raw_symptoms_text),
        bool(state.lab_pdf_path),
        bool(state.xray_image_path),
        bool(state.medications_raw),
        isinstance(state.rag_result, RAGSearchResult),
    ])

    return active / 5


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