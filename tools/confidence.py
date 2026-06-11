"""
tools/confidence.py — Pipeline confidence calculation.

Single authoritative implementation of the confidence formula defined
in the technical spec. Called after all tools have run and truncation
flags have been set on AegisState.

Formula (three non-overlapping signals):
    confidence = 0.4 * coverage + 0.4 * success_rate + 0.2 * truncation
    clamped to [0.0, 1.0]

Signal definitions:

    Signal 1 — Modality coverage (weight 0.4)
        coverage = active_modalities / 5
        Modalities: symptoms · lab · xray · medications · rag
        RAG counted if rag_result is not None (ran, regardless of passages).
        Denominator=5 is a post-Week-3 tuning candidate if RAG inclusion
        in coverage proves misleading to clinicians.

    Signal 2 — Tool success rate (weight 0.4)
        success_rate = len(tools_run) / max(len(tools_run) + len(tools_failed), 1)
        Precondition: tools_run ∩ tools_failed = ∅ (enforced by AegisPipeline).

    Signal 3 — Truncation score (weight 0.2)
        1.0 — no truncation
        0.7 — enrichment fields truncated (minor penalty)
        0.5 — core fields truncated (severe penalty — should never occur
               in normal operation; logged as ERROR by report_generator.py)

        Both flags can be True simultaneously. Each flag can only lower
        the score, never raise it — min() enforces this explicitly.

Severity level does NOT reduce confidence — LOW severity on clean data
should yield high confidence.

Weights are tunable after Week 3 measurement without touching callers.
"""
from __future__ import annotations

from schemas.state import AegisState
from schemas.errors import ToolError

# ── Weights — tunable after Week 3 ───────────────────────────────
_WEIGHT_COVERAGE     = 0.4
_WEIGHT_SUCCESS_RATE = 0.4
_WEIGHT_TRUNCATION   = 0.2

# ── Truncation scores ─────────────────────────────────────────────
_TRUNCATION_NONE       = 1.0
_TRUNCATION_ENRICHMENT = 0.7
_TRUNCATION_CORE       = 0.5   # core truncation is an ERROR state


def calculate_confidence(state: AegisState) -> float:
    """
    Compute pipeline confidence from the completed AegisState.

    Precondition: all tools have run (tools_run ∩ tools_failed = ∅).
    Should be called after ReportGenerator has set truncation flags.

    Returns a float in [0.0, 1.0].
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


# ── Signal helpers ────────────────────────────────────────────────

def _modality_coverage(state: AegisState) -> float:
    """
    Fraction of the five clinical modalities that were active.

    A modality is active if its input was submitted. RAG is counted
    if rag_result is not None — it ran, even if passages=[] (zero
    results is valid output, not absence).
    """
    active = sum([
        bool(state.raw_symptoms_text),
        bool(state.lab_pdf_path),
        bool(state.xray_image_path),
        bool(state.medications_raw),
        state.rag_result is not None,
    ])
    return active / 5


def _tool_success_rate(state: AegisState) -> float:
    """
    Fraction of tools that ran successfully.

    tools_run ∩ tools_failed = ∅ is enforced by AegisPipeline.
    Denominator floor of 1 prevents ZeroDivisionError on empty pipelines.
    """
    total = len(state.tools_run) + len(state.tools_failed)
    return len(state.tools_run) / max(total, 1)


def _truncation_score(state: AegisState) -> float:
    """
    Penalty for context truncation.

    Each flag can only lower the score, never raise it.
    min() handles both flags being True simultaneously — core truncation
    dominates because _TRUNCATION_CORE < _TRUNCATION_ENRICHMENT, but
    this is a consequence of the values, not a special case in the logic.
    """
    score = _TRUNCATION_NONE

    if state.enrichment_fields_truncated:
        score = min(score, _TRUNCATION_ENRICHMENT)

    if state.core_fields_truncated:
        score = min(score, _TRUNCATION_CORE)

    return score