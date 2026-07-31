"""
tools/confidence.py — Pipeline confidence calculation.

Single authoritative implementation of the confidence formula.

This scores how TRUSTWORTHY the generated report is, not just whether the
pipeline finished. It blends four signals:

    confidence = 0.40 * agreement      (does the LLM narrative match the
                                        deterministic rule-based severity?)
               + 0.25 * rule_strength  (how clear/strong the fired rule was)
               + 0.20 * evidence       (how many modalities backed the result)
               + 0.15 * pipeline_health (no tool failures, no truncation)

A hard ceiling (0.97) means a report is never shown as 100% "certain" — a
single clean symptom-only run with narrative↔rule agreement lands around
0.80, a multi-modal agreeing report around 0.95+, and a report whose
narrative disagrees with the rules drops to ~0.55.

IMPORTANT / honest limitation:
    The report text is produced by a small local LLM, which does not emit a
    calibrated certainty for its own narrative. These four signals are the
    best PROXIES for reliability we have; this is an engineered estimate, not
    a ground-truth probability.

Called in agents/pipeline.py AFTER the RuleValidator step so the agreement
signal exists.
"""

from __future__ import annotations

from schemas.errors import ToolError
from schemas.rag import RAGSearchResult
from schemas.state import AegisState


# ── Weights ───────────────────────────────────────────────────────
_WEIGHT_AGREEMENT = 0.40
_WEIGHT_RULE = 0.25
_WEIGHT_EVIDENCE = 0.20
_WEIGHT_HEALTH = 0.15

# Never display as 100% "certain" — medical confidence always leaves room.
_CONFIDENCE_CEILING = 0.97

# Defaults when a signal is unavailable (tool didn't run / didn't return).
_DEFAULT_RULE_STRENGTH = 0.70
_DEFAULT_AGREEMENT = 0.75  # validator didn't run → can't confirm or deny

# Truncation penalties.
_TRUNCATION_NONE = 1.0
_TRUNCATION_ENRICHMENT = 0.7
_TRUNCATION_CORE = 0.5

# Distinct evidence modalities the user can contribute.
_N_EVIDENCE_MODALITIES = 4  # symptoms, lab, x-ray, medications


def calculate_confidence(state: AegisState) -> float:
    """
    Compute pipeline confidence from completed AegisState.

    Precondition: SeverityScorer, ReportGenerator and RuleValidator have run.
    Returns float in [0.0, _CONFIDENCE_CEILING].
    """
    raw = (
        _WEIGHT_AGREEMENT * _agreement_score(state)
        + _WEIGHT_RULE * _rule_strength(state)
        + _WEIGHT_EVIDENCE * _evidence_score(state)
        + _WEIGHT_HEALTH * _pipeline_health(state)
    )
    return max(0.0, min(_CONFIDENCE_CEILING, raw))


# ── Signal 1: deterministic ↔ LLM-narrative agreement ────────────
def _agreement_score(state: AegisState) -> float:
    """
    How well the generated narrative severity agrees with the rule-based
    severity, from RuleValidatorResult.status.
        agreement -> 1.00   narrative matches the deterministic analysis
        warning   -> 0.65   minor mismatch, worth a second look
        override  -> 0.35   narrative contradicts the rules (low trust)
        <none>    -> 0.75   validator didn't run → neutral
    """
    rv = getattr(state, "rule_validator_result", None)
    if rv is None:
        return _DEFAULT_AGREEMENT
    status = getattr(rv, "status", None)
    value = status.value if hasattr(status, "value") else str(status)
    value = (value or "").strip().lower()
    if value == "agreement":
        return 1.0
    if value == "warning":
        return 0.65
    if value == "override":
        return 0.35
    return _DEFAULT_AGREEMENT


# ── Signal 2: rule strength ──────────────────────────────────────
def _rule_strength(state: AegisState) -> float:
    """
    The fired rule's own confidence (SeverityResult.confidence), i.e. how
    clear/strong the clinical evidence for the chosen severity was.
    Falls back to a neutral default when the scorer didn't produce a value.
    """
    sr = getattr(state, "severity_result", None)
    if sr is None:
        return _DEFAULT_RULE_STRENGTH
    confidence = getattr(sr, "confidence", None)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return _DEFAULT_RULE_STRENGTH
    if confidence != confidence:  # NaN guard
        return _DEFAULT_RULE_STRENGTH
    return max(0.0, min(1.0, confidence))


# ── Signal 3: evidence richness ──────────────────────────────────
def _evidence_score(state: AegisState) -> float:
    """
    Fraction of the four evidence modalities that were both submitted AND
    handled. Rewards multi-modality: a symptom-only report is 0.25, a
    symptoms+labs+xray+meds report can reach 1.0.
    """
    pairs = [
        (
            bool(state.raw_symptoms_text) or bool(state.audio_file_path),
            _handled(state.symptom_result),
        ),
        (
            bool(state.lab_pdf_path),
            _handled(state.lab_result),
        ),
        (
            bool(state.xray_image_path)
            or bool(state.xray_findings_raw)
            or bool(state.xray_free_text_raw),
            _handled(state.xray_result),
        ),
        (
            bool(state.medications_raw),
            _handled(state.drug_result),
        ),
    ]
    handled = sum(1 for submitted, ok in pairs if submitted and ok)
    return min(1.0, handled / float(_N_EVIDENCE_MODALITIES))


# ── Signal 4: pipeline health (kept from the original formula) ────
def _pipeline_health(state: AegisState) -> float:
    """success_rate × truncation_score — penalises failures/truncation."""
    return _tool_success_rate(state) * _truncation_score(state)


def _handled(result: object) -> bool:
    """True when a result exists and is not a ToolError."""
    return result is not None and not isinstance(result, ToolError)


def _tool_success_rate(state: AegisState) -> float:
    """Fraction of tools that completed successfully (floor of 1)."""
    total = len(state.tools_run) + len(state.tools_failed)
    return len(state.tools_run) / max(total, 1)


def _truncation_score(state: AegisState) -> float:
    """Penalty for context truncation. Each flag only lowers the score."""
    score = _TRUNCATION_NONE
    if state.enrichment_fields_truncated:
        score = min(score, _TRUNCATION_ENRICHMENT)
    if state.core_fields_truncated:
        score = min(score, _TRUNCATION_CORE)
    return score
