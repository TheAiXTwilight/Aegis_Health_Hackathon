"""
tools/severity_calibrator.py — Multi-signal severity confidence calibrator.

Adjusts severity confidence (and optionally severity level) based on:
  - Number and diversity of fired rules
  - Symptom-lab correlation
  - Text-finding correlations (from FIX #3)
  - Patient context escalations (from FIX #4)
  - Clinical pattern matches (from FIX #7 Part 2)
  - Diagnostic ambiguity signals

Design principles:
  - Registry-driven: adjustments and escalations live in ordered lists.
    Adding a new calibration rule is one dict entry.
  - Universal: no biomarker-specific logic; operates on rule metadata,
    context flags, and pattern IDs.
  - Additive: existing rule confidence values in severity_scorer are
    respected as a baseline. Calibration only shifts confidence and
    may escalate level; it never invents new rules.
  - Fail-safe: any error returns the original (level, confidence,
    reasons) unchanged so severity_scorer's contract is preserved.
  - Explainable: every adjustment records an audit entry so downstream
    can trace why confidence/level was changed.
  - Bounded: confidence is clamped to [0.0, 1.0] after all adjustments.

Consumed by:
  tools.severity_scorer.SeverityScorer.score() — applied AFTER the
  base rule engine has determined the initial (level, confidence,
  fired_rules), BEFORE returning SeverityResult.
"""
from __future__ import annotations

from typing import Any, Callable

from loguru import logger
from tools.unit_normalizer import normalize_for_comparison, _resolve_canonical_key


# ═══════════════════════════════════════════════════════════════════
# CONTEXT BUILDER
# ═══════════════════════════════════════════════════════════════════
# The calibrator receives a rich dict of contextual signals derived
# from the pipeline state. This isolates all state access to one place
# and makes adjustment conditions purely data-driven.

def _build_calibration_context(
    state,
    fired_rules: list,
    current_level: str,
    current_confidence: float,
) -> dict:
    """
    Assemble a context dict from pipeline state for use by adjustment
    condition lambdas. All potentially-missing fields default to safe
    empty values so lambdas can inspect them without try/except.
    """
    try:
        from schemas.errors import ToolError
    except Exception:
        ToolError = tuple()  # sentinel that isinstance will always return False for

    # Fired rule breakdown
    high_rules = [r for r in fired_rules if getattr(r, "level", None) == "HIGH"]
    medium_rules = [r for r in fired_rules if getattr(r, "level", None) == "MEDIUM"]

    # Symptoms
    symptoms: list[str] = []
    sym = getattr(state, "symptom_result", None)
    if sym and not isinstance(sym, ToolError):
        symptoms = list(getattr(sym, "symptoms", []) or [])
    raw_sym_text = str(getattr(state, "raw_symptoms_text", "") or "")

    # Lab flags
    lab = getattr(state, "lab_result", None)
    lab_ok = lab and not isinstance(lab, ToolError)
    abnormal_count = 0

    lab_measurements: dict = {}
    if lab_ok:
        abnormal_count = len(getattr(lab, "abnormal_values", []) or [])
        raw_measurements = dict(getattr(lab, "measurements", {}) or {})
        raw_units = dict(getattr(lab, "units", {}) or {})
        # Build measurement dicts that _measurement_in_range expects:
        # {"key": str, "value": float, "unit": str}
        lab_measurements = {
            k: {"key": k, "value": v, "unit": raw_units.get(k, "")}
            for k, v in raw_measurements.items()
        }

    # X-ray flags
    xray = getattr(state, "xray_result", None)
    xray_ok = xray and not isinstance(xray, ToolError)
    xray_finding_count = 0
    if xray_ok:
        xray_findings = list(getattr(xray, "findings", []) or [])
        xray_finding_count = len([
            f for f in xray_findings
            if f and f.lower() not in ("normal", "normal / no significant findings")
        ])

    # Drug flags
    drug = getattr(state, "drug_result", None)
    drug_ok = drug and not isinstance(drug, ToolError)
    drug_interaction_count = 0
    if drug_ok:
        drug_interaction_count = len(getattr(drug, "interactions", []) or [])

    # Text finding pattern matches (FIX #3 / FIX #10)
    text_pattern_ids = list(
        getattr(state, "text_finding_matched_patterns", []) or []
    )

    return {
        "fired_rules": list(fired_rules),
        "high_rules": high_rules,
        "medium_rules": medium_rules,
        "current_level": str(current_level).upper(),
        "current_confidence": float(current_confidence),

        "symptoms": symptoms,
        "has_symptoms": bool(symptoms or raw_sym_text.strip()),
        "raw_symptoms_text": raw_sym_text.lower(),

        "abnormal_lab_count": abnormal_count,
        "has_lab_flags": abnormal_count > 0,
        "lab_measurements": lab_measurements,

        "xray_finding_count": xray_finding_count,
        "has_xray_flags": xray_finding_count > 0,

        "drug_interaction_count": drug_interaction_count,
        "has_drug_flags": drug_interaction_count > 0,

        "text_pattern_ids": text_pattern_ids,
        "has_text_findings": bool(text_pattern_ids),
    }


# ═══════════════════════════════════════════════════════════════════
# SYMPTOM-LAB CORRELATION HELPERS
# ═══════════════════════════════════════════════════════════════════
# Small helpers that inspect the context to detect correlation patterns.
# All fail-safe: return False on any error.

def _symptom_contains_any(ctx: dict, terms: list[str]) -> bool:
    combined = " ".join(str(s).lower() for s in ctx.get("symptoms", []))
    combined += " " + ctx.get("raw_symptoms_text", "")
    return any(t.lower() in combined for t in terms)


def _measurement_in_range(
    measurement: dict,
    range_condition: dict,
) -> bool:
    """
    Check if a single measurement satisfies a range condition.

    Range condition shape (all fields optional):
      {
        "key":   str,          # biomarker key to match
        "op":    str,          # ">", "<", ">=", "<=", "==", "between"
        "value": float,        # threshold (for non-between ops)
        "low":   float,        # lower bound (for "between")
        "high":  float,        # upper bound (for "between")
      }

    Routes numeric comparisons through normalize_for_comparison() so
    clinical shorthand thresholds are compared against raw parser values
    in the correct canonical unit.

    Fail-safe: any error returns False.
    """
    key_wanted = str(range_condition.get("key") or "").lower()
    op         = range_condition.get("op")

    if not key_wanted or not op:
        return False

    m_key = str(measurement.get("key") or "").lower()

    # Key match: direct or via alias
    if m_key != key_wanted:
        if _resolve_canonical_key(m_key) != _resolve_canonical_key(key_wanted):
            return False

    raw_value = measurement.get("value")
    raw_unit  = measurement.get("unit") or measurement.get("units") or ""

    # Normalize value to canonical unit
    try:
        norm = normalize_for_comparison(m_key, raw_value, raw_unit)
        v = norm.value
    except Exception:
        try:
            v = float(raw_value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    # ── Between comparison ────────────────────────────────────────────
    if op == "between":
        try:
            low  = float(range_condition.get("low",  0))
            high = float(range_condition.get("high", 0))
            return low <= v <= high
        except (TypeError, ValueError):
            return False

    # ── Standard numeric comparison ───────────────────────────────────
    threshold = range_condition.get("value")
    try:
        threshold_f = float(threshold)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False

    try:
        if op == ">"  and v >  threshold_f: return True
        if op == "<"  and v <  threshold_f: return True
        if op == ">=" and v >= threshold_f: return True
        if op == "<=" and v <= threshold_f: return True
        if op == "==" and v == threshold_f: return True
    except TypeError:
        pass

    return False


def _has_fever_platelets_correlation(ctx: dict) -> bool:
    """Fever symptom + platelets < 150 K/µL = viral fever pattern."""
    try:
        if not _symptom_contains_any(ctx, ["fever", "temperature", "chills"]):
            return False
        # Locate the platelets measurement dict from the lab_measurements
        # context (keyed by canonical key or common aliases).
        lab = ctx.get("lab_measurements") or {}
        platelet_m = (
            lab.get("platelets")
            or lab.get("plt")
            or lab.get("platelet_count")
        )
        if platelet_m is None:
            return False
        # Ensure it's a dict shape _measurement_in_range expects.
        if not isinstance(platelet_m, dict):
            # Some adapters store raw floats keyed by biomarker name.
            # Wrap into the expected shape with no unit so normalize
            # returns the raw value as-is.
            platelet_m = {"key": "platelets", "value": platelet_m, "unit": ""}
        return _measurement_in_range(
            platelet_m,
            {"key": "platelets", "op": "<", "value": 150},
        )
    except Exception:
        return False


def _has_chest_pain_troponin_correlation(ctx: dict) -> bool:
    """Chest pain + troponin > 0.04 ng/mL = high-confidence cardiac pattern."""
    try:
        if not _symptom_contains_any(
            ctx, ["chest pain", "chest tightness", "chest pressure"]
        ):
            return False
        lab = ctx.get("lab_measurements") or {}
        troponin_m = (
            lab.get("troponin")
            or lab.get("troponin_i")
            or lab.get("troponin_t")
            or lab.get("hs_troponin")
        )
        if troponin_m is None:
            return False
        if not isinstance(troponin_m, dict):
            troponin_m = {"key": "troponin", "value": troponin_m, "unit": ""}
        return _measurement_in_range(
            troponin_m,
            {"key": "troponin", "op": ">", "value": 0.04},
        )
    except Exception:
        return False


def _has_symptom_lab_correlation(ctx: dict) -> bool:
    """
    Universal correlation check: any known symptom-biomarker pairing
    is satisfied. Returns True if ANY correlation pattern hits.
    """
    if not ctx.get("has_symptoms") or not ctx.get("has_lab_flags"):
        return False
    checks = [
        _has_fever_platelets_correlation(ctx),
        _has_chest_pain_troponin_correlation(ctx),
        # Additional correlations can be added here — each is a small
        # named function for testability.
    ]
    return any(checks)


def _has_ambiguous_pattern(ctx: dict) -> bool:
    """
    Diagnostic ambiguity marker: signals present but not a clear
    single-pattern match. Example: multiple abnormal labs + symptoms
    but text findings didn't narrow to a specific pattern.
    """
    try:
        return (
            ctx.get("abnormal_lab_count", 0) >= 3
            and ctx.get("has_symptoms")
            and not ctx.get("has_text_findings")
            and len(ctx.get("high_rules", [])) == 0
        )
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════
# CONFIDENCE ADJUSTMENT REGISTRY
# ═══════════════════════════════════════════════════════════════════
# Each entry:
#   id:                unique identifier
#   condition:         callable(ctx) → bool
#   confidence_delta:  float applied to confidence when condition True
#   reason:            audit-trail sentence

_CONFIDENCE_ADJUSTMENTS: list[dict[str, Any]] = [
    {
        "id": "multiple_high_rules_corroborate",
        "condition": lambda ctx: len(ctx["high_rules"]) >= 2,
        "confidence_delta": +0.03,
        "reason": (
            "Multiple high-priority rules fired concurrently — "
            "corroborating clinical signals."
        ),
    },
    {
        "id": "symptom_lab_correlation",
        "condition": _has_symptom_lab_correlation,
        "confidence_delta": +0.04,
        "reason": (
            "Reported symptoms correlate with laboratory findings — "
            "correlation strengthens diagnostic confidence."
        ),
    },
    {
        "id": "text_finding_reinforces",
        "condition": lambda ctx: (
            ctx["has_text_findings"] and ctx["has_lab_flags"]
        ),
        "confidence_delta": +0.02,
        "reason": (
            "Peripheral smear or interpretive findings reinforce the "
            "numeric biomarker abnormalities."
        ),
    },
    {
        "id": "isolated_abnormal_lab_only",
        "condition": lambda ctx: (
            len(ctx["fired_rules"]) == 1
            and ctx["abnormal_lab_count"] == 1
            and not ctx["has_symptoms"]
            and not ctx["has_xray_flags"]
        ),
        "confidence_delta": -0.10,
        "reason": (
            "Single abnormal lab value with no symptom or imaging "
            "corroboration — reduced diagnostic certainty."
        ),
    },
    {
        "id": "abnormal_lab_no_symptoms",
        "condition": lambda ctx: (
            ctx["has_lab_flags"]
            and not ctx["has_symptoms"]
            and len(ctx["high_rules"]) == 0
        ),
        "confidence_delta": -0.05,
        "reason": (
            "Laboratory abnormalities present without corroborating "
            "symptoms — clinical correlation needed."
        ),
    },
    {
        "id": "ambiguous_multi_signal_pattern",
        "condition": _has_ambiguous_pattern,
        "confidence_delta": -0.08,
        "reason": (
            "Multiple abnormal signals present but do not converge "
            "on a single specific clinical pattern — diagnostic "
            "uncertainty."
        ),
    },
    {
        "id": "high_severity_symptom_confirmed",
        "condition": lambda ctx: (
            ctx["current_level"] == "HIGH"
            and ctx["has_symptoms"]
        ),
        "confidence_delta": +0.02,
        "reason": (
            "High-severity rule fired alongside relevant reported "
            "symptoms — clinical picture consistent."
        ),
    },
    {
        "id": "drug_and_lab_flags_combined",
        "condition": lambda ctx: (
            ctx["has_drug_flags"] and ctx["has_lab_flags"]
        ),
        "confidence_delta": +0.02,
        "reason": (
            "Drug interaction risk combined with laboratory "
            "abnormalities — increased clinical relevance."
        ),
    },
]


# ═══════════════════════════════════════════════════════════════════
# SEVERITY ESCALATION REGISTRY
# ═══════════════════════════════════════════════════════════════════
# Escalation rules may change the severity LEVEL (not just confidence)
# when clusters of corroborating signals are detected.
#
# Each entry:
#   id:          unique identifier
#   from_level:  level required to consider escalation
#   to_level:    level the severity is upgraded to
#   condition:   callable(ctx) → bool
#   reason:      audit-trail sentence
#
# Never downgrades — only escalates.

_SEVERITY_ESCALATIONS: list[dict[str, Any]] = [
    {
        "id": "medium_with_symptom_lab_correlation_cluster",
        "from_level": "MEDIUM",
        "to_level": "HIGH",
        "condition": lambda ctx: (
            ctx["current_level"] == "MEDIUM"
            and _has_symptom_lab_correlation(ctx)
            and (
                ctx["abnormal_lab_count"] >= 2
                or ctx["has_text_findings"]
            )
        ),
        "reason": (
            "Symptoms correlate with multiple abnormal labs or "
            "interpretive findings — severity escalated to reflect "
            "convergent clinical concern."
        ),
    },
    {
        "id": "medium_with_many_corroborating_rules",
        "from_level": "MEDIUM",
        "to_level": "HIGH",
        "condition": lambda ctx: (
            ctx["current_level"] == "MEDIUM"
            and len(ctx["fired_rules"]) >= 4
            and ctx["has_lab_flags"]
            and ctx["has_symptoms"]
        ),
        "reason": (
            "Four or more clinical rules fired alongside symptoms "
            "and labs — severity escalated to reflect corroborating "
            "signal density."
        ),
    },
    {
        "id": "text_finding_severity_boost_significant",
        "from_level": "MEDIUM",
        "to_level": "HIGH",
        "condition": lambda ctx: (
            ctx["current_level"] == "MEDIUM"
            and len(ctx["text_pattern_ids"]) >= 2
            and ctx["has_lab_flags"]
        ),
        "reason": (
            "Multiple interpretive text-finding patterns matched "
            "alongside laboratory abnormalities — severity escalated."
        ),
    },
]


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════

def calibrate_severity(
    state,
    fired_rules: list,
    current_level: str,
    current_confidence: float,
    current_reasons: list[str],
) -> dict:
    """
    Adjust severity confidence and (optionally) level based on
    contextual signals across the full pipeline state.

    Args:
        state:              Full pipeline state (AegisState).
        fired_rules:        List of Rule objects that fired.
        current_level:      Severity level from base rule engine.
        current_confidence: Confidence from base rule engine
                            (typically highest.rule_confidence).
        current_reasons:    Reasons list from base rule engine.

    Returns:
        {
          "level":        str,          # possibly escalated
          "confidence":   float,        # adjusted, clamped to [0, 1]
          "reasons":      list[str],    # original + calibration reasons
          "adjustments":  list[dict],   # audit trail
        }

    Fail-safe: any error returns the inputs unchanged in the same
    output shape so severity_scorer's contract is preserved.
    """
    try:
        ctx = _build_calibration_context(
            state, fired_rules, current_level, current_confidence,
        )

        adjusted_confidence = float(current_confidence)
        adjusted_level = str(current_level).upper()
        adjustments: list[dict] = []
        additional_reasons: list[str] = []

        # ── Layer 1: Apply confidence adjustments ─────────────────
        for adj in _CONFIDENCE_ADJUSTMENTS:
            try:
                condition_fn: Callable = adj["condition"]
                if condition_fn(ctx):
                    delta = float(adj["confidence_delta"])
                    adjusted_confidence += delta
                    adjustments.append({
                        "id": adj["id"],
                        "delta": delta,
                        "reason": adj["reason"],
                        "type": "confidence",
                    })
            except Exception:
                # Skip failed adjustment, continue evaluating others
                continue

        # ── Layer 2: Apply severity escalations ───────────────────
        # Refresh ctx with current_level in case a prior escalation
        # already elevated it (allows chained escalations if defined).
        ctx["current_level"] = adjusted_level
        for esc in _SEVERITY_ESCALATIONS:
            try:
                if adjusted_level != esc.get("from_level"):
                    continue
                condition_fn = esc["condition"]
                if condition_fn(ctx):
                    old_level = adjusted_level
                    adjusted_level = str(esc["to_level"]).upper()
                    ctx["current_level"] = adjusted_level
                    adjustments.append({
                        "id": esc["id"],
                        "from_level": old_level,
                        "to_level": adjusted_level,
                        "reason": esc["reason"],
                        "type": "escalation",
                    })
            except Exception:
                continue

        # ── Clamp confidence to valid range ────────────────────────
        if adjusted_confidence < 0.0:
            adjusted_confidence = 0.0
        if adjusted_confidence > 1.0:
            adjusted_confidence = 1.0

        # ── Assemble final reasons list ────────────────────────────
        final_reasons = list(current_reasons or []) + additional_reasons

        if adjustments:
            logger.info(
                "severity_calibrator · adjustments applied",
                original_level=current_level,
                final_level=adjusted_level,
                original_confidence=round(float(current_confidence), 3),
                final_confidence=round(float(adjusted_confidence), 3),
                adjustment_count=len(adjustments),
                adjustment_ids=[a["id"] for a in adjustments],
            )

        return {
            "level": adjusted_level,
            "confidence": adjusted_confidence,
            "reasons": final_reasons,
            "adjustments": adjustments,
        }

    except Exception:
        logger.exception(
            "severity_calibrator · calibration failed; returning inputs unchanged"
        )
        return {
            "level": str(current_level).upper(),
            "confidence": float(current_confidence),
            "reasons": list(current_reasons or []),
            "adjustments": [],
        }


__all__ = ["calibrate_severity"]