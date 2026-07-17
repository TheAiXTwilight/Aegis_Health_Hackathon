"""
tools/clinical_picture_synthesizer.py — Cross-registry pattern
aggregation, confidence scoring, and differential diagnosis synthesis.

Consumes the outputs of the three existing pattern registries:
  - tools.text_finding_analyzer          (peripheral smear / impressions)
  - tools.clinical_synthesis             (multi-signal clinical patterns)
  - tools.patient_context_adapter        (safety escalations)

Produces a UNIFIED CLINICAL PICTURE:
  - Confident findings   (high-confidence single-pattern matches)
  - Differential list    (ranked alternatives for ambiguous cases)
  - Picture summary      (one-paragraph synthesis)
  - Confidence scores    (per-pattern 0.0–1.0)

Design principles:
  - Consumes existing registries — never duplicates patterns.
  - Universal confidence formula — same math applies to every pattern
    regardless of clinical domain.
  - Additive: renders as a new subsection ABOVE per-biomarker
    recommendations only when meaningful output is produced.
  - Fail-safe: any error returns an empty result so the report renders
    exactly as it would without this layer.
  - Registry-driven pattern metadata lookup — this file adds NO new
    clinical patterns; it only scores and ranks existing ones.

Row 3 dashboard support:
  Each scored pattern now also carries a `narrative_short` field
  passed through from its source registry (text finding
  `narrative_short`, clinical synthesis `narrative_short`, or safety
  rule `safety_warning_short`). The dashboard Clinical Picture Summary
  card prefers this short form; falls back to the full `narrative`
  when a pattern predates the short-form field.

Consumed by:
  tools.report_generator._build_deterministic_report() — hooks the
  synthesizer after clinical_synthesis has run, and renders the
  clinical picture HTML above per-biomarker recommendations.
"""
from __future__ import annotations

from typing import Any

from loguru import logger


# ═══════════════════════════════════════════════════════════════════
# CONFIDENCE SCORING PARAMETERS
# Single point of tuning for the universal scoring formula.
# All values in [0.0, 1.0].
# ═══════════════════════════════════════════════════════════════════

# _BASE_SCORE:
#   Starting confidence before signal boosts or penalties.
#   Set above _DIFFERENTIAL_THRESHOLD so a pattern with at least one
#   corroborating signal survives even with a small ambiguity penalty.
#   Not so high that zero-signal patterns pass through freely.
_BASE_SCORE = 0.58

# _DIFFERENTIAL_THRESHOLD:
#   Minimum confidence for a pattern to appear in the differential list.
#   Clinically meaningful floor — do not lower this.
_DIFFERENTIAL_THRESHOLD = 0.55

# _CONFIDENT_THRESHOLD:
#   Minimum confidence for a pattern to be promoted to "confident finding".
#   Lowered from 0.75 → 0.72 so strong multi-signal patterns reach this
#   tier without requiring every possible boost to fire simultaneously.
_CONFIDENT_THRESHOLD = 0.72

# _MAX_SIGNAL_BOOST:
#   Hard cap on total signal boost per pattern.
#   Raised from 0.40 → 0.50 to allow multi-biomarker patterns with strong
#   corroboration to reach the confident tier.
_MAX_SIGNAL_BOOST = 0.50

# _SIGNAL_BOOST_PER_TYPE:
#   Confidence added per corroborated trigger type (symptom, biomarker,
#   condition, text finding). Capped by _MAX_SIGNAL_BOOST.
_SIGNAL_BOOST_PER_TYPE = 0.12

# _MULTI_TRIGGER_BOOST:
#   Extra confidence for patterns that declare 2+ distinct trigger types.
#   Rewards specificity — a pattern requiring symptoms AND biomarker AND
#   condition is more discriminating than one requiring biomarker alone.
_MULTI_TRIGGER_BOOST = 0.06

# _AMBIGUITY_PENALTY_FLAT:
#   Legacy sentinel — retained for any external reference.
#   Scoring uses _ambiguity_penalty() function, not this constant.
_AMBIGUITY_PENALTY_FLAT = 0.10


# ═══════════════════════════════════════════════════════════════════
# PATTERN METADATA LOOKUP
# ═══════════════════════════════════════════════════════════════════
# Fetches pattern definitions from the source registries so we can
# inspect their triggers for scoring. Fail-safe: returns {} on any
# error so scoring proceeds with just the pattern ID (base score only).

def _lookup_text_finding_pattern(pattern_id: str) -> dict:
    try:
        from tools.text_finding_analyzer import get_registered_patterns
        for p in get_registered_patterns():
            if p.get("id") == pattern_id:
                return p
    except Exception:
        pass
    return {}


def _lookup_clinical_synthesis_pattern(pattern_id: str) -> dict:
    try:
        from tools.clinical_synthesis import _CLINICAL_PATTERNS
        for p in _CLINICAL_PATTERNS:
            if p.get("id") == pattern_id:
                return dict(p)
    except Exception:
        pass
    return {}


def _lookup_safety_escalation_rule(rule_id: str) -> dict:
    try:
        from tools.patient_context_adapter import _SAFETY_ESCALATION_RULES
        for r in _SAFETY_ESCALATION_RULES:
            if r.get("id") == rule_id:
                return dict(r)
    except Exception:
        pass
    return {}


# ═══════════════════════════════════════════════════════════════════
# SIGNAL COUNTING (universal — no clinical domain logic)
# ═══════════════════════════════════════════════════════════════════

def _count_trigger_types(pattern: dict) -> int:
    """
    Count how many distinct trigger types are declared on this pattern.
    A pattern requiring symptoms + biomarker + condition scores higher
    than one requiring only biomarker because it satisfies more signals.
    """
    triggers = pattern.get("triggers") or {}
    return sum(1 for k in (
        "symptoms", "biomarker", "biomarker_combo",
        "patient_conditions", "text_patterns",
    ) if k in triggers)


def _count_matching_context_signals(
    pattern: dict,
    context: dict,
) -> int:
    """
    Count how many of the pattern's declared trigger types are actively
    corroborated by the provided context. This is the "signal boost"
    input in the scoring formula.

    Uses only presence checks (not full re-match logic) because the
    pattern is already confirmed matched at this point.
    """
    triggers = pattern.get("triggers") or {}
    supporting = 0

    if "symptoms" in triggers and context.get("has_symptoms"):
        supporting += 1
    if ("biomarker" in triggers or "biomarker_combo" in triggers) \
            and context.get("has_lab_flags"):
        supporting += 1
    if "patient_conditions" in triggers and context.get("has_patient_conditions"):
        supporting += 1
    if "text_patterns" in triggers and context.get("has_text_findings"):
        supporting += 1

    return supporting


# ═══════════════════════════════════════════════════════════════════
# CONTEXT BUILDER
# ═══════════════════════════════════════════════════════════════════

def _build_scoring_context(
    flagged_items: list[dict],
    symptoms: list[str],
    patient_context: dict,
    text_pattern_ids: list[str],
) -> dict:
    """Build a small context dict used by signal-counting helpers."""
    return {
        "has_symptoms": bool(symptoms),
        "has_lab_flags": bool(flagged_items),
        "has_patient_conditions": bool(
            (patient_context or {}).get("conditions")
        ),
        "has_text_findings": bool(text_pattern_ids),
        "flagged_count": len(flagged_items or []),
        "symptom_count": len(symptoms or []),
    }


# ═══════════════════════════════════════════════════════════════════
# AMBIGUITY PENALTY (replaces flat constant)
# ═══════════════════════════════════════════════════════════════════

def _ambiguity_penalty(n_patterns: int) -> float:
    """
    Compute the ambiguity penalty when multiple patterns are
    simultaneously active.

    Design (universal — depends only on count, not clinical domain):
      - 1–3 patterns: no penalty. Co-occurrence of a few patterns is
        expected in any complex patient with multiple flagged biomarkers.
      - 4+ patterns: gentle linear penalty of 0.03 per pattern above 3.
      - Hard cap of 0.12 so even a 20-pattern report does not collapse
        all scores below the differential threshold.

    Replaces the previous flat _AMBIGUITY_PENALTY constant (0.10 applied
    at >= 3 patterns) which caused cliff-edge score collapse.

    Examples:
      n=1  -> 0.00
      n=3  -> 0.00
      n=4  -> 0.03
      n=5  -> 0.06
      n=6  -> 0.09
      n=7  -> 0.12  (capped)
      n=20 -> 0.12  (capped)
    """
    if n_patterns <= 3:
        return 0.0
    return min(0.03 * (n_patterns - 3), 0.12)


# ═══════════════════════════════════════════════════════════════════
# UNIVERSAL SCORING FORMULA
# ═══════════════════════════════════════════════════════════════════

def _score_pattern(
    pattern: dict,
    scoring_context: dict,
    total_matched_count: int,
) -> float:
    """
    Compute a confidence score in [0.0, 1.0] for a matched pattern.

    Formula:
      score = _BASE_SCORE
            + signal_boost        (capped at _MAX_SIGNAL_BOOST)
            + multi_trigger_boost (if pattern declares >= 2 trigger types)
            - _ambiguity_penalty(total_matched_count)

    Universal: no clinical-domain knobs. Same math for every pattern
    regardless of what it represents clinically.

    Fail-safe: any exception returns _BASE_SCORE so the pattern still
    appears in output rather than being silently excluded.
    """
    try:
        score = _BASE_SCORE

        # ── Signal boost ──────────────────────────────────────────────
        # Each corroborated trigger type adds _SIGNAL_BOOST_PER_TYPE,
        # capped at _MAX_SIGNAL_BOOST.
        supporting_signals = _count_matching_context_signals(
            pattern, scoring_context
        )
        signal_boost = min(
            supporting_signals * _SIGNAL_BOOST_PER_TYPE,
            _MAX_SIGNAL_BOOST,
        )
        score += signal_boost

        # ── Multi-trigger boost ───────────────────────────────────────
        # Patterns requiring multiple trigger types are more specific
        # and earn a modest extra boost.
        trigger_type_count = _count_trigger_types(pattern)
        if trigger_type_count >= 2:
            score += _MULTI_TRIGGER_BOOST

        # ── Ambiguity penalty ─────────────────────────────────────────
        # Gently scaled by _ambiguity_penalty() — does not collapse
        # scores at 3+ patterns the way a flat constant did.
        score -= _ambiguity_penalty(total_matched_count)

        # ── Clamp to valid range ──────────────────────────────────────
        return round(max(0.0, min(1.0, score)), 3)

    except Exception:
        logger.warning(
            "clinical_picture_synthesizer · _score_pattern failed; "
            "returning base score",
            pattern_id=pattern.get("id"),
        )
        return _BASE_SCORE


# ═══════════════════════════════════════════════════════════════════
# PATTERN RESOLUTION — read all sources, produce normalized entries
# ═══════════════════════════════════════════════════════════════════

def _resolve_all_matched_patterns(
    text_pattern_ids: list[str],
    clinical_pattern_ids: list[str],
    safety_rule_ids: list[str],
) -> list[dict]:
    """
    Fetch metadata for every matched pattern from all three registries
    and normalize into a common shape:
      {
        "id":              str,
        "source":          "text_finding" | "clinical_synthesis" | "safety",
        "narrative":       str | None,      # patient-facing full sentence
        "narrative_short": str | None,      # dashboard-friendly summary
        "triggers":        dict,            # for scoring
        "priority":        int,             # optional
      }
    """
    resolved: list[dict] = []

    # ── Text finding patterns ─────────────────────────────────────────
    for pid in (text_pattern_ids or []):
        raw = _lookup_text_finding_pattern(pid)
        if not raw:
            # Pattern ID exists but registry lookup failed — include
            # with minimal data so scoring can still assign base score.
            resolved.append({
                "id": pid,
                "source": "text_finding",
                "narrative": None,
                "narrative_short": None,
                "triggers": {},
                "priority": 0,
            })
            continue
        resolved.append({
            "id": pid,
            "source": "text_finding",
            "narrative": raw.get("observation"),
            "narrative_short": raw.get("narrative_short"),
            "triggers": raw.get("triggers", {}) or {},
            "priority": 0,
        })

    # ── Clinical synthesis patterns ───────────────────────────────────
    for pid in (clinical_pattern_ids or []):
        raw = _lookup_clinical_synthesis_pattern(pid)
        if not raw:
            resolved.append({
                "id": pid,
                "source": "clinical_synthesis",
                "narrative": None,
                "narrative_short": None,
                "triggers": {},
                "priority": 0,
            })
            continue
        resolved.append({
            "id": pid,
            "source": "clinical_synthesis",
            "narrative": raw.get("narrative"),
            "narrative_short": raw.get("narrative_short"),
            "triggers": raw.get("triggers", {}) or {},
            "priority": int(raw.get("priority") or 0),
        })

    # ── Safety escalation rules ───────────────────────────────────────
    # Always high-confidence because they already require both patient
    # condition AND biomarker match to fire.
    for rid in (safety_rule_ids or []):
        raw = _lookup_safety_escalation_rule(rid)
        if not raw:
            resolved.append({
                "id": rid,
                "source": "safety",
                "narrative": None,
                "narrative_short": None,
                "triggers": {},
                "priority": 100,
            })
            continue
        # Safety rules encode triggers differently — normalize to
        # the common shape so _count_trigger_types works correctly.
        triggers = {
            "patient_conditions": raw.get("requires_conditions", []),
            "biomarker": raw.get("requires_biomarker", {}),
        }
        resolved.append({
            "id": rid,
            "source": "safety",
            "narrative": raw.get("safety_warning"),
            "narrative_short": raw.get("safety_warning_short"),
            "triggers": triggers,
            "priority": 100,
        })

    return resolved


# ═══════════════════════════════════════════════════════════════════
# CATEGORIZATION — confident vs differential
# ═══════════════════════════════════════════════════════════════════

def _categorize_by_confidence(
    scored_patterns: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Split scored patterns into (confident_findings, differential).
    Both lists are sorted by descending confidence then descending
    priority.

    Confident   : score >= _CONFIDENT_THRESHOLD
    Differential: _DIFFERENTIAL_THRESHOLD <= score < _CONFIDENT_THRESHOLD
    Excluded    : score < _DIFFERENTIAL_THRESHOLD (not shown)
    """
    confident = [
        p for p in scored_patterns
        if p["confidence"] >= _CONFIDENT_THRESHOLD
    ]
    differential = [
        p for p in scored_patterns
        if _DIFFERENTIAL_THRESHOLD <= p["confidence"] < _CONFIDENT_THRESHOLD
    ]

    confident.sort(
        key=lambda p: (-p["confidence"], -p.get("priority", 0))
    )
    differential.sort(
        key=lambda p: (-p["confidence"], -p.get("priority", 0))
    )

    return confident, differential


# ═══════════════════════════════════════════════════════════════════
# NARRATIVE COMPOSITION
# ═══════════════════════════════════════════════════════════════════

def _compose_clinical_picture_summary(
    confident: list[dict],
    differential: list[dict],
) -> str:
    """
    Compose a single-paragraph synthesis of the clinical picture.
    Returns empty string when nothing meaningful to say.
    """
    if not confident and not differential:
        return ""

    parts: list[str] = []

    if confident:
        if len(confident) == 1:
            parts.append(
                "The findings converge on a single primary clinical picture."
            )
        else:
            parts.append(
                f"The findings support {len(confident)} concurrent "
                f"clinical patterns."
            )

    if differential:
        if not confident:
            parts.append(
                "The combination of findings does not converge on a single "
                "clear pattern; several possibilities are worth considering."
            )
        else:
            parts.append(
                f"In addition, {len(differential)} alternative "
                f"pattern{'s' if len(differential) != 1 else ''} "
                f"remain{'s' if len(differential) == 1 else ''} possible "
                f"and warrant{'s' if len(differential) == 1 else ''} "
                f"clinical consideration."
            )

    parts.append(
        "Clinical correlation with your full medical history and "
        "physician judgement remains essential."
    )

    return " ".join(parts)


def _format_pattern_line(p: dict) -> str:
    """
    Human-readable one-line description of a scored pattern for
    rendering. Format:
      "<narrative or ID-derived title> (confidence XX%)"
    """
    narrative = (p.get("narrative") or "").strip()
    if not narrative:
        pid = str(p.get("id") or "").strip()
        title = pid.replace("_", " ").title()
        narrative = f"Pattern: {title}"

    confidence = float(p.get("confidence") or 0.0)
    confidence_pct = int(round(confidence * 100))
    return f"{narrative} (confidence {confidence_pct}%)"


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════

def synthesize_clinical_picture(
    flagged_items: list[dict],
    symptoms: list[str],
    patient_context: dict,
    text_pattern_ids: list[str],
    clinical_pattern_ids: list[str],
    safety_rule_ids: list[str],
) -> dict:
    """
    Aggregate matched patterns across all three registries, score each
    by confidence, and produce a unified clinical picture.

    Args:
        flagged_items:         All flagged measurements (any non-normal status)
        symptoms:              Reported symptoms (strings)
        patient_context:       From tools.patient_context_adapter
        text_pattern_ids:      From state.text_finding_matched_patterns
        clinical_pattern_ids:  From clinical_synthesis.matched_pattern_ids
        safety_rule_ids:       From safety_result.matched_rule_ids

    Returns:
        {
          "confident_findings":       list[dict],
          "differential_findings":    list[dict],
          "clinical_picture_summary": str,
          "confidence_scores":        dict[str, float],
          "all_scored_patterns":      list[dict],
        }

    Each entry in confident_findings / differential_findings:
        {
          "id":              str,
          "source":          str,
          "narrative":       str | None,
          "narrative_short": str | None,
          "confidence":      float,
          "priority":        int,
        }

    Fail-safe: any error returns an empty result so downstream renders
    exactly as it would without this layer.
    """
    try:
        # ── Collect and normalize all matched patterns ─────────────────
        resolved = _resolve_all_matched_patterns(
            text_pattern_ids or [],
            clinical_pattern_ids or [],
            safety_rule_ids or [],
        )

        if not resolved:
            return {
                "confident_findings": [],
                "differential_findings": [],
                "clinical_picture_summary": "",
                "confidence_scores": {},
                "all_scored_patterns": [],
            }

        # ── Deduplicate by (id, source) ────────────────────────────────
        # The same pattern can theoretically appear from multiple sources
        # if IDs collide; keep the first occurrence.
        seen: set[tuple[str, str]] = set()
        unique: list[dict] = []
        for p in resolved:
            key = (p["id"], p["source"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(p)

        # ── Build scoring context ──────────────────────────────────────
        scoring_context = _build_scoring_context(
            flagged_items or [],
            symptoms or [],
            patient_context or {},
            text_pattern_ids or [],
        )

        total_matched = len(unique)

        # ── Score every pattern using the universal formula ────────────
        scored: list[dict] = []
        confidence_scores: dict[str, float] = {}

        for p in unique:
            score = _score_pattern(p, scoring_context, total_matched)
            entry = {
                "id":              p["id"],
                "source":          p["source"],
                "narrative":       p.get("narrative"),
                "narrative_short": p.get("narrative_short"),
                "confidence":      score,
                "priority":        p.get("priority", 0),
            }
            scored.append(entry)
            confidence_scores[p["id"]] = score

        # ── Categorize by confidence bands ─────────────────────────────
        confident, differential = _categorize_by_confidence(scored)

        # ── Compose summary paragraph ──────────────────────────────────
        summary = _compose_clinical_picture_summary(confident, differential)

        logger.info(
            "clinical_picture_synthesizer · complete",
            total_matched=total_matched,
            confident_count=len(confident),
            differential_count=len(differential),
            subthreshold=total_matched - len(confident) - len(differential),
        )

        return {
            "confident_findings":       confident,
            "differential_findings":    differential,
            "clinical_picture_summary": summary,
            "confidence_scores":        confidence_scores,
            "all_scored_patterns":      scored,
        }

    except Exception:
        logger.exception(
            "clinical_picture_synthesizer · synthesis failed; "
            "returning empty result"
        )
        return {
            "confident_findings":       [],
            "differential_findings":    [],
            "clinical_picture_summary": "",
            "confidence_scores":        {},
            "all_scored_patterns":      [],
        }


__all__ = ["synthesize_clinical_picture"]