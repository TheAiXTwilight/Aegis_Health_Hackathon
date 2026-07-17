"""
tools/dynamic_recommendations.py — Section 10 recommendation synthesizer.

Builds the final "Recommendations" section of the report from:
  - Top flagged findings (most clinically urgent)
  - Reported symptoms
  - Patient context (allergies, conditions, pregnancy, age)
  - Severity level
  - Text finding patterns (peripheral smear / impressions)
  - Safety escalations

Design principles:
  - Universal: no biomarker-specific hardcoded prose. All references
    to biomarkers use the item's runtime 'name' field.
  - Composable: small phrase-builders that each produce one clinical
    sentence; orchestrator assembles them.
  - Fail-safe: any error returns static severity-based recommendations
    so the report is NEVER left without a Recommendations section.
  - Data-driven: red flag detection uses a registry so new clinical
    patterns are added as dict entries, not code.
  - Preserves existing shape: returns list[str] matching the current
    _build_deterministic_report() recommendation_lines contract.

Consumed by:
  tools.report_generator._build_deterministic_report() — Section 10
"""
from __future__ import annotations

import re as _re
from typing import Any

from loguru import logger
from tools.unit_normalizer import normalize_for_comparison, _resolve_canonical_key


# ═══════════════════════════════════════════════════════════════════
# MODULE-LEVEL IMPORTS FOR CROSS-MODULE PATTERN REGISTRIES
# ═══════════════════════════════════════════════════════════════════
# Imported at module level so any failure surfaces at startup in logs
# rather than being silently swallowed inside a try/except at runtime.
# Each import has a graceful fallback so the module still loads even
# if a dependency is broken — but the failure is now VISIBLE.

try:
    from tools.clinical_synthesis import (
        _CLINICAL_PATTERNS as _CS_CLINICAL_PATTERNS,
        _pattern_matches   as _cs_pattern_matches,
    )
    _CS_IMPORT_OK = True
except Exception as _cs_err:
    logger.error(
        "dynamic_recommendations · clinical_synthesis import failed at "
        "module load — clinical pattern boost (bucket 2) will be disabled. "
        "Error: {}",
        str(_cs_err),
    )
    _CS_CLINICAL_PATTERNS = []  # type: ignore[assignment]
    _cs_pattern_matches   = None
    _CS_IMPORT_OK         = False

try:
    from tools.patient_context_adapter import (
        _SAFETY_ESCALATION_RULES as _SAFETY_RULES,
    )
    _SAFETY_IMPORT_OK = True
except Exception as _safety_err:
    logger.error(
        "dynamic_recommendations · patient_context_adapter import failed at "
        "module load — safety escalation boost (bucket 0) will be disabled. "
        "Error: {}",
        str(_safety_err),
    )
    _SAFETY_RULES     = []  # type: ignore[assignment]
    _SAFETY_IMPORT_OK = False


# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

_TIER_CRITICAL  = 0
_TIER_OUT_RANGE = 1
_TIER_BORDERLINE = 2


def _status_tier_of(item: dict) -> int:
    """Local copy of tier logic to avoid circular imports."""
    status = str(item.get("status") or "").lower()
    if status in ("critical_low", "critical_high", "critical"):
        return _TIER_CRITICAL
    if status in ("high", "low", "abnormal"):
        return _TIER_OUT_RANGE
    if status == "borderline":
        return _TIER_BORDERLINE
    return 99


# ═══════════════════════════════════════════════════════════════════
# SYMPTOM DISPLAY NORMALIZATION MAP
# ═══════════════════════════════════════════════════════════════════
# Applied at DISPLAY LAYER ONLY inside _build_symptoms_context_phrase.
# Never mutates the upstream symptoms list used by pattern matching.

_SYMPTOM_DISPLAY_MAP: dict[str, str] = {
    "tired":               "fatigue",
    "tiredness":           "fatigue",
    "fatigue":             "fatigue",
    "dizzy":               "dizziness",
    "dizziness":           "dizziness",
    "lightheaded":         "lightheadedness",
    "light headed":        "lightheadedness",
    "light-headed":        "lightheadedness",
    "nauseated":           "nausea",
    "nausea":              "nausea",
    "vomiting":            "vomiting",
    "throwing up":         "vomiting",
    "breathless":          "breathlessness",
    "breathlessness":      "breathlessness",
    "sob":                 "shortness of breath",
    "short of breath":     "shortness of breath",
    "shortness of breath": "shortness of breath",
    "chest tightness":     "chest tightness",
    "chest pain":          "chest pain",
    "chest pressure":      "chest pressure",
    "sweating":            "diaphoresis",
    "sweaty":              "diaphoresis",
    "night sweats":        "night sweats",
    "fever":               "fever",
    "high temperature":    "fever",
    "high temp":           "fever",
    "chills":              "chills",
    "rigors":              "rigors",
    "headache":            "headache",
    "head ache":           "headache",
    "joint pain":          "joint pain",
    "joint ache":          "joint pain",
    "muscle pain":         "myalgia",
    "muscle ache":         "myalgia",
    "body ache":           "myalgia",
    "body pain":           "myalgia",
    "weak":                "weakness",
    "weakness":            "weakness",
    "pale":                "pallor",
    "pallor":              "pallor",
    "swollen":             "swelling",
    "swelling":            "swelling",
    "bloated":             "bloating",
    "bloating":            "bloating",
    "constipated":         "constipation",
    "constipation":        "constipation",
    "diarrhea":            "diarrhoea",
    "diarrhoea":           "diarrhoea",
    "loose stools":        "diarrhoea",
    "indigestion":         "indigestion",
    "heartburn":           "heartburn",
    "palpitations":        "palpitations",
    "heart racing":        "palpitations",
    "racing heart":        "palpitations",
    "hair loss":           "hair loss",
    "hair fall":           "hair loss",
    "weight gain":         "weight gain",
    "weight loss":         "weight loss",
    "frequent urination":  "polyuria",
    "increased thirst":    "polydipsia",
    "excessive thirst":    "polydipsia",
    "blurred vision":      "blurred vision",
    "blurry vision":       "blurred vision",
    "numbness":            "numbness",
    "tingling":            "tingling",
    "anxiety":             "anxiety",
    "depression":          "low mood",
    "low mood":            "low mood",
    "insomnia":            "insomnia",
    "poor sleep":          "insomnia",
    "sleep problems":      "insomnia",
    "cold":                "cold symptoms",
    "runny nose":          "rhinorrhoea",
    "stuffy nose":         "nasal congestion",
    "sore throat":         "sore throat",
    "cough":               "cough",
}


# ═══════════════════════════════════════════════════════════════════
# RED FLAG REGISTRY
# ═══════════════════════════════════════════════════════════════════

_RED_FLAG_PATTERNS: list[dict[str, Any]] = [
    {
        "id": "fever_low_platelets_viral",
        "triggers": {
            "symptoms": ["fever", "high temperature", "temperature", "chills"],
            "biomarker": {"key": "platelets", "op": "<", "value": 150},
        },
        "red_flags": [
            "worsening bleeding tendency (gum bleeding, nose bleeds, bruising)",
            "persistent fever above 102°F (39°C)",
            "severe abdominal pain",
            "breathing difficulty",
            "new petechiae or rashes",
            "severe weakness or dizziness",
        ],
        "specific_action": (
            "physician evaluation within 24–48 hours to rule out viral "
            "fever including dengue"
        ),
    },
    {
        "id": "chest_pain_or_sob",
        "triggers": {
            "symptoms": [
                "chest pain", "chest tightness", "chest pressure",
                "shortness of breath", "difficulty breathing",
            ],
        },
        "red_flags": [
            "chest pain radiating to the arm, jaw, or back",
            "severe breathing difficulty",
            "cold sweats or clamminess",
            "fainting or near-fainting",
            "irregular heartbeat",
        ],
        "specific_action": (
            "emergency evaluation without delay — chest and breathing "
            "symptoms warrant immediate assessment"
        ),
    },
    {
        "id": "diabetic_symptoms",
        "triggers": {
            "patient_conditions": ["diabetes"],
            "biomarker": {"key": "glucose", "op": ">", "value": 250},
        },
        "red_flags": [
            "extreme thirst or frequent urination",
            "nausea or vomiting",
            "fruity-smelling breath",
            "confusion or drowsiness",
            "rapid deep breathing",
        ],
        "specific_action": (
            "urgent physician review for possible diabetic emergency; "
            "seek emergency care if any listed red flag develops"
        ),
    },
    {
        "id": "anticoagulated_bleeding_risk",
        "triggers": {
            "patient_conditions": ["anticoagulation"],
            "biomarker": {"key": "platelets", "op": "<", "value": 100},
        },
        "red_flags": [
            "any unusual bruising",
            "blood in urine or stool",
            "prolonged bleeding from minor cuts",
            "severe headache (possible bleeding)",
            "black or tarry stools",
        ],
        "specific_action": (
            "contact your prescribing physician urgently to review "
            "anticoagulation given elevated bleeding risk"
        ),
    },
    {
        "id": "severe_anemia",
        "triggers": {
            "biomarker": {"key": "haemoglobin", "op": "<", "value": 8},
        },
        "red_flags": [
            "severe fatigue or weakness",
            "shortness of breath at rest",
            "chest pain or palpitations",
            "dizziness or fainting",
            "very pale skin or nail beds",
        ],
        "specific_action": (
            "urgent medical evaluation for severe anemia; blood work and "
            "possible transfusion consideration may be required"
        ),
    },
    {
        "id": "hyperkalemia",
        "triggers": {
            "biomarker": {"key": "potassium", "op": ">", "value": 6.0},
        },
        "red_flags": [
            "muscle weakness or paralysis",
            "irregular heartbeat or palpitations",
            "chest pain",
            "nausea",
        ],
        "specific_action": (
            "emergency evaluation — critically elevated potassium can cause "
            "dangerous heart rhythm disturbances"
        ),
    },
    {
        "id": "critical_troponin",
        "triggers": {
            "biomarker": {"key": "troponin", "op": ">", "value": 0.04},
        },
        "red_flags": [
            "chest pain or pressure",
            "pain radiating to arm, jaw, or back",
            "shortness of breath",
            "cold sweats",
            "nausea",
        ],
        "specific_action": (
            "emergency cardiology evaluation immediately — elevated troponin "
            "may indicate cardiac injury"
        ),
    },
    {
        "id": "pregnancy_low_platelets",
        "triggers": {
            "patient_conditions": ["pregnancy"],
            "biomarker": {"key": "platelets", "op": "<", "value": 150},
        },
        "red_flags": [
            "severe headache",
            "visual changes or blurred vision",
            "swelling of face or hands",
            "upper abdominal pain",
            "reduced fetal movement",
        ],
        "specific_action": (
            "urgent obstetric review to rule out pre-eclampsia or other "
            "pregnancy complications"
        ),
    },
    {
        "id": "neutropenic_fever_risk",
        "triggers": {
            "patient_conditions": ["chemotherapy", "immunocompromised"],
            "biomarker": {"key": "neutrophils", "op": "<", "value": 1.5},
        },
        "red_flags": [
            "any fever above 100.4°F (38°C)",
            "chills or rigors",
            "sore throat or mouth sores",
            "productive cough",
            "burning urination",
        ],
        "specific_action": (
            "emergency care immediately if febrile — neutropenic fever is "
            "a medical emergency"
        ),
    },
    {
        "id": "smear_blast_cells_or_auer",
        "triggers": {
            "text_patterns": ["smear_blast_cells", "smear_auer_rods"],
        },
        "red_flags": [
            "unusual bruising or bleeding",
            "persistent fever",
            "extreme fatigue",
            "bone pain",
            "swollen lymph nodes",
        ],
        "specific_action": (
            "emergency hematology evaluation — findings suggest possible "
            "acute leukemia requiring immediate workup"
        ),
    },
    {
        "id": "malaria_parasite_detected",
        "triggers": {
            "text_patterns": ["smear_malaria_parasite"],
        },
        "red_flags": [
            "high fever with chills and rigors",
            "confusion or drowsiness",
            "dark or reduced urine",
            "yellowing of skin or eyes",
            "severe weakness",
        ],
        "specific_action": (
            "urgent antimalarial therapy required; seek immediate medical care"
        ),
    },
    {
        "id": "suspicious_malignancy",
        "triggers": {
            "text_patterns": ["impression_suggestive_of_malignancy"],
        },
        "red_flags": [
            "unexplained weight loss",
            "persistent unusual bleeding",
            "new or growing lumps",
            "night sweats",
            "persistent fatigue",
        ],
        "specific_action": (
            "urgent oncology consultation within 1 week for confirmatory "
            "workup and biopsy"
        ),
    },
]


_DEFAULT_RED_FLAGS = [
    "worsening or new severe symptoms",
    "breathing difficulty",
    "chest pain or pressure",
    "confusion or altered consciousness",
    "fainting or severe dizziness",
    "uncontrolled bleeding",
    "persistent high fever",
    "signs of dehydration",
]


# ═══════════════════════════════════════════════════════════════════
# UNIVERSAL KEY MATCHER
# ═══════════════════════════════════════════════════════════════════

def _keys_match(m: dict, key_wanted: str, wanted_resolved: str) -> bool:
    """
    Universal three-layer key matcher.

    Layer 1: exact key string match
    Layer 2: alias resolution match (both sides resolved)
    Layer 3: name-based substring match — handles cases where the
             dashboard extractor stores a long canonical slug that
             doesn't round-trip through the alias resolver back to
             the short registry key used in pattern triggers.

    This function is the single place where key matching logic lives.
    All comparison helpers (_biomarker_match, _one_biomarker_condition_met,
    _item_key) delegate here so the fix applies universally.

    Fail-safe: any error returns False (conservative — no false matches).
    """
    try:
        m_key  = str(m.get("key")  or "").lower()
        m_name = str(m.get("name") or "").lower()

        # Layer 1: exact
        if m_key == key_wanted:
            return True

        # Layer 2: alias resolution
        if _resolve_canonical_key(m_key) == wanted_resolved:
            return True

        # Layer 3: name-based
        # key_wanted with underscores → spaces for natural name matching
        key_as_words = key_wanted.replace("_", " ")
        if key_as_words in m_name:
            return True
        if key_wanted in m_name:
            return True

        return False

    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════
# TRIGGER MATCHERS
# ═══════════════════════════════════════════════════════════════════

def _symptoms_match(triggers: dict, symptoms: list[str]) -> bool:
    keywords = triggers.get("symptoms") or []
    if not keywords:
        return True
    symptoms_lower = [str(s).lower() for s in symptoms if s]
    if not symptoms_lower:
        return False
    combined = " ".join(symptoms_lower)
    return any(kw.lower() in combined for kw in keywords)


def _biomarker_match(condition: dict, measurements: list[dict]) -> bool:
    """
    Universal biomarker condition checker.

    Uses _keys_match for three-layer key resolution so biomarkers
    with aliased key slugs (e.g. TSH stored as any variant) are
    correctly identified regardless of what the parser stored.

    Supports numeric threshold and status-string comparisons.
    Routes numeric comparisons through normalize_for_comparison()
    for correct unit handling.
    Fail-safe: per-measurement errors are skipped.
    """
    key_wanted = str(condition.get("key") or "").lower()
    op         = condition.get("op")
    threshold  = condition.get("value")

    if not key_wanted or not op or threshold is None:
        return False

    wanted_resolved = _resolve_canonical_key(key_wanted)

    # ── Status string comparison ──────────────────────────────────
    if op == "status":
        target_status = str(threshold).lower()
        for m in measurements:
            if not _keys_match(m, key_wanted, wanted_resolved):
                continue
            if str(m.get("status") or "").lower() == target_status:
                return True
        return False

    # ── Numeric threshold comparison ──────────────────────────────
    try:
        threshold_f = float(threshold)
    except (TypeError, ValueError):
        return False

    for m in measurements:
        if not _keys_match(m, key_wanted, wanted_resolved):
            continue

        m_key     = str(m.get("key") or "").lower()
        raw_value = m.get("value")
        raw_unit  = m.get("unit") or m.get("units") or ""

        try:
            norm = normalize_for_comparison(m_key, raw_value, raw_unit)
            v = norm.value
        except Exception:
            try:
                v = float(raw_value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue

        if v is None:
            continue

        try:
            if op == ">"  and v >  threshold_f: return True
            if op == "<"  and v <  threshold_f: return True
            if op == ">=" and v >= threshold_f: return True
            if op == "<=" and v <= threshold_f: return True
            if op == "==" and v == threshold_f: return True
        except TypeError:
            continue

    return False


def _conditions_match(triggers: dict, patient_context: dict) -> bool:
    required = triggers.get("patient_conditions") or []
    if not required:
        return True
    patient_conditions = set(patient_context.get("conditions") or [])
    return bool(set(required) & patient_conditions)


def _text_patterns_match(triggers: dict, text_pattern_ids: list[str]) -> bool:
    required = triggers.get("text_patterns") or []
    if not required:
        return True
    matched = set(text_pattern_ids or [])
    return bool(set(required) & matched)


def _pattern_matches(
    pattern: dict,
    symptoms: list[str],
    flagged_items: list[dict],
    patient_context: dict,
    text_pattern_ids: list[str],
) -> bool:
    triggers = pattern.get("triggers") or {}
    checks = []
    if "symptoms" in triggers:
        checks.append(_symptoms_match(triggers, symptoms))
    if "biomarker" in triggers:
        checks.append(_biomarker_match(triggers["biomarker"], flagged_items))
    if "patient_conditions" in triggers:
        checks.append(_conditions_match(triggers, patient_context))
    if "text_patterns" in triggers:
        checks.append(_text_patterns_match(triggers, text_pattern_ids))
    return len(checks) > 0 and all(checks)


# ═══════════════════════════════════════════════════════════════════
# PRIORITY BOOST HELPERS
# ═══════════════════════════════════════════════════════════════════

def _get_safety_escalation_biomarker_keys(
    patient_context: dict,
    flagged_items: list[dict],
) -> set[str]:
    """
    Return canonical biomarker keys implicated in any active safety
    escalation rule. Uses module-level _SAFETY_RULES import.
    Fail-safe: returns empty set on any error.
    """
    if not _SAFETY_IMPORT_OK or not _SAFETY_RULES:
        return set()

    try:
        patient_conditions = set(patient_context.get("conditions") or [])
        boosted_keys: set[str] = set()

        for rule in _SAFETY_RULES:
            required = set(rule.get("requires_conditions") or [])
            if required and not (required & patient_conditions):
                continue

            biomarker_cond = rule.get("requires_biomarker") or {}
            if not biomarker_cond:
                continue

            if _biomarker_match(biomarker_cond, flagged_items):
                key = str(biomarker_cond.get("key") or "").lower()
                if key:
                    boosted_keys.add(key)

        return boosted_keys

    except Exception:
        logger.warning(
            "dynamic_recommendations · _get_safety_escalation_biomarker_keys "
            "failed; no safety boost applied"
        )
        return set()


def _get_clinical_synthesis_correlated_keys(
    flagged_items: list[dict],
    symptoms: list[str],
    patient_context: dict,
    text_pattern_ids: list[str],
) -> set[str]:
    """
    Return canonical biomarker keys implicated in any matched clinical
    synthesis pattern. Uses module-level _CS_CLINICAL_PATTERNS import
    so failures are visible in startup logs, not silently swallowed.

    Fail-safe: returns empty set on any error.
    """
    if not _CS_IMPORT_OK or not _CS_CLINICAL_PATTERNS or _cs_pattern_matches is None:
        return set()

    try:
        boosted_keys: set[str] = set()

        for pattern in _CS_CLINICAL_PATTERNS:
            try:
                if not _cs_pattern_matches(
                    pattern, symptoms, flagged_items,
                    patient_context, text_pattern_ids,
                ):
                    continue
            except Exception:
                continue

            triggers = pattern.get("triggers") or {}

            biomarker_cond = triggers.get("biomarker") or {}
            key = str(biomarker_cond.get("key") or "").lower()
            if key:
                boosted_keys.add(key)

            for cond in (triggers.get("biomarker_combo") or []):
                key = str(cond.get("key") or "").lower()
                if key:
                    boosted_keys.add(key)

        logger.debug(
            "dynamic_recommendations · clinical pattern boost keys",
            keys=sorted(boosted_keys),
        )
        return boosted_keys

    except Exception:
        logger.warning(
            "dynamic_recommendations · _get_clinical_synthesis_correlated_keys "
            "failed; no clinical pattern boost applied"
        )
        return set()


def _get_symptom_correlated_keys(
    symptoms: list[str],
    flagged_items: list[dict],
) -> set[str]:
    """
    Return canonical biomarker keys whose red-flag pattern biomarker
    trigger fires AND whose symptom trigger also matches.
    Fail-safe: returns empty set on any error.
    """
    try:
        correlated_keys: set[str] = set()
        for pattern in _RED_FLAG_PATTERNS:
            triggers = pattern.get("triggers") or {}
            if "symptoms" not in triggers or "biomarker" not in triggers:
                continue
            if not _symptoms_match(triggers, symptoms):
                continue
            if not _biomarker_match(triggers["biomarker"], flagged_items):
                continue
            key = str(triggers["biomarker"].get("key") or "").lower()
            if key:
                correlated_keys.add(key)
        return correlated_keys
    except Exception:
        logger.warning(
            "dynamic_recommendations · _get_symptom_correlated_keys failed"
        )
        return set()


def _get_text_pattern_correlated_keys(
    text_pattern_ids: list[str],
    flagged_items: list[dict],
) -> set[str]:
    """
    Return canonical biomarker keys referenced by any red-flag pattern
    whose text_patterns trigger fired.
    Fail-safe: returns empty set on any error.
    """
    try:
        if not text_pattern_ids:
            return set()
        matched_text_ids = set(text_pattern_ids)
        correlated_keys: set[str] = set()
        for pattern in _RED_FLAG_PATTERNS:
            triggers = pattern.get("triggers") or {}
            required_text = triggers.get("text_patterns") or []
            if not required_text:
                continue
            if not (set(required_text) & matched_text_ids):
                continue
            biomarker_cond = triggers.get("biomarker") or {}
            key = str(biomarker_cond.get("key") or "").lower()
            if key:
                correlated_keys.add(key)
        return correlated_keys
    except Exception:
        logger.warning(
            "dynamic_recommendations · _get_text_pattern_correlated_keys failed"
        )
        return set()


# ═══════════════════════════════════════════════════════════════════
# PHRASE BUILDERS
# ═══════════════════════════════════════════════════════════════════

def _build_top_findings_phrase(
    flagged_items: list[dict],
    max_items: int = 3,
    patient_context: dict | None = None,
    text_pattern_ids: list[str] | None = None,
    symptoms: list[str] | None = None,
) -> str:
    """
    Enumerate top flagged findings by clinical priority.

    Bucket ordering (highest → lowest priority):
      0 — Safety escalation (active safety rule for this patient)
      1 — Symptom-correlated (red flag pattern with symptom gate)
      2 — Clinical synthesis pattern (e.g. TSH in hypothyroid pattern)
      3 — Text-pattern-correlated
      4 — Critical status (critical_high / critical_low)
      5 — Fallback: tier + alphabetical

    Within each bucket: sorted by (tier, name).

    Key membership check uses _in_boost_set() which applies the same
    three-layer matching as _keys_match() so findings with aliased key
    slugs are correctly boosted regardless of parser key format.
    """
    if not flagged_items:
        return ""

    _patient_context  = patient_context  or {}
    _text_pattern_ids = text_pattern_ids or []
    _symptoms         = symptoms         or []

    try:
        safety_keys = _get_safety_escalation_biomarker_keys(
            _patient_context, flagged_items,
        )
    except Exception:
        safety_keys = set()

    try:
        symptom_corr_keys = _get_symptom_correlated_keys(
            _symptoms, flagged_items,
        )
    except Exception:
        symptom_corr_keys = set()

    try:
        clinical_pattern_keys = _get_clinical_synthesis_correlated_keys(
            flagged_items, _symptoms, _patient_context, _text_pattern_ids,
        )
    except Exception:
        clinical_pattern_keys = set()

    try:
        text_corr_keys = _get_text_pattern_correlated_keys(
            _text_pattern_ids, flagged_items,
        )
    except Exception:
        text_corr_keys = set()

    def _in_boost_set(it: dict, boost_set: set[str]) -> bool:
        """
        Check if a flagged item belongs to a boost key set.

        Uses the same three-layer matching logic as _keys_match so
        items with non-standard key slugs are correctly boosted.
        Universal: no biomarker-specific logic here.
        """
        if not boost_set:
            return False
        raw_key = str(it.get("key")  or "").lower()
        m_name  = str(it.get("name") or "").lower()
        resolved = _resolve_canonical_key(raw_key)

        # Layer 1 + 2: key and alias
        if raw_key in boost_set or resolved in boost_set:
            return True

        # Layer 3: name-based — any boost key appears in item name
        for bk in boost_set:
            bk_words = bk.replace("_", " ")
            if bk_words in m_name or bk in m_name:
                return True

        return False

    def _item_sort_key(it: dict) -> tuple:
        status = str(it.get("status") or "").lower()
        name   = str(it.get("name") or it.get("key") or "").lower()
        tier   = _status_tier_of(it)

        if _in_boost_set(it, safety_keys):
            bucket = 0
        elif _in_boost_set(it, symptom_corr_keys):
            bucket = 1
        elif _in_boost_set(it, clinical_pattern_keys):
            bucket = 2
        elif _in_boost_set(it, text_corr_keys):
            bucket = 3
        elif status in ("critical_high", "critical_low", "critical"):
            bucket = 4
        else:
            bucket = 5

        return (bucket, tier, name)

    try:
        ordered = sorted(flagged_items, key=_item_sort_key)
    except Exception:
        logger.warning(
            "dynamic_recommendations · boost sort failed; using fallback"
        )
        ordered = sorted(flagged_items, key=lambda it: (
            _status_tier_of(it),
            (it.get("name") or "").lower(),
        ))

    top = ordered[:max_items]

    parts: list[str] = []
    for it in top:
        name   = str(it.get("name") or it.get("key") or "").strip()
        if not name:
            continue
        status = str(it.get("status") or "").lower()
        if status == "critical_low":
            parts.append(f"critically low {name}")
        elif status == "low":
            parts.append(f"low {name}")
        elif status == "critical_high":
            parts.append(f"critically elevated {name}")
        elif status == "high":
            parts.append(f"elevated {name}")
        elif status == "borderline":
            parts.append(f"borderline {name}")
        elif status == "critical":
            parts.append(f"critical {name}")
        else:
            parts.append(name)

    if not parts:
        return ""
    if len(parts) == 1:
        return f"your {parts[0]}"
    if len(parts) == 2:
        return f"your {parts[0]} and {parts[1]}"
    return "your " + ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _build_symptoms_context_phrase(
    symptoms: list[str],
    flagged_items: list[dict],
) -> str:
    """
    Produce a linking phrase from symptoms for Section 10.
    Applies _SYMPTOM_DISPLAY_MAP at display layer only.
    """
    if not symptoms:
        return ""

    symptoms_clean = [str(s).strip() for s in symptoms if s]
    if not symptoms_clean:
        return ""

    symptoms_display = [
        _SYMPTOM_DISPLAY_MAP.get(s.lower(), s)
        for s in symptoms_clean
    ]

    seen: set[str] = set()
    symptoms_deduped: list[str] = []
    for s in symptoms_display:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            symptoms_deduped.append(s)

    if not symptoms_deduped:
        return ""

    if len(symptoms_deduped) == 1:
        symptoms_phrase = symptoms_deduped[0]
    elif len(symptoms_deduped) == 2:
        symptoms_phrase = f"{symptoms_deduped[0]} and {symptoms_deduped[1]}"
    else:
        symptoms_phrase = (
            ", ".join(symptoms_deduped[:-1]) + f", and {symptoms_deduped[-1]}"
        )

    if flagged_items:
        return f"Given your reported {symptoms_phrase} combined with"
    return f"Given your reported {symptoms_phrase},"


def _build_patient_context_phrase(patient_context: dict) -> str:
    if not patient_context:
        return ""
    conditions = patient_context.get("conditions") or []
    if not conditions:
        return ""
    labels = {
        "diabetes":          "diabetes",
        "hypertension":      "high blood pressure",
        "hypothyroidism":    "hypothyroidism",
        "hyperthyroidism":   "hyperthyroidism",
        "ckd":               "kidney disease",
        "anticoagulation":   "anticoagulation therapy",
        "pregnancy":         "pregnancy",
        "cardiovascular":    "cardiovascular history",
        "chemotherapy":      "ongoing cancer treatment",
        "autoimmune":        "autoimmune condition",
        "copd_asthma":       "respiratory condition",
        "liver_disease":     "liver disease",
        "immunocompromised": "immunocompromised state",
    }
    condition_phrases = [labels.get(c, c) for c in conditions if labels.get(c, c)]
    if not condition_phrases:
        return ""
    if len(condition_phrases) == 1:
        return f"given your {condition_phrases[0]}"
    if len(condition_phrases) == 2:
        return f"given your {condition_phrases[0]} and {condition_phrases[1]}"
    return f"given your {', '.join(condition_phrases[:-1])}, and {condition_phrases[-1]}"


def _build_severity_action_phrase(severity_level: str) -> str:
    lvl = str(severity_level or "").upper()
    if lvl == "HIGH":
        return "urgent medical evaluation is recommended as soon as possible"
    if lvl in ("MODERATE", "MEDIUM"):
        return "a physician evaluation within 24–48 hours is advised"
    if lvl == "LOW":
        return "routine follow-up with a physician is advised"
    return "clinical review is advised"


def _select_red_flag_pattern(
    symptoms: list[str],
    flagged_items: list[dict],
    patient_context: dict,
    text_pattern_ids: list[str],
) -> dict | None:
    for pattern in _RED_FLAG_PATTERNS:
        try:
            if _pattern_matches(
                pattern, symptoms, flagged_items,
                patient_context, text_pattern_ids,
            ):
                return pattern
        except Exception:
            continue
    return None


def _build_red_flags_phrase(pattern: dict | None) -> str:
    flags = (pattern.get("red_flags") if pattern else None) or _DEFAULT_RED_FLAGS
    return f"Watch for red flags: {', '.join(flags)}."


def _build_specific_action_phrase(pattern: dict | None) -> str:
    if not pattern:
        return ""
    action = str(pattern.get("specific_action") or "").strip()
    if not action:
        return ""
    return action[:1].upper() + action[1:]


# ═══════════════════════════════════════════════════════════════════
# STATIC FALLBACK
# ═══════════════════════════════════════════════════════════════════

def _static_recommendations(
    severity_level: str,
    input_is_symptom_only: bool,
    review_materials_text: str,
) -> list[str]:
    lvl = str(severity_level or "").upper()
    if lvl == "LOW":
        if input_is_symptom_only:
            return [
                "- Continue monitoring your symptoms over the next 24 to 48 hours.",
                "- Maintain hydration, adequate rest, and supportive care.",
                "- Track temperature, cough, breathing comfort, appetite, and overall energy level.",
                "- Consult a qualified healthcare professional if symptoms persist, worsen, or new symptoms appear.",
                "- Seek urgent medical care if breathing difficulty, chest pain, confusion, fainting, bluish lips, severe weakness, dehydration, persistent high fever, or rapid worsening develops.",
                "- If lab reports, X-ray images, or medication details become available later, repeat the assessment with those included.",
            ]
        return [
            "- Follow up with a qualified healthcare professional if symptoms persist or worsen.",
            "- Discuss the submitted lab, imaging, or medication findings with a clinician for full interpretation.",
            "- Continue monitoring symptoms and overall condition over the next 24 to 48 hours.",
            "- Seek urgent medical care if red-flag symptoms such as breathing difficulty, chest pain, confusion, fainting, or rapid worsening develop.",
        ]
    if lvl in ("MODERATE", "MEDIUM"):
        return [
            "- Arrange a timely review with a qualified healthcare professional.",
            "- Monitor symptoms closely and do not delay care if symptoms progress.",
            "- Keep a record of temperature, breathing symptoms, medications taken, and any new symptoms.",
            "- Seek urgent medical attention if red-flag symptoms such as breathing difficulty, chest pain, confusion, fainting, or severe worsening occur.",
            f"- Bring the {review_materials_text} and this generated summary to the clinician for review.",
        ]
    return [
        "- Seek urgent medical evaluation as soon as possible.",
        "- Do not rely on automated triage alone for high-risk symptoms.",
        "- Share the submitted symptoms, uploaded reports, and this generated summary with a qualified clinician.",
        "- If symptoms are severe or rapidly worsening, use emergency medical services immediately.",
    ]


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API — ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

def build_dynamic_recommendations(
    severity_level: str,
    flagged_items: list[dict],
    symptoms: list[str],
    patient_context: dict,
    text_pattern_ids: list[str],
    input_is_symptom_only: bool,
    review_materials_text: str,
) -> list[str]:
    """
    Synthesize Section 10 recommendation lines from all available context.

    Returns list[str] with "- " prefix matching the existing contract.
    Fail-safe: on any exception returns static severity-based fallback.
    """
    try:
        symptoms         = symptoms         or []
        flagged_items    = flagged_items    or []
        patient_context  = patient_context  or {}
        text_pattern_ids = text_pattern_ids or []

        lines: list[str] = []

        symptoms_phrase = _build_symptoms_context_phrase(symptoms, flagged_items)
        findings_phrase = _build_top_findings_phrase(
            flagged_items,
            max_items=3,
            patient_context=patient_context,
            text_pattern_ids=text_pattern_ids,
            symptoms=symptoms,
        )
        patient_phrase = _build_patient_context_phrase(patient_context)
        urgency_phrase = _build_severity_action_phrase(severity_level)

        matched_pattern = _select_red_flag_pattern(
            symptoms, flagged_items, patient_context, text_pattern_ids,
        )
        specific_action = _build_specific_action_phrase(matched_pattern)

        summary_parts: list[str] = []
        if symptoms_phrase and findings_phrase:
            summary_parts.append(f"{symptoms_phrase} {findings_phrase},")
        elif findings_phrase:
            summary_parts.append(f"Given {findings_phrase},")
        elif symptoms_phrase:
            summary_parts.append(symptoms_phrase)

        if patient_phrase:
            summary_parts.append(patient_phrase + ",")

        if specific_action:
            summary_parts.append(specific_action + ".")
        elif urgency_phrase:
            summary_parts.append(urgency_phrase + ".")

        if summary_parts:
            summary_line = " ".join(summary_parts).strip()
            summary_line = _re.sub(r"\s+", " ", summary_line)
            summary_line = _re.sub(r"\s+,", ",", summary_line)
            summary_line = _re.sub(r",\s*\.", ".", summary_line)
            summary_line = summary_line[:1].upper() + summary_line[1:]
            lines.append(f"- {summary_line}")

        red_flags_line = _build_red_flags_phrase(matched_pattern)
        if red_flags_line:
            lines.append(f"- {red_flags_line}")

        if not input_is_symptom_only:
            lines.append(
                f"- Bring the {review_materials_text} and this generated "
                f"summary to the clinician for review."
            )
        else:
            lines.append(
                "- If lab reports, imaging, or medication details become "
                "available later, repeat this assessment with those included."
            )

        lvl = str(severity_level or "").upper()
        if lvl == "HIGH":
            lines.append(
                "- Do not delay: if any listed red flag develops, use "
                "emergency medical services immediately."
            )
        elif lvl in ("MODERATE", "MEDIUM"):
            lines.append(
                "- Monitor symptoms closely over the next 24–48 hours and "
                "seek prompt care if any red flag develops."
            )
        else:
            lines.append(
                "- Continue routine self-monitoring; escalate to medical "
                "care if any red flag develops."
            )

        if len(lines) < 2:
            logger.warning(
                "dynamic_recommendations · sparse output; using static fallback"
            )
            return _static_recommendations(
                severity_level, input_is_symptom_only, review_materials_text,
            )

        logger.info(
            "dynamic_recommendations · synthesized",
            line_count=len(lines),
            matched_pattern=(matched_pattern or {}).get("id"),
            clinical_boost_keys=sorted(
                _get_clinical_synthesis_correlated_keys(
                    flagged_items, symptoms, patient_context, text_pattern_ids
                )
            ),
        )
        return lines

    except Exception:
        logger.exception(
            "dynamic_recommendations · synthesis failed; using static fallback"
        )
        return _static_recommendations(
            severity_level, input_is_symptom_only, review_materials_text,
        )


__all__ = ["build_dynamic_recommendations"]