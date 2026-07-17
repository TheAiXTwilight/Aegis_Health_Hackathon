"""
tools/clinical_synthesis.py — Cross-cutting clinical pattern synthesizer.

Detects multi-signal clinical patterns from the combined context of:
  - Flagged biomarker findings
  - Reported symptoms
  - Patient conditions (from patient_context_adapter)
  - Text-finding pattern IDs (from text_finding_analyzer)
  - Severity level

For each matched pattern, produces:
  - A patient-facing narrative for Section 6 (Personalized Recommendations)
  - Additive care plan lines for Section 7 (Care Plan)

Design principles:
  - Registry-driven: adding a new clinical pattern = one dict entry
  - Universal: no biomarker-specific hardcoded prose; uses runtime
    'name' fields from measurement items
  - Additive: existing per-biomarker KB advice is completely preserved.
    This module produces cross-cutting narratives that appear IN
    ADDITION to per-biomarker recommendations.
  - Fail-safe: any error returns empty synthesis so sections render
    exactly as they do without this module
  - Symmetric to text_finding_analyzer + dynamic_recommendations:
    same registry-driven pattern approach for consistency

Row 3 dashboard support:
  Each pattern carries a `narrative_short` (8–14 words) used by the
  dashboard Clinical Picture Summary card. The full-length narrative
  remains the source of truth for the report body — the short form
  is only used by the dashboard card.

Consumed by:
  tools.report_generator._build_deterministic_report() — hooks the
  synthesizer into Sections 6 and 7 rendering.
"""
from __future__ import annotations

from typing import Any

from loguru import logger
from tools.unit_normalizer import normalize_for_comparison, _resolve_canonical_key


# ═══════════════════════════════════════════════════════════════════
# CLINICAL PATTERN REGISTRY
# ═══════════════════════════════════════════════════════════════════
# Each pattern is a dict with:
#   id:                    unique identifier
#   triggers:              conditions that must be met (see matchers below)
#     symptoms:            list of substrings; ANY match satisfies
#     patient_conditions:  list of context condition IDs; ANY match satisfies
#     text_patterns:       list of pattern IDs; ANY match satisfies
#     biomarker:           single {key, op, value} condition
#     biomarker_combo:     list of {key, op, value}; combo_min_matches
#                          controls how many must be satisfied
#     combo_min_matches:   int; default 2 when biomarker_combo present
#   narrative:             patient-facing sentence for Section 6
#                          (None = pattern is silent for narratives)
#   narrative_short:       8–14 word summary for dashboard Clinical
#                          Picture card
#   care_plan_synthesis:   dict of bucket → list[str] for Section 7 merge
#   priority:              int; higher = more clinically important
#                          (used for narrative ordering when many match)

_CLINICAL_PATTERNS: list[dict[str, Any]] = [

    # ═════════════ VIRAL FEVER PATTERN ═════════════
    {
        "id": "viral_fever_with_thrombocytopenia",
        "triggers": {
            "symptoms": ["fever", "high temperature", "chills"],
            "biomarker": {"key": "platelets", "op": "<", "value": 150},
        },
        "narrative": (
            "Your reported fever combined with a reduced platelet count "
            "is a pattern often seen with viral fevers, including dengue "
            "in endemic regions. Prompt clinical evaluation and viral "
            "serology testing are recommended."
        ),
        "narrative_short": "Fever + low platelets — viral fever likely, consider dengue",
        "care_plan_synthesis": {
            "immediate": [
                "Discuss viral fever workup with your physician within "
                "24–48 hours — dengue NS1/IgM and repeat CBC may be advised."
            ],
            "short_term": [
                "Repeat CBC in 3–5 days to track platelet trend."
            ],
        },
        "priority": 90,
    },

    # ═════════════ IRON DEFICIENCY ANEMIA ═════════════
    {
        "id": "iron_deficiency_anemia",
        "triggers": {
            "biomarker_combo": [
                {"key": "haemoglobin", "op": "<", "value": 12.0},
                {"key": "ferritin", "op": "<", "value": 30},
                {"key": "mcv", "op": "<", "value": 80},
            ],
            "combo_min_matches": 2,
        },
        "narrative": (
            "The combination of reduced haemoglobin and low iron markers "
            "is consistent with iron-deficiency anemia. A structured iron "
            "replacement plan and evaluation for the underlying cause of "
            "iron loss (dietary, menstrual, gastrointestinal) is recommended."
        ),
        "narrative_short": "Low Hb + low iron markers — iron deficiency anemia",
        "care_plan_synthesis": {
            "short_term": [
                "Complete full iron studies (serum iron, ferritin, TIBC, "
                "transferrin saturation) if not already done."
            ],
            "lifestyle": [
                "Include iron-rich foods (leafy greens, lentils, red meat) "
                "paired with Vitamin C for improved absorption."
            ],
            "follow_up": [
                "Recheck haemoglobin and iron panel in 2–3 months after "
                "starting any dietary or supplement intervention."
            ],
        },
        "priority": 85,
    },

    # ═════════════ B12/FOLATE DEFICIENCY PATTERN ═════════════
    {
        "id": "b12_folate_deficiency",
        "triggers": {
            "biomarker_combo": [
                {"key": "vitamin_b12", "op": "<", "value": 200},
                {"key": "folate", "op": "<", "value": 3},
                {"key": "mcv", "op": ">", "value": 100},
            ],
            "combo_min_matches": 2,
        },
        "narrative": (
            "Multiple markers point toward Vitamin B12 or folate "
            "deficiency with macrocytic changes. Confirmatory testing "
            "(homocysteine, methylmalonic acid) and replacement therapy "
            "should be discussed with your physician."
        ),
        "narrative_short": "Low B12/folate + macrocytosis — deficiency anemia",
        "care_plan_synthesis": {
            "short_term": [
                "Check homocysteine and methylmalonic acid to confirm "
                "functional deficiency."
            ],
            "lifestyle": [
                "Include B12- and folate-rich foods: meat, fish, dairy, "
                "eggs, leafy greens, legumes, and fortified cereals."
            ],
            "follow_up": [
                "Recheck CBC and vitamin panel in 2–3 months after "
                "starting replacement therapy."
            ],
        },
        "priority": 85,
    },

    # ═════════════ METABOLIC SYNDROME PATTERN ═════════════
    {
        "id": "metabolic_syndrome_risk",
        "triggers": {
            "biomarker_combo": [
                {"key": "hba1c", "op": ">=", "value": 5.7},
                {"key": "glucose", "op": ">=", "value": 100},
                {"key": "triglycerides", "op": ">", "value": 150},
                {"key": "hdl_cholesterol", "op": "<", "value": 40},
                {"key": "ldl_cholesterol", "op": ">=", "value": 130},
            ],
            "combo_min_matches": 2,
        },
        "narrative": (
            "Multiple metabolic markers — spanning glucose regulation and "
            "lipid balance — indicate elevated metabolic syndrome risk. "
            "A structured cardio-metabolic review with your physician is "
            "recommended to prevent progression to diabetes or "
            "cardiovascular disease."
        ),
        "narrative_short": "Glucose + lipid abnormalities — metabolic syndrome risk",
        "care_plan_synthesis": {
            "short_term": [
                "Book a metabolic risk assessment: weight, waist "
                "circumference, blood pressure, and full lipid panel."
            ],
            "lifestyle": [
                "Adopt a Mediterranean-style diet, aim for 150+ minutes/"
                "week of aerobic exercise, and progressive weight "
                "management if clinically indicated."
            ],
            "follow_up": [
                "Recheck HbA1c, fasting glucose, and lipid panel in "
                "3 months to assess progress."
            ],
        },
        "priority": 80,
    },

    # ═════════════ SUBCLINICAL HYPOTHYROIDISM PATTERN ═════════════
    {
        "id": "subclinical_hypothyroid_pattern",
        "triggers": {
            "biomarker": {"key": "tsh", "op": ">", "value": 4.5},
        },
        "narrative": (
            "Elevated TSH suggests possible subclinical or overt "
            "hypothyroidism. A full thyroid workup and clinical "
            "evaluation are recommended to guide any need for therapy."
        ),
        "narrative_short": "Elevated TSH — possible hypothyroidism, workup advised",
        "care_plan_synthesis": {
            "short_term": [
                "Complete Free T3, Free T4, and anti-TPO antibody testing "
                "if not already done."
            ],
            "follow_up": [
                "Recheck TSH in 6–8 weeks alongside the full thyroid "
                "panel to confirm the trend."
            ],
        },
        "priority": 70,
    },

    # ═════════════ HYPOTHYROID + LOW T3/T4 CONFIRMATION ═════════════
    {
        "id": "hypothyroidism_confirmed_pattern",
        "triggers": {
            "biomarker_combo": [
                {"key": "tsh", "op": ">", "value": 4.5},
                {"key": "t3", "op": "<", "value": 80},
                {"key": "t4", "op": "<", "value": 4.5},
                {"key": "free_t3", "op": "<", "value": 2.0},
                {"key": "free_t4", "op": "<", "value": 0.8},
            ],
            "combo_min_matches": 2,
        },
        "narrative": (
            "Combined pattern of elevated TSH with reduced thyroid "
            "hormones is consistent with overt hypothyroidism. "
            "Endocrinology review for initiation or adjustment of "
            "thyroid replacement therapy is recommended."
        ),
        "narrative_short": "High TSH + low T3/T4 — overt hypothyroidism, endocrine review",
        "care_plan_synthesis": {
            "immediate": [
                "Book endocrinology consultation within 1–2 weeks to "
                "discuss thyroid hormone replacement."
            ],
            "long_term": [
                "Long-term thyroid monitoring will be required with "
                "periodic TSH/T4 rechecks per specialist guidance."
            ],
        },
        "priority": 88,
    },

    # ═════════════ UNCONTROLLED DIABETES ═════════════
    {
        "id": "uncontrolled_diabetes_pattern",
        "triggers": {
            "patient_conditions": ["diabetes"],
            "biomarker_combo": [
                {"key": "hba1c", "op": ">", "value": 8.0},
                {"key": "glucose", "op": ">", "value": 200},
            ],
            "combo_min_matches": 1,
        },
        "narrative": (
            "Your glycemic markers indicate suboptimal diabetes control. "
            "Review of your current diabetes regimen with your "
            "endocrinologist is recommended to reduce long-term "
            "complication risk."
        ),
        "narrative_short": "High HbA1c/glucose — diabetes control suboptimal",
        "care_plan_synthesis": {
            "immediate": [
                "Book endocrinology review within 1–2 weeks to discuss "
                "regimen adjustment."
            ],
            "short_term": [
                "Complete comprehensive diabetes workup: kidney function, "
                "urine albumin, lipid panel, and eye examination if not "
                "current."
            ],
            "long_term": [
                "Establish 3-month HbA1c monitoring cycle to track "
                "glycemic improvement."
            ],
        },
        "priority": 90,
    },

    # ═════════════ CARDIOVASCULAR RISK CLUSTER ═════════════
    {
        "id": "cardiovascular_risk_cluster",
        "triggers": {
            "biomarker_combo": [
                {"key": "ldl_cholesterol", "op": ">=", "value": 130},
                {"key": "total_cholesterol", "op": ">=", "value": 200},
                {"key": "triglycerides", "op": ">", "value": 150},
                {"key": "hdl_cholesterol", "op": "<", "value": 40},
                {"key": "bp_systolic", "op": ">=", "value": 130},
            ],
            "combo_min_matches": 2,
        },
        "narrative": (
            "Multiple cardiovascular risk factors are present in your "
            "results. A cardiovascular risk assessment with your "
            "physician is recommended to guide preventive strategy."
        ),
        "narrative_short": "Multiple CV risk factors — cardiovascular assessment needed",
        "care_plan_synthesis": {
            "short_term": [
                "Discuss a full 10-year cardiovascular risk assessment "
                "with your physician."
            ],
            "lifestyle": [
                "Adopt a heart-healthy diet, regular aerobic exercise, "
                "smoking cessation if applicable, and weight management."
            ],
            "follow_up": [
                "Recheck lipid panel and blood pressure in 3 months "
                "after intervention."
            ],
        },
        "priority": 82,
    },

    # ═════════════ RENAL FUNCTION DECLINE ═════════════
    {
        "id": "renal_function_decline",
        "triggers": {
            "biomarker_combo": [
                {"key": "creatinine", "op": ">", "value": 1.3},
                {"key": "urea", "op": ">", "value": 45},
                {"key": "bun", "op": ">", "value": 20},
            ],
            "combo_min_matches": 2,
        },
        "narrative": (
            "Multiple kidney function markers are outside the reference "
            "range, suggesting reduced renal function. Nephrology "
            "evaluation and calculation of eGFR are recommended."
        ),
        "narrative_short": "Elevated kidney markers — reduced renal function likely",
        "care_plan_synthesis": {
            "immediate": [
                "Consult a nephrologist within 1–2 weeks to evaluate "
                "kidney function and calculate eGFR."
            ],
            "lifestyle": [
                "Avoid NSAIDs and other nephrotoxic medications. "
                "Follow your physician's guidance on fluid and "
                "protein intake."
            ],
        },
        "priority": 85,
    },

    # ═════════════ LIVER DYSFUNCTION PATTERN ═════════════
    {
        "id": "liver_dysfunction_pattern",
        "triggers": {
            "biomarker_combo": [
                {"key": "sgpt_alt", "op": ">", "value": 56},
                {"key": "sgot_ast", "op": ">", "value": 48},
                {"key": "alp", "op": ">", "value": 130},
                {"key": "ggt", "op": ">", "value": 60},
                {"key": "bilirubin", "op": ">", "value": 1.2},
            ],
            "combo_min_matches": 2,
        },
        "narrative": (
            "Multiple liver enzymes are elevated, suggesting hepatic "
            "irritation or dysfunction. Avoiding alcohol and reviewing "
            "all medications and supplements with your physician are "
            "important next steps."
        ),
        "narrative_short": "Elevated liver enzymes — hepatic irritation or dysfunction",
        "care_plan_synthesis": {
            "immediate": [
                "Avoid alcohol and review all medications and supplements "
                "with your physician."
            ],
            "short_term": [
                "Complete viral hepatitis screening (HBV, HCV) and "
                "abdominal ultrasound if not already done."
            ],
            "follow_up": [
                "Recheck LFT panel in 4–8 weeks to confirm trend."
            ],
        },
        "priority": 82,
    },

    # ═════════════ PROTEIN / NUTRITIONAL DEPLETION ═════════════
    {
        "id": "protein_nutritional_depletion",
        "triggers": {
            "biomarker_combo": [
                {"key": "total_protein", "op": "<", "value": 6.0},
                {"key": "albumin", "op": "<", "value": 3.5},
                {"key": "globulin", "op": "<", "value": 2.0},
            ],
            "combo_min_matches": 2,
        },
        "narrative": (
            "Reduced protein markers may reflect nutritional depletion, "
            "chronic illness, or liver/kidney involvement. A structured "
            "nutritional review with your physician is recommended."
        ),
        "narrative_short": "Low protein markers — nutritional or organ cause",
        "care_plan_synthesis": {
            "short_term": [
                "Discuss with your physician for nutritional and "
                "hepato-renal evaluation."
            ],
            "lifestyle": [
                "Ensure adequate protein intake through lean meat, dairy, "
                "legumes, eggs, and fish per dietary preference."
            ],
            "follow_up": [
                "Recheck total protein, albumin, and globulin in "
                "2–3 months."
            ],
        },
        "priority": 70,
    },

    # ═════════════ INFLAMMATORY PATTERN ═════════════
    {
        "id": "systemic_inflammation_pattern",
        "triggers": {
            "biomarker_combo": [
                {"key": "crp", "op": ">", "value": 10},
                {"key": "esr", "op": ">", "value": 30},
            ],
            "combo_min_matches": 2,
        },
        "narrative": (
            "Multiple inflammatory markers are elevated, suggesting "
            "an active inflammatory or infectious process. Clinical "
            "correlation with symptoms and further workup are recommended."
        ),
        "narrative_short": "High CRP + ESR — active inflammation or infection",
        "care_plan_synthesis": {
            "short_term": [
                "Consult your physician to identify the underlying "
                "cause of inflammation."
            ],
        },
        "priority": 75,
    },

    # ═════════════ HEMOLYSIS PATTERN (correlates with smear) ═════════════
    {
        "id": "hemolysis_pattern",
        "triggers": {
            "text_patterns": ["smear_schistocytes", "smear_spherocytes"],
            "biomarker_combo": [
                {"key": "haemoglobin", "op": "<", "value": 11},
                {"key": "bilirubin", "op": ">", "value": 1.5},
                {"key": "ldh", "op": ">", "value": 250},
            ],
            "combo_min_matches": 1,
        },
        "narrative": (
            "The combination of morphological changes on the peripheral "
            "smear with laboratory markers suggests possible hemolysis. "
            "Urgent hematology evaluation is recommended."
        ),
        "narrative_short": "Smear changes + lab markers — possible hemolysis, urgent review",
        "care_plan_synthesis": {
            "immediate": [
                "Book urgent hematology consultation within 1 week."
            ],
            "short_term": [
                "Complete LDH, haptoglobin, reticulocyte count, and "
                "direct Coombs test."
            ],
        },
        "priority": 92,
    },

    # ═════════════ VIRAL PATTERN CORRELATION ═════════════
    {
        "id": "viral_infection_correlated_pattern",
        "triggers": {
            "text_patterns": ["smear_reactive_lymphocyte"],
            "biomarker_combo": [
                {"key": "lymphocytes", "op": ">", "value": 40},
                {"key": "neutrophils", "op": "<", "value": 45},
            ],
            "combo_min_matches": 1,
        },
        "narrative": (
            "The combination of reactive lymphocytes on the peripheral "
            "smear with shifted white cell differential strongly "
            "supports an active viral infection. Clinical correlation "
            "with symptoms is recommended, and viral serology may be "
            "considered."
        ),
        "narrative_short": "Reactive lymphocytes + WBC shift — active viral infection",
        "care_plan_synthesis": {
            "short_term": [
                "If febrile or symptomatic, discuss viral serology "
                "(EBV, CMV, dengue, others) with your physician."
            ],
            "follow_up": [
                "Repeat CBC in 2–4 weeks to confirm resolution."
            ],
        },
        "priority": 80,
    },
]


# ═══════════════════════════════════════════════════════════════════
# TRIGGER MATCHERS
# ═══════════════════════════════════════════════════════════════════

def _symptoms_match(triggers: dict, symptoms: list[str]) -> bool:
    """Return True if any symptom keyword substring is present."""
    keywords = triggers.get("symptoms")
    if not keywords:
        return True  # No requirement
    if not symptoms:
        return False
    combined = " ".join(str(s).lower() for s in symptoms if s)
    return any(kw.lower() in combined for kw in keywords)


def _conditions_match(triggers: dict, patient_context: dict) -> bool:
    """Return True if any required condition ID is in patient context."""
    required = triggers.get("patient_conditions")
    if not required:
        return True
    ctx_conditions = set(patient_context.get("conditions") or [])
    return bool(set(required) & ctx_conditions)


def _text_patterns_match(triggers: dict, text_pattern_ids: list[str]) -> bool:
    """Return True if any required text pattern ID matched."""
    required = triggers.get("text_patterns")
    if not required:
        return True
    matched = set(text_pattern_ids or [])
    return bool(set(required) & matched)


def _single_biomarker_match(triggers: dict, flagged_items: list[dict]) -> bool:
    """Check the single 'biomarker' condition if specified."""
    cond = triggers.get("biomarker")
    if not cond:
        return True
    return _one_biomarker_condition_met(cond, flagged_items)


def _one_biomarker_condition_met(
    condition: dict,
    measurements: list[dict],
) -> bool:
    """
    Check if any measurement satisfies a single biomarker threshold.

    Matching strategy (three layers, all fail-safe):
      1. Exact key match:      m["key"] == key_wanted
      2. Alias resolution:     _resolve_canonical_key(m["key"]) ==
                               _resolve_canonical_key(key_wanted)
      3. Name-based fallback:  key_wanted appears as a word-boundary
                               substring of m["name"].lower() — handles
                               the case where the dashboard extractor
                               stored a long canonical slug (e.g.
                               "thyroid_stimulating_hormone") that
                               doesn't alias-resolve back to the short
                               registry key ("tsh") used in pattern
                               triggers.

    Routes numeric comparisons through normalize_for_comparison() so
    clinical shorthand thresholds are compared against raw parser values
    in the correct canonical unit.

    Fail-safe: errors per measurement are skipped; returns False if
    no measurement satisfies the condition.
    """
    import re as _re

    key_wanted = str(condition.get("key") or "").lower()
    op         = condition.get("op")
    threshold  = condition.get("value")

    if not key_wanted or not op or threshold is None:
        return False

    try:
        threshold_f = float(threshold)
    except (TypeError, ValueError):
        return False

    # Pre-compute resolved form once
    wanted_resolved = _resolve_canonical_key(key_wanted)

    # Name-based pattern: key_wanted as a word-boundary token in name
    # e.g. "tsh" matches "TSH", "Thyroid Stimulating Hormone" won't
    # false-positive on "potassium" because \b protects boundaries.
    try:
        _name_pattern = _re.compile(
            r"\b" + _re.escape(key_wanted.replace("_", " ")) + r"\b",
            _re.IGNORECASE,
        )
    except _re.error:
        _name_pattern = None

    for m in measurements:
        m_key  = str(m.get("key")  or "").lower()
        m_name = str(m.get("name") or "").lower()

        # Layer 1: exact key match
        matched = (m_key == key_wanted)

        # Layer 2: alias resolution
        if not matched:
            matched = (
                _resolve_canonical_key(m_key) == wanted_resolved
            )

        # Layer 3: name-based fallback
        if not matched and _name_pattern:
            matched = bool(_name_pattern.search(m_name))

        # Also try the key_wanted directly as a substring of the name
        # for short abbreviations (e.g. "tsh" in "tsh 6.05")
        if not matched:
            matched = (key_wanted in m_name)

        if not matched:
            continue

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


def _biomarker_combo_match(triggers: dict, flagged_items: list[dict]) -> bool:
    """Return True if at least combo_min_matches conditions are satisfied."""
    combo = triggers.get("biomarker_combo")
    if not combo:
        return True

    min_matches = int(triggers.get("combo_min_matches") or 2)
    matched = sum(
        1 for cond in combo
        if _one_biomarker_condition_met(cond, flagged_items)
    )
    return matched >= min_matches


def _pattern_matches(
    pattern: dict,
    symptoms: list[str],
    flagged_items: list[dict],
    patient_context: dict,
    text_pattern_ids: list[str],
) -> bool:
    """A pattern matches when ALL of its specified triggers are satisfied."""
    triggers = pattern.get("triggers") or {}
    checks: list[bool] = []

    if "symptoms" in triggers:
        checks.append(_symptoms_match(triggers, symptoms))
    if "patient_conditions" in triggers:
        checks.append(_conditions_match(triggers, patient_context))
    if "text_patterns" in triggers:
        checks.append(_text_patterns_match(triggers, text_pattern_ids))
    if "biomarker" in triggers:
        checks.append(_single_biomarker_match(triggers, flagged_items))
    if "biomarker_combo" in triggers:
        checks.append(_biomarker_combo_match(triggers, flagged_items))

    return len(checks) > 0 and all(checks)


# ═══════════════════════════════════════════════════════════════════
# EMPTY RESULT SHAPE
# ═══════════════════════════════════════════════════════════════════
def _empty_result() -> dict:
    return {
        "recommendation_narratives": [],
        "care_plan_narratives": {
            "immediate": [],
            "short_term": [],
            "lifestyle": [],
            "follow_up": [],
            "long_term": [],
        },
        "matched_pattern_ids": [],
    }


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════

def synthesize_personalized_insights(
    flagged_items: list[dict],
    symptoms: list[str],
    patient_context: dict,
    text_pattern_ids: list[str],
    severity_level: str,
) -> dict:
    """
    Detect cross-cutting clinical patterns and produce synthesized
    content for Sections 6 (Personalized Recommendations) and 7
    (Care Plan) of the report.

    Universal design: registry-driven pattern matching with no
    biomarker-specific hardcoded prose. Adding a new pattern means
    adding one entry to _CLINICAL_PATTERNS.

    Args:
        flagged_items:      All flagged measurements (any status !=normal)
        symptoms:           Reported symptoms (strings)
        patient_context:    From tools.patient_context_adapter.normalize_*
        text_pattern_ids:   From tools.text_finding_analyzer matches
        severity_level:     Current severity level (unused now, reserved
                            for future severity-gated patterns)

    Returns:
        {
          "recommendation_narratives": list[str],       # Section 6
          "care_plan_narratives": {                     # Section 7
              "immediate": list[str],
              "short_term": list[str],
              "lifestyle": list[str],
              "follow_up": list[str],
              "long_term": list[str],
          },
          "matched_pattern_ids": list[str],             # for logging/tests
        }

    Fail-safe: any exception returns empty result so Sections 6 and 7
    render exactly as they do without this module.
    """
    try:
        flagged_items = flagged_items or []
        symptoms = symptoms or []
        patient_context = patient_context or {}
        text_pattern_ids = text_pattern_ids or []

        result = _empty_result()
        matched_ids: set[str] = set()

        # Score and select matched patterns
        matched_patterns: list[dict] = []
        for pattern in _CLINICAL_PATTERNS:
            pid = pattern.get("id") or ""
            if not pid or pid in matched_ids:
                continue
            try:
                if _pattern_matches(
                    pattern, symptoms, flagged_items,
                    patient_context, text_pattern_ids,
                ):
                    matched_ids.add(pid)
                    matched_patterns.append(pattern)
            except Exception:
                # Skip failed pattern, continue evaluating others
                continue

        # Sort matched patterns by priority (highest first) for output ordering
        matched_patterns.sort(
            key=lambda p: int(p.get("priority") or 0),
            reverse=True,
        )

        # Collect narratives and care plan additions
        for pattern in matched_patterns:
            pid = pattern.get("id") or ""
            result["matched_pattern_ids"].append(pid)

            narrative = pattern.get("narrative")
            if narrative:
                narrative = str(narrative).strip()
                if narrative and narrative not in result["recommendation_narratives"]:
                    result["recommendation_narratives"].append(narrative)

            care_plan_synth = pattern.get("care_plan_synthesis") or {}
            for bucket in (
                "immediate", "short_term", "lifestyle",
                "follow_up", "long_term",
            ):
                entries = care_plan_synth.get(bucket) or []
                for entry in entries:
                    entry = str(entry).strip()
                    if entry and entry not in result["care_plan_narratives"][bucket]:
                        result["care_plan_narratives"][bucket].append(entry)

        if matched_ids:
            logger.info(
                "clinical_synthesis · patterns matched",
                count=len(matched_ids),
                ids=sorted(matched_ids),
            )

        return result

    except Exception:
        logger.exception(
            "clinical_synthesis · synthesis failed; returning empty result"
        )
        return _empty_result()


__all__ = ["synthesize_personalized_insights"]