"""
tools/text_finding_analyzer.py — Text finding pattern analyzer.

Scans free-text clinical findings from lab reports (peripheral smear
morphology, impressions, parasite screening, cytology comments, etc.)
and produces structured signals that feed into:

  - Critical Observations & Flags   (observations list)
  - Personalized Recommendations    (via correlated_findings)
  - Care Plan buckets               (care_plan_additions)
  - Severity scoring                (severity_boost — consumed by FIX #10)

Design principles:
  - Universal: driven by _TEXT_PATTERNS registry; adding a new
    pattern is one dict entry, no code change elsewhere.
  - Fail-safe: any exception in analysis returns an empty result
    shape so report generation is NEVER blocked.
  - Additive: text findings still render verbatim in the
    "Peripheral Smear & Morphology" subsection — this analyzer
    adds interpretation on top of that display.
  - Correlation-aware: patterns can optionally merge with numeric
    biomarker findings to produce stronger, unified observations.
  - Case-insensitive substring matching for keywords, with word-
    boundary anchoring where medically necessary to avoid collisions.

Consumed by:
  tools.report_generator._build_deterministic_report() after
  flagged_items are built and before Critical Observations HTML
  is rendered.

Row 3 dashboard support:
  Each pattern with an observation now also carries a `narrative_short`
  (8–14 words) used by the dashboard Clinical Picture Summary card.
  Correlation blocks with a `merged_observation` also carry
  `merged_narrative_short`. The full-length narratives remain the
  source of truth for the report body — the short forms are only
  used by the dashboard card.
"""
from __future__ import annotations

import re
from typing import Any

from loguru import logger


# ═══════════════════════════════════════════════════════════════════
# Empty result shape — returned on any error or when no findings match.
# ═══════════════════════════════════════════════════════════════════
def _empty_result() -> dict:
    return {
        "observations": [],
        "care_plan_additions": {
            "immediate": [],
            "short_term": [],
            "lifestyle": [],
            "follow_up": [],
            "long_term": [],
        },
        "severity_boost": 0,
        "correlated_findings": [],
        "matched_pattern_ids": [],
    }


# ═══════════════════════════════════════════════════════════════════
# PATTERN REGISTRY — single source of truth for clinical text patterns.
# ═══════════════════════════════════════════════════════════════════
# Each pattern is a dict with these fields:
#   id:              unique string identifier (used by severity_scorer)
#   keywords:        list of substrings to match (case-insensitive)
#   category:        rbc_morphology | wbc_morphology | platelet_morphology
#                    | parasites | pathology_impression | inflammation
#   severity_class:  reassuring | informational | concerning | urgent | critical
#   severity_boost:  0 | 1 | 2  (added to overall severity via FIX #10)
#   observation:     patient-facing sentence for Critical Observations
#                    (None = pattern is reassuring, no observation added)
#   narrative_short: 8–14 word summary for dashboard Clinical Picture card
#                    (only present when observation is not None)
#   care_plan:       dict with 5 buckets; each value is str or None
#   correlation:     optional cross-biomarker rule:
#     {
#       "biomarker_condition": {
#         "key": "lymphocytes",           # canonical key
#         "status": ["high", "borderline"], # any of these statuses
#         "value_op": ">",                  # optional: "<", ">", "<=", ">="
#         "value_threshold": 40,            # optional: numeric threshold
#       },
#       "merged_observation": "combined text...",
#       "merged_narrative_short": "short combined text",
#       "boost_severity_by": 1,
#     }

_TEXT_PATTERNS: list[dict[str, Any]] = [

    # ═══════════════════ RBC MORPHOLOGY ═══════════════════
    {
        "id": "smear_normocytic_normochromic",
        "keywords": ["normocytic normochromic", "normocytic and normochromic"],
        "category": "rbc_morphology",
        "severity_class": "reassuring",
        "severity_boost": 0,
        "observation": None,
        "care_plan": {},
    },
    {
        "id": "smear_microcytic_hypochromic",
        "keywords": ["microcytic hypochromic", "microcytic and hypochromic",
                     "microcytic, hypochromic"],
        "category": "rbc_morphology",
        "severity_class": "concerning",
        "severity_boost": 1,
        "observation": "Peripheral smear shows microcytic hypochromic red cells, a pattern classically associated with iron deficiency.",
        "narrative_short": "Microcytic hypochromic RBCs — iron deficiency likely",
        "care_plan": {
            "short_term": "Complete iron studies (serum iron, ferritin, TIBC, transferrin saturation) to confirm iron deficiency.",
            "lifestyle": "Include iron-rich foods (leafy greens, lentils, red meat) with Vitamin C for better absorption.",
            "follow_up": "Recheck CBC and iron panel in 2–3 months after any dietary or supplement changes.",
        },
        "correlation": {
            "biomarker_condition": {
                "key": "haemoglobin",
                "status": ["low", "critical_low"],
            },
            "merged_observation": "Microcytic hypochromic red cells on peripheral smear combined with low haemoglobin strongly supports iron-deficiency anemia. Iron studies and clinical review are recommended.",
            "merged_narrative_short": "Low Hb + microcytic RBCs — iron deficiency anemia",
            "boost_severity_by": 1,
        },
    },
    {
        "id": "smear_macrocytic",
        "keywords": ["macrocytic", "megaloblastic"],
        "category": "rbc_morphology",
        "severity_class": "concerning",
        "severity_boost": 1,
        "observation": "Peripheral smear shows macrocytic red cells, which may indicate Vitamin B12 or folate deficiency.",
        "narrative_short": "Macrocytic RBCs — possible B12/folate deficiency",
        "care_plan": {
            "short_term": "Check serum Vitamin B12, folate, and homocysteine to evaluate for deficiency.",
            "lifestyle": "Include B12- and folate-rich foods (dairy, eggs, leafy greens, legumes).",
            "follow_up": "Recheck CBC and vitamin panel in 2–3 months.",
        },
        "correlation": {
            "biomarker_condition": {
                "key": "vitamin_b12",
                "status": ["low", "critical_low"],
            },
            "merged_observation": "Macrocytic red cells on peripheral smear combined with low Vitamin B12 confirms megaloblastic changes secondary to B12 deficiency. B12 replacement therapy should be discussed with your physician.",
            "merged_narrative_short": "Macrocytic RBCs + low B12 — megaloblastic anemia",
            "boost_severity_by": 1,
        },
    },
    {
        "id": "smear_sickle_cells",
        "keywords": ["sickle cells", "sickle cell", "sickled cells"],
        "category": "rbc_morphology",
        "severity_class": "urgent",
        "severity_boost": 2,
        "observation": "Sickle cells identified on peripheral smear — urgent hematology evaluation is required to confirm sickle cell disease or trait.",
        "narrative_short": "Sickle cells seen — urgent hematology needed",
        "care_plan": {
            "immediate": "Book urgent hematology consultation within 1 week.",
            "short_term": "Complete haemoglobin electrophoresis and sickle cell screening.",
            "follow_up": "Long-term hematology follow-up is required.",
        },
    },
    {
        "id": "smear_target_cells",
        "keywords": ["target cells", "codocytes"],
        "category": "rbc_morphology",
        "severity_class": "concerning",
        "severity_boost": 1,
        "observation": "Target cells noted on peripheral smear — may suggest thalassemia, chronic liver disease, or post-splenectomy state; clinical correlation is recommended.",
        "narrative_short": "Target cells — check thalassemia or liver disease",
        "care_plan": {
            "short_term": "Complete haemoglobin electrophoresis and liver function testing.",
            "follow_up": "Hematology consultation may be advised.",
        },
    },
    {
        "id": "smear_schistocytes",
        "keywords": ["schistocytes", "fragmented cells", "fragmented red cells",
                     "helmet cells"],
        "category": "rbc_morphology",
        "severity_class": "urgent",
        "severity_boost": 2,
        "observation": "Schistocytes (fragmented red cells) noted on peripheral smear — this can indicate microangiopathic hemolysis and requires urgent hematology evaluation.",
        "narrative_short": "Schistocytes — possible microangiopathic hemolysis, urgent review",
        "care_plan": {
            "immediate": "Seek hematology evaluation urgently.",
            "short_term": "Complete LDH, haptoglobin, bilirubin, reticulocyte count, and peripheral smear review.",
            "follow_up": "Close hematology follow-up is required.",
        },
    },
    {
        "id": "smear_spherocytes",
        "keywords": ["spherocytes", "spherocytosis"],
        "category": "rbc_morphology",
        "severity_class": "concerning",
        "severity_boost": 1,
        "observation": "Spherocytes noted on peripheral smear — may suggest autoimmune hemolytic anemia or hereditary spherocytosis; hematology review is recommended.",
        "narrative_short": "Spherocytes — possible autoimmune hemolysis or hereditary cause",
        "care_plan": {
            "short_term": "Complete direct Coombs test, reticulocyte count, LDH, and bilirubin.",
            "follow_up": "Hematology consultation recommended.",
        },
    },
    {
        "id": "smear_rouleaux",
        "keywords": ["rouleaux formation", "rouleaux"],
        "category": "rbc_morphology",
        "severity_class": "informational",
        "severity_boost": 0,
        "observation": "Rouleaux formation noted on peripheral smear — often associated with elevated inflammatory markers or plasma protein abnormalities; clinical correlation is recommended.",
        "narrative_short": "Rouleaux formation — check inflammation or plasma proteins",
        "care_plan": {
            "short_term": "Consider ESR, CRP, and serum protein electrophoresis if not already done.",
        },
    },
    {
        "id": "smear_tear_drop_cells",
        "keywords": ["tear drop cells", "teardrop cells", "dacrocytes"],
        "category": "rbc_morphology",
        "severity_class": "concerning",
        "severity_boost": 1,
        "observation": "Tear-drop cells noted on peripheral smear — may indicate bone marrow infiltration or myelofibrosis; hematology evaluation is recommended.",
        "narrative_short": "Tear-drop cells — possible marrow infiltration",
        "care_plan": {
            "short_term": "Hematology consultation for further workup.",
        },
    },
    {
        "id": "smear_rbc_pool_decrease",
        "keywords": ["decrease in rbc pool", "decreased rbc pool",
                     "reduced rbc pool"],
        "category": "rbc_morphology",
        "severity_class": "informational",
        "severity_boost": 0,
        "observation": "Reduced red cell pool noted on peripheral smear — please correlate with the RBC count and haemoglobin level.",
        "narrative_short": "Reduced RBC pool — correlate with count and Hb",
        "care_plan": {},
    },

    # ═══════════════════ WBC MORPHOLOGY ═══════════════════
    {
        "id": "smear_reactive_lymphocyte",
        "keywords": ["reactive lymphocyte", "reactive lymphocytes",
                     "atypical lymphocyte", "atypical lymphocytes"],
        "category": "wbc_morphology",
        "severity_class": "informational",
        "severity_boost": 0,
        "observation": "Reactive (atypical) lymphocytes noted on peripheral smear — this pattern is commonly seen with viral infections; clinical correlation with symptoms is recommended.",
        "narrative_short": "Reactive lymphocytes — commonly viral infection",
        "care_plan": {
            "short_term": "If febrile or unwell, discuss viral serology (EBV, CMV, dengue, others) with your physician.",
            "follow_up": "Repeat CBC in 2–4 weeks to confirm resolution.",
        },
        "correlation": {
            "biomarker_condition": {
                "key": "lymphocytes",
                "status": ["high", "borderline"],
            },
            "merged_observation": "Reactive lymphocytes on peripheral smear combined with elevated lymphocyte count strongly suggests an active viral infection. Clinical correlation with fever and other symptoms is recommended; viral serology may be considered.",
            "merged_narrative_short": "Reactive lymphocytes + high lymph count — active viral infection",
            "boost_severity_by": 1,
        },
    },
    {
        "id": "smear_blast_cells",
        "keywords": ["blast cells", "blasts seen", "immature cells",
                     "myeloblasts", "lymphoblasts"],
        "category": "wbc_morphology",
        "severity_class": "critical",
        "severity_boost": 2,
        "observation": "Blast cells (immature white cells) identified on peripheral smear — this is a critical finding requiring urgent hematology evaluation to rule out acute leukemia.",
        "narrative_short": "Blast cells seen — urgent leukemia workup required",
        "care_plan": {
            "immediate": "Seek urgent hematology consultation within 24–48 hours.",
            "short_term": "Bone marrow examination and flow cytometry will likely be required.",
            "follow_up": "Long-term hematology/oncology care is required.",
        },
    },
    {
        "id": "smear_toxic_granulation",
        "keywords": ["toxic granulation", "toxic granules"],
        "category": "wbc_morphology",
        "severity_class": "concerning",
        "severity_boost": 1,
        "observation": "Toxic granulation noted in neutrophils on peripheral smear — often associated with severe bacterial infection or sepsis; urgent clinical correlation is recommended.",
        "narrative_short": "Toxic granulation — possible severe bacterial infection",
        "care_plan": {
            "immediate": "If febrile or unwell, seek medical evaluation promptly.",
            "short_term": "Complete blood cultures, CRP, and infection workup.",
        },
    },
    {
        "id": "smear_hypersegmented_neutrophils",
        "keywords": ["hypersegmented neutrophils", "hypersegmentation"],
        "category": "wbc_morphology",
        "severity_class": "concerning",
        "severity_boost": 1,
        "observation": "Hypersegmented neutrophils noted on peripheral smear — a classic marker of Vitamin B12 or folate deficiency; deficiency workup is recommended.",
        "narrative_short": "Hypersegmented neutrophils — B12/folate deficiency likely",
        "care_plan": {
            "short_term": "Check serum Vitamin B12, folate, and homocysteine.",
            "follow_up": "Recheck CBC and vitamin panel in 2–3 months.",
        },
        "correlation": {
            "biomarker_condition": {
                "key": "vitamin_b12",
                "status": ["low", "critical_low"],
            },
            "merged_observation": "Hypersegmented neutrophils on peripheral smear combined with low Vitamin B12 confirms megaloblastic changes secondary to B12 deficiency. B12 replacement therapy should be discussed with your physician.",
            "merged_narrative_short": "Hypersegmented neutrophils + low B12 — megaloblastic anemia",
            "boost_severity_by": 1,
        },
    },
    {
        "id": "smear_left_shift",
        "keywords": ["left shift", "shift to the left"],
        "category": "wbc_morphology",
        "severity_class": "informational",
        "severity_boost": 0,
        "observation": "Left shift noted on peripheral smear — often reflects active infection or inflammation; clinical correlation is recommended.",
        "narrative_short": "Left shift — active infection or inflammation likely",
        "care_plan": {
            "short_term": "If symptomatic, discuss with your physician for evaluation of possible infection.",
        },
    },
    {
        "id": "smear_auer_rods",
        "keywords": ["auer rods", "auer rod"],
        "category": "wbc_morphology",
        "severity_class": "critical",
        "severity_boost": 2,
        "observation": "Auer rods identified on peripheral smear — a hallmark of acute myeloid leukemia. Emergency hematology consultation is required.",
        "narrative_short": "Auer rods — acute myeloid leukemia, emergency review",
        "care_plan": {
            "immediate": "Seek emergency hematology consultation immediately.",
            "short_term": "Bone marrow examination and molecular studies will be required urgently.",
            "follow_up": "Long-term hematology/oncology care is required.",
        },
    },

    # ═══════════════════ PLATELET MORPHOLOGY ═══════════════════
    {
        "id": "smear_platelets_adequate",
        "keywords": ["platelets adequate", "platelets are adequate",
                     "platelets normal", "platelets appear normal",
                     "platelets appear adequate"],
        "category": "platelet_morphology",
        "severity_class": "reassuring",
        "severity_boost": 0,
        "observation": None,
        "care_plan": {},
    },
    {
        "id": "smear_platelets_low",
        "keywords": ["platelets low", "platelets mild low", "platelets decreased",
                     "platelets reduced", "platelets mildly reduced",
                     "platelets appear low", "platelets appear decreased"],
        "category": "platelet_morphology",
        "severity_class": "concerning",
        "severity_boost": 1,
        "observation": "Platelets appear reduced on peripheral smear — please correlate with the platelet count and clinical picture.",
        "narrative_short": "Platelets low on smear — confirm and evaluate cause",
        "care_plan": {
            "short_term": "Repeat CBC and consider peripheral smear review by a hematologist if persistently low.",
            "follow_up": "Recheck platelets as advised by your physician.",
        },
        "correlation": {
            "biomarker_condition": {
                "key": "platelets",
                "status": ["low", "critical_low", "borderline"],
            },
            "merged_observation": "Reduced platelets on peripheral smear are consistent with the low platelet count in the CBC — thrombocytopenia is confirmed. Avoid trauma and NSAIDs; clinical evaluation for the underlying cause is recommended.",
            "merged_narrative_short": "Low platelets on smear + count — thrombocytopenia confirmed",
            "boost_severity_by": 1,
        },
    },
    {
        "id": "smear_platelets_increased",
        "keywords": ["platelets increased", "platelets elevated",
                     "platelets appear increased", "platelet clumping",
                     "platelets clumping"],
        "category": "platelet_morphology",
        "severity_class": "informational",
        "severity_boost": 0,
        "observation": "Platelets appear increased or show clumping on peripheral smear — repeat CBC (ideally in a citrate tube if clumping suspected) is advised.",
        "narrative_short": "Platelets high or clumped — repeat CBC advised",
        "care_plan": {
            "short_term": "Repeat CBC in 2–4 weeks to confirm persistent thrombocytosis.",
        },
    },
    {
        "id": "smear_giant_platelets",
        "keywords": ["giant platelets", "large platelets"],
        "category": "platelet_morphology",
        "severity_class": "concerning",
        "severity_boost": 1,
        "observation": "Giant platelets noted on peripheral smear — may indicate accelerated platelet production or an inherited platelet disorder; hematology review is recommended.",
        "narrative_short": "Giant platelets — possible marrow or inherited disorder",
        "care_plan": {
            "short_term": "Hematology consultation for further evaluation.",
        },
    },

    # ═══════════════════ PARASITES / INFECTIONS ═══════════════════
    {
        "id": "smear_malaria_parasite",
        "keywords": ["malaria parasite", "malarial parasite", "mp seen",
                     "p. falciparum", "p. vivax", "plasmodium",
                     "malaria parasite seen"],
        "category": "parasites",
        "severity_class": "urgent",
        "severity_boost": 2,
        "observation": "Malaria parasites identified on peripheral smear — urgent medical treatment is required.",
        "narrative_short": "Malaria parasites seen — urgent treatment required",
        "care_plan": {
            "immediate": "Seek urgent medical care for antimalarial therapy.",
            "short_term": "Complete parasite species identification and parasitemia quantification.",
            "follow_up": "Repeat smear post-treatment to confirm clearance.",
        },
    },
    {
        "id": "smear_microfilaria",
        "keywords": ["microfilaria", "microfilariae"],
        "category": "parasites",
        "severity_class": "urgent",
        "severity_boost": 2,
        "observation": "Microfilaria identified on peripheral smear — filariasis workup and treatment are required.",
        "narrative_short": "Microfilaria seen — filariasis workup needed",
        "care_plan": {
            "immediate": "Seek medical care for filariasis evaluation and treatment.",
            "short_term": "Complete species identification and infectious disease consultation.",
        },
    },

    # ═══════════════════ PATHOLOGY IMPRESSIONS ═══════════════════
    {
        "id": "impression_no_malignancy",
        "keywords": ["no malignant cells", "no evidence of malignancy",
                     "no atypia", "no malignancy seen", "negative for malignancy"],
        "category": "pathology_impression",
        "severity_class": "reassuring",
        "severity_boost": 0,
        "observation": None,
        "care_plan": {},
    },
    {
        "id": "impression_suggestive_of_malignancy",
        "keywords": ["suggestive of malignancy", "atypical cells suspicious",
                     "malignant cells seen", "suspicious for malignancy"],
        "category": "pathology_impression",
        "severity_class": "urgent",
        "severity_boost": 2,
        "observation": "Findings are suggestive of possible malignancy — urgent oncology evaluation and confirmatory testing are required.",
        "narrative_short": "Findings suspicious for malignancy — urgent oncology needed",
        "care_plan": {
            "immediate": "Seek urgent oncology consultation within 1 week.",
            "short_term": "Confirmatory biopsy and staging workup will likely be required.",
            "follow_up": "Long-term oncology care is required.",
        },
    },
    {
        "id": "impression_chronic_inflammation",
        "keywords": ["chronic inflammation", "chronic inflammatory changes"],
        "category": "inflammation",
        "severity_class": "informational",
        "severity_boost": 0,
        "observation": "Chronic inflammation noted — please correlate with inflammatory markers (CRP, ESR) and clinical history.",
        "narrative_short": "Chronic inflammation — correlate with CRP, ESR, history",
        "care_plan": {
            "short_term": "Discuss with your physician; consider CRP/ESR if not already done.",
        },
    },
    {
        "id": "impression_granulomatous_inflammation",
        "keywords": ["granulomatous inflammation", "granulomatous changes",
                     "granulomas seen"],
        "category": "inflammation",
        "severity_class": "concerning",
        "severity_boost": 1,
        "observation": "Granulomatous inflammation noted — workup for tuberculosis, sarcoidosis, or other granulomatous conditions is recommended.",
        "narrative_short": "Granulomatous changes — workup for TB or sarcoidosis",
        "care_plan": {
            "immediate": "Discuss with your physician within 1–2 weeks.",
            "short_term": "Consider TB testing (Mantoux/IGRA), chest imaging, and ACE level.",
            "follow_up": "Specialist referral may be advised.",
        },
    },
]


# ═══════════════════════════════════════════════════════════════════
# CORE MATCHER — case-insensitive substring detection.
# ═══════════════════════════════════════════════════════════════════
def _normalize_text(text: str) -> str:
    """Lowercase + collapse whitespace + strip punctuation edges."""
    if not text:
        return ""
    lowered = text.lower()
    # Collapse whitespace but preserve structure
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _pattern_matches(pattern: dict, normalized_text: str) -> bool:
    """Return True if any keyword in the pattern appears in the text."""
    keywords = pattern.get("keywords") or []
    for kw in keywords:
        if kw and kw.lower() in normalized_text:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════
# CORRELATION CHECKER — cross-references with numeric biomarkers.
# ═══════════════════════════════════════════════════════════════════
def _correlation_matches(
    correlation: dict,
    measurements: list[dict],
) -> bool:
    """
    Check whether a pattern's correlation condition is satisfied by any
    measurement in the list.
    """
    cond = correlation.get("biomarker_condition") or {}
    key_wanted = str(cond.get("key") or "").lower()
    statuses_wanted = [s.lower() for s in (cond.get("status") or [])]
    value_op = cond.get("value_op")
    value_threshold = cond.get("value_threshold")

    if not key_wanted:
        return False

    for m in measurements:
        m_key = str(m.get("key") or "").lower()
        if m_key != key_wanted:
            continue

        # Status check (if specified)
        if statuses_wanted:
            m_status = str(m.get("status") or "").lower()
            if m_status not in statuses_wanted:
                continue

        # Value check (if specified)
        if value_op and value_threshold is not None:
            try:
                v = float(m.get("value"))
                t = float(value_threshold)
                if value_op == ">" and not (v > t):
                    continue
                if value_op == "<" and not (v < t):
                    continue
                if value_op == ">=" and not (v >= t):
                    continue
                if value_op == "<=" and not (v <= t):
                    continue
            except (TypeError, ValueError):
                continue

        return True

    return False


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════
def analyze_text_findings(
    text_findings: list[str],
    measurements: list[dict],
) -> dict:
    """
    Analyze free-text clinical findings and return structured signals.

    Universal design:
      - Iterates every text finding
      - Matches against _TEXT_PATTERNS registry (case-insensitive)
      - Applies correlation logic when biomarker conditions are defined
      - Aggregates observations, care plan actions, and severity boost
      - Deduplicates observations by pattern id
      - Fail-safe: any exception returns empty result shape

    Args:
        text_findings: List of raw text finding strings from the lab report.
        measurements: List of enriched measurement dicts (from
                      _collect_all_measurements). Used for correlation.

    Returns:
        {
          "observations": [str, ...],
          "care_plan_additions": {
              "immediate": [str, ...],
              "short_term": [str, ...],
              "lifestyle": [str, ...],
              "follow_up": [str, ...],
              "long_term": [str, ...],
          },
          "severity_boost": int,          # 0..N (capped at 3 to prevent runaway)
          "correlated_findings": [str, ...],
          "matched_pattern_ids": [str, ...],
        }
    """
    try:
        if not text_findings or not isinstance(text_findings, list):
            return _empty_result()

        # Combine all text findings into one searchable string
        # (also keep individual lines for context)
        combined_text = _normalize_text(" ".join(str(t) for t in text_findings if t))
        if not combined_text:
            return _empty_result()

        measurements = measurements or []

        result = _empty_result()
        matched_ids: set[str] = set()

        for pattern in _TEXT_PATTERNS:
            pid = pattern.get("id") or ""
            if pid in matched_ids:
                continue

            if not _pattern_matches(pattern, combined_text):
                continue

            matched_ids.add(pid)
            result["matched_pattern_ids"].append(pid)

            # Correlation check — if satisfied, use merged observation
            correlation = pattern.get("correlation")
            correlation_hit = False
            if correlation and _correlation_matches(correlation, measurements):
                correlation_hit = True

            # Observation
            if correlation_hit and correlation.get("merged_observation"):
                obs = correlation["merged_observation"]
                result["correlated_findings"].append(obs)
                result["observations"].append(obs)
            else:
                obs = pattern.get("observation")
                if obs:
                    result["observations"].append(obs)

            # Care plan additions
            care_plan = pattern.get("care_plan") or {}
            for bucket in (
                "immediate", "short_term", "lifestyle",
                "follow_up", "long_term",
            ):
                text = care_plan.get(bucket)
                if text and text not in result["care_plan_additions"][bucket]:
                    result["care_plan_additions"][bucket].append(text)

            # Severity boost (base + correlation boost)
            boost = int(pattern.get("severity_boost") or 0)
            if correlation_hit:
                boost += int(correlation.get("boost_severity_by") or 0)
            result["severity_boost"] += boost

        # Cap total boost to prevent runaway when many patterns hit
        if result["severity_boost"] > 3:
            result["severity_boost"] = 3

        # Deduplicate observations while preserving order
        seen = set()
        deduped: list[str] = []
        for obs in result["observations"]:
            key = obs.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(obs)
        result["observations"] = deduped

        if matched_ids:
            logger.info(
                "text_finding_analyzer · patterns matched",
                count=len(matched_ids),
                ids=sorted(matched_ids),
                boost=result["severity_boost"],
            )

        return result

    except Exception:
        logger.exception("text_finding_analyzer · analysis failed; returning empty result")
        return _empty_result()


def get_registered_patterns() -> list[dict[str, Any]]:
    """
    Return a shallow copy of the pattern registry so external consumers
    (e.g. tools.severity_scorer, tools.clinical_picture_synthesizer)
    can synthesize rules from it.

    Each returned dict contains at minimum:
      - id:              pattern identifier
      - severity_class:  reassuring | informational | concerning | urgent | critical
      - severity_boost:  0 | 1 | 2
      - observation:     patient-facing sentence (or None)
      - narrative_short: dashboard-friendly summary (or absent when
                         observation is None)

    Using this API means new patterns added to _TEXT_PATTERNS are
    automatically discovered by consumers without any change in
    those files — the single source of truth remains this registry.
    """
    return [dict(p) for p in _TEXT_PATTERNS]


__all__ = ["analyze_text_findings", "get_registered_patterns"]