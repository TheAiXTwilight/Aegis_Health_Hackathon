"""
tools/unit_normalizer.py — Canonical unit normalization for threshold comparisons.

Solves the unit-mismatch bug where PDF parsers store raw lab values in
instrument units (e.g. platelets as 147000 /cumm) while all threshold
registries across the pipeline use clinical shorthand units
(e.g. platelets < 100  meaning 100 K/µL).

Design principles:
  - Registry-driven: adding a new biomarker normalization rule is one
    dict entry in _UNIT_REGISTRY. No biomarker-specific logic lives
    outside that registry.
  - Universal: every comparison helper in the pipeline routes through
    normalize_for_comparison(); the registry decides what happens.
  - Fail-safe: invalid values return NormalizedValue(value=None, ...).
    Callers that receive None skip the comparison rather than crash or
    produce silent wrong answers.
  - Auditable: NormalizedValue carries the original value, original
    unit, canonical unit, and conversion factor so callers can log
    what happened.
  - Alias-aware: _resolve_canonical_key() maps common parser key
    variants (e.g. "plt", "platelet_count", "plts") to the canonical
    registry key ("platelets") so key matching is robust to PDF parser
    variation.

Public API:
  normalize_for_comparison(canonical_key, value, unit) → NormalizedValue
  _resolve_canonical_key(raw_key) → str   (semi-public; used by matchers)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger


# ═══════════════════════════════════════════════════════════════════
# RESULT TYPE
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NormalizedValue:
    """
    Result of a normalization attempt.

    Attributes:
        value:             The value in canonical clinical units, or None
                           if normalization failed (invalid input, unknown
                           key, non-numeric value, etc.).
        canonical_unit:    The unit the value is now expressed in
                           (e.g. "K/µL", "mg/dL"). Empty string if unknown.
        original_value:    The raw value before conversion.
        original_unit:     The raw unit string before conversion.
        conversion_factor: Factor applied (original × factor = canonical).
                           1.0 when no conversion was needed.
        was_converted:     True when a non-trivial conversion was applied.
    """
    value: float | None
    canonical_unit: str
    original_value: Any
    original_unit: str
    conversion_factor: float
    was_converted: bool


def _null_result(raw_value: Any, raw_unit: str) -> NormalizedValue:
    """Return a failed normalization result (value=None)."""
    return NormalizedValue(
        value=None,
        canonical_unit="",
        original_value=raw_value,
        original_unit=raw_unit,
        conversion_factor=1.0,
        was_converted=False,
    )


# ═══════════════════════════════════════════════════════════════════
# KEY ALIAS REGISTRY
# ═══════════════════════════════════════════════════════════════════
# Maps every known parser key variant → canonical key used in the
# threshold registries across the pipeline.
#
# Rules:
#   - All entries lowercase.
#   - The canonical key (value) must exist as a key in _UNIT_REGISTRY
#     OR be left as-is for keys that need no unit conversion.
#   - Add new aliases freely — no logic changes required.

_KEY_ALIASES: dict[str, str] = {
    # ── Platelets ──────────────────────────────────────────────────
    "plt":                      "platelets",
    "plts":                     "platelets",
    "platelet":                 "platelets",
    "platelet_count":           "platelets",
    "thrombocytes":             "platelets",
    "platelet count":           "platelets",

    # ── Neutrophils (absolute) ─────────────────────────────────────
    "anc":                      "neutrophils_abs",
    "abs_neutrophils":          "neutrophils_abs",
    "absolute_neutrophils":     "neutrophils_abs",
    "neutrophil_abs":           "neutrophils_abs",
    "neutrophils_absolute":     "neutrophils_abs",
    "neutrophils abs":          "neutrophils_abs",
    # percentage key — separate from absolute
    "neu%":                     "neutrophils",
    "neut%":                    "neutrophils",
    "neutrophil%":              "neutrophils",

    # ── WBC (absolute) ────────────────────────────────────────────
    "wbc":                      "wbc",
    "white blood cells":        "wbc",
    "white_blood_cells":        "wbc",
    "leukocytes":               "wbc",
    "total_wbc":                "wbc",
    "total wbc":                "wbc",

    # ── RBC ───────────────────────────────────────────────────────
    "rbc":                      "rbc",
    "red blood cells":          "rbc",
    "red_blood_cells":          "rbc",
    "erythrocytes":             "rbc",

    # ── Haemoglobin ───────────────────────────────────────────────
    "hb":                       "haemoglobin",
    "hgb":                      "haemoglobin",
    "hemoglobin":               "haemoglobin",
    "haemoglobin":              "haemoglobin",

    # ── Glucose ───────────────────────────────────────────────────
    "blood_glucose":            "glucose",
    "fasting_glucose":          "glucose",
    "fbs":                      "glucose",
    "rbs":                      "glucose",
    "blood glucose":            "glucose",
    "fasting glucose":          "glucose",

    # ── Creatinine ────────────────────────────────────────────────
    "serum_creatinine":         "creatinine",
    "s_creatinine":             "creatinine",
    "creat":                    "creatinine",

    # ── Urea / BUN ────────────────────────────────────────────────
    "blood_urea":               "urea",
    "blood urea":               "urea",
    "serum_urea":               "urea",
    "bun":                      "bun",
    "blood_urea_nitrogen":      "bun",

    # ── TSH ───────────────────────────────────────────────────────
    "thyroid_stimulating_hormone": "tsh",
    "thyrotropin":              "tsh",

    # ── Thyroid hormones ──────────────────────────────────────────
    "t3_total":                 "t3",
    "total_t3":                 "t3",
    "t4_total":                 "t4",
    "total_t4":                 "t4",
    "ft3":                      "free_t3",
    "free t3":                  "free_t3",
    "ft4":                      "free_t4",
    "free t4":                  "free_t4",

    # ── HbA1c ─────────────────────────────────────────────────────
    "hba1c":                    "hba1c",
    "glycated_hemoglobin":      "hba1c",
    "glycated_haemoglobin":     "hba1c",
    "a1c":                      "hba1c",

    # ── Lipids ────────────────────────────────────────────────────
    "ldl":                      "ldl_cholesterol",
    "ldl_c":                    "ldl_cholesterol",
    "low_density_lipoprotein":  "ldl_cholesterol",
    "hdl":                      "hdl_cholesterol",
    "hdl_c":                    "hdl_cholesterol",
    "high_density_lipoprotein": "hdl_cholesterol",
    "total_cholesterol":        "total_cholesterol",
    "cholesterol":              "total_cholesterol",
    "tg":                       "triglycerides",
    "trigs":                    "triglycerides",

    # ── Liver enzymes ─────────────────────────────────────────────
    "alt":                      "sgpt_alt",
    "sgpt":                     "sgpt_alt",
    "ast":                      "sgot_ast",
    "sgot":                     "sgot_ast",
    "alp":                      "alp",
    "alkaline_phosphatase":     "alp",
    "ggt":                      "ggt",
    "gamma_gt":                 "ggt",

    # ── Bilirubin ─────────────────────────────────────────────────
    "total_bilirubin":          "bilirubin",
    "tbil":                     "bilirubin",
    "t_bili":                   "bilirubin",

    # ── Proteins ──────────────────────────────────────────────────
    "total_protein":            "total_protein",
    "serum_protein":            "total_protein",
    "albumin":                  "albumin",
    "serum_albumin":            "albumin",
    "globulin":                 "globulin",

    # ── Electrolytes ──────────────────────────────────────────────
    "serum_sodium":             "sodium",
    "na":                       "sodium",
    "serum_potassium":          "potassium",
    "k":                        "potassium",
    "serum_chloride":           "chloride",
    "cl":                       "chloride",

    # ── Blood pressure ────────────────────────────────────────────
    "systolic":                 "bp_systolic",
    "systolic_bp":              "bp_systolic",
    "sbp":                      "bp_systolic",
    "diastolic":                "bp_diastolic",
    "diastolic_bp":             "bp_diastolic",
    "dbp":                      "bp_diastolic",

    # ── Troponin ──────────────────────────────────────────────────
    "troponin_i":               "troponin",
    "troponin_t":               "troponin",
    "high_sensitivity_troponin":"troponin",
    "hs_troponin":              "troponin",

    # ── Inflammatory markers ──────────────────────────────────────
    "c_reactive_protein":       "crp",
    "crp":                      "crp",
    "esr":                      "esr",
    "erythrocyte_sedimentation_rate": "esr",

    # ── Vitamins ──────────────────────────────────────────────────
    "vitamin_d":                "vitamin_d",
    "25_oh_vitamin_d":          "vitamin_d",
    "25ohd":                    "vitamin_d",
    "vit_d":                    "vitamin_d",
    "vitamin_b12":              "vitamin_b12",
    "b12":                      "vitamin_b12",
    "cobalamin":                "vitamin_b12",
    "folate":                   "folate",
    "folic_acid":               "folate",

    # ── Ferritin / Iron ───────────────────────────────────────────
    "serum_ferritin":           "ferritin",
    "serum_iron":               "iron",
    "fe":                       "iron",

    # ── CBC indices ───────────────────────────────────────────────
    "mcv":                      "mcv",
    "mean_corpuscular_volume":  "mcv",
    "mch":                      "mch",
    "mchc":                     "mchc",
    "rdw":                      "rdw",

    # ── SpO2 / vitals ─────────────────────────────────────────────
    "spo2":                     "spo2",
    "oxygen_saturation":        "spo2",
    "o2_sat":                   "spo2",
    "respiratory_rate":         "respiratory_rate",
    "rr":                       "respiratory_rate",

    # ── Potassium (already above, belt-and-braces) ─────────────────
    "k+":                       "potassium",
    "na+":                      "sodium",
}


# ═══════════════════════════════════════════════════════════════════
# UNIT CONVERSION REGISTRY
# ═══════════════════════════════════════════════════════════════════
# Maps canonical_key → list of unit-conversion rules.
#
# Each rule:
#   "raw_units":      list of raw unit strings the parser may emit
#                     (case-insensitive matching applied at runtime)
#   "factor":         multiply raw_value × factor → canonical value
#   "canonical_unit": the unit the result is now in
#
# Rules are evaluated in order; first match wins.
# If no rule matches the raw unit, the value is returned as-is
# (assumed already in canonical units — no conversion applied).
#
# Adding a new biomarker: add one entry here. No logic changes.

_UNIT_REGISTRY: dict[str, list[dict]] = {

    # ── Platelets ─────────────────────────────────────────────────
    # Canonical: K/µL  (= ×10³/µL = ×10³/cumm)
    # Parser emits: 147000 /cumm  →  need 147.0 K/µL
    "platelets": [
        {
            "raw_units": [
                "/cumm", "/mm3", "/mm³", "per cumm", "per mm3",
                "cells/cumm", "cells/mm3", "/µl", "/ul",
                "10^3/ul",       # some parsers emit this without the ×10³ prefix
            ],
            "factor": 1e-3,       # raw ÷ 1000 → K/µL
            "canonical_unit": "K/µL",
        },
        {
            "raw_units": [
                "k/µl", "k/ul", "k/cumm", "×10³/µl", "x10^3/ul",
                "10³/µl", "thou/µl", "thou/ul",
            ],
            "factor": 1.0,        # already in K/µL
            "canonical_unit": "K/µL",
        },
    ],

    # ── WBC (total leukocytes) ─────────────────────────────────────
    # Canonical: K/µL
    "wbc": [
        {
            "raw_units": [
                "/cumm", "/mm3", "/mm³", "cells/cumm", "cells/mm3",
                "/µl", "/ul",
            ],
            "factor": 1e-3,
            "canonical_unit": "K/µL",
        },
        {
            "raw_units": [
                "k/µl", "k/ul", "×10³/µl", "x10^3/ul", "10³/µl",
                "thou/µl",
            ],
            "factor": 1.0,
            "canonical_unit": "K/µL",
        },
    ],

    # ── RBC ───────────────────────────────────────────────────────
    # Canonical: M/µL  (= ×10⁶/µL)
    "rbc": [
        {
            "raw_units": [
                "/cumm", "/mm3", "/mm³", "cells/cumm",
            ],
            "factor": 1e-6,       # raw ÷ 1,000,000 → M/µL
            "canonical_unit": "M/µL",
        },
        {
            "raw_units": [
                "m/µl", "m/ul", "×10⁶/µl", "x10^6/ul", "10⁶/µl",
                "mill/µl",
            ],
            "factor": 1.0,
            "canonical_unit": "M/µL",
        },
    ],

    # ── Neutrophils (absolute) ─────────────────────────────────────
    # Canonical: K/µL
    # Thresholds in registry: neutrophils < 1.5  means 1.5 K/µL
    "neutrophils_abs": [
        {
            "raw_units": [
                "/cumm", "/mm3", "cells/cumm", "/µl", "/ul",
            ],
            "factor": 1e-3,
            "canonical_unit": "K/µL",
        },
        {
            "raw_units": [
                "k/µl", "k/ul", "×10³/µl", "x10^3/ul",
            ],
            "factor": 1.0,
            "canonical_unit": "K/µL",
        },
    ],

    # ── Neutrophils % (differential percentage) ───────────────────
    # Canonical: %  (no conversion; stays as percent)
    "neutrophils": [
        {
            "raw_units": ["%", "percent", "pct"],
            "factor": 1.0,
            "canonical_unit": "%",
        },
    ],

    # ── Haemoglobin ───────────────────────────────────────────────
    # Canonical: g/dL  (no common raw-unit mismatch; included for
    # completeness and future-proofing)
    "haemoglobin": [
        {
            "raw_units": ["g/dl", "gm/dl", "g/100ml", "g%"],
            "factor": 1.0,
            "canonical_unit": "g/dL",
        },
    ],

    # ── Glucose ───────────────────────────────────────────────────
    # Canonical: mg/dL
    "glucose": [
        {
            "raw_units": ["mg/dl", "mg%", "mg/100ml"],
            "factor": 1.0,
            "canonical_unit": "mg/dL",
        },
        {
            "raw_units": ["mmol/l", "mmol/l"],
            "factor": 18.0,       # mmol/L × 18 → mg/dL
            "canonical_unit": "mg/dL",
        },
    ],

    # ── Creatinine ────────────────────────────────────────────────
    # Canonical: mg/dL
    "creatinine": [
        {
            "raw_units": ["mg/dl", "mg%"],
            "factor": 1.0,
            "canonical_unit": "mg/dL",
        },
        {
            "raw_units": ["µmol/l", "umol/l"],
            "factor": 1 / 88.42,  # µmol/L ÷ 88.42 → mg/dL
            "canonical_unit": "mg/dL",
        },
    ],

    # ── Urea ──────────────────────────────────────────────────────
    # Canonical: mg/dL
    "urea": [
        {
            "raw_units": ["mg/dl", "mg%"],
            "factor": 1.0,
            "canonical_unit": "mg/dL",
        },
        {
            "raw_units": ["mmol/l"],
            "factor": 6.006,      # mmol/L × 6.006 → mg/dL
            "canonical_unit": "mg/dL",
        },
    ],

    # ── BUN ───────────────────────────────────────────────────────
    # Canonical: mg/dL
    "bun": [
        {
            "raw_units": ["mg/dl", "mg%"],
            "factor": 1.0,
            "canonical_unit": "mg/dL",
        },
        {
            "raw_units": ["mmol/l"],
            "factor": 2.8,        # mmol/L × 2.8 → mg/dL (BUN)
            "canonical_unit": "mg/dL",
        },
    ],

    # ── TSH ───────────────────────────────────────────────────────
    # Canonical: µIU/mL  (= mIU/L — numerically identical)
    "tsh": [
        {
            "raw_units": [
                "µiu/ml", "uiu/ml", "miu/l", "µu/ml",
                "miu/ml", "iu/ml",
            ],
            "factor": 1.0,
            "canonical_unit": "µIU/mL",
        },
    ],

    # ── T3 (total) ────────────────────────────────────────────────
    # Canonical: ng/dL
    "t3": [
        {
            "raw_units": ["ng/dl"],
            "factor": 1.0,
            "canonical_unit": "ng/dL",
        },
        {
            "raw_units": ["nmol/l"],
            "factor": 65.1,       # nmol/L × 65.1 → ng/dL
            "canonical_unit": "ng/dL",
        },
    ],

    # ── T4 (total) ────────────────────────────────────────────────
    # Canonical: µg/dL
    "t4": [
        {
            "raw_units": ["µg/dl", "ug/dl"],
            "factor": 1.0,
            "canonical_unit": "µg/dL",
        },
        {
            "raw_units": ["nmol/l"],
            "factor": 1 / 12.87,  # nmol/L ÷ 12.87 → µg/dL
            "canonical_unit": "µg/dL",
        },
    ],

    # ── Free T3 ───────────────────────────────────────────────────
    # Canonical: pg/mL
    "free_t3": [
        {
            "raw_units": ["pg/ml"],
            "factor": 1.0,
            "canonical_unit": "pg/mL",
        },
        {
            "raw_units": ["pmol/l"],
            "factor": 0.651,      # pmol/L × 0.651 → pg/mL
            "canonical_unit": "pg/mL",
        },
    ],

    # ── Free T4 ───────────────────────────────────────────────────
    # Canonical: ng/dL
    "free_t4": [
        {
            "raw_units": ["ng/dl"],
            "factor": 1.0,
            "canonical_unit": "ng/dL",
        },
        {
            "raw_units": ["pmol/l"],
            "factor": 0.0777,     # pmol/L × 0.0777 → ng/dL
            "canonical_unit": "ng/dL",
        },
    ],

    # ── HbA1c ─────────────────────────────────────────────────────
    # Canonical: %  (NGSP/DCCT)
    "hba1c": [
        {
            "raw_units": ["%", "percent", "ngsp"],
            "factor": 1.0,
            "canonical_unit": "%",
        },
        {
            "raw_units": ["mmol/mol", "ifcc"],
            # IFCC → NGSP: (mmol/mol × 0.0915) + 2.15
            # Handled via offset; approximation: ×0.0915 + 2.15
            # Factor-only approach inadequate — flag for special handling.
            # For now: not converting; return as-is if unit is mmol/mol.
            # TODO: add offset-based conversions to NormalizedValue.
            "factor": 1.0,
            "canonical_unit": "mmol/mol",
        },
    ],

    # ── Lipids ────────────────────────────────────────────────────
    # Canonical: mg/dL
    "ldl_cholesterol": [
        {
            "raw_units": ["mg/dl", "mg%"],
            "factor": 1.0,
            "canonical_unit": "mg/dL",
        },
        {
            "raw_units": ["mmol/l"],
            "factor": 38.67,
            "canonical_unit": "mg/dL",
        },
    ],
    "hdl_cholesterol": [
        {
            "raw_units": ["mg/dl", "mg%"],
            "factor": 1.0,
            "canonical_unit": "mg/dL",
        },
        {
            "raw_units": ["mmol/l"],
            "factor": 38.67,
            "canonical_unit": "mg/dL",
        },
    ],
    "total_cholesterol": [
        {
            "raw_units": ["mg/dl", "mg%"],
            "factor": 1.0,
            "canonical_unit": "mg/dL",
        },
        {
            "raw_units": ["mmol/l"],
            "factor": 38.67,
            "canonical_unit": "mg/dL",
        },
    ],
    "triglycerides": [
        {
            "raw_units": ["mg/dl", "mg%"],
            "factor": 1.0,
            "canonical_unit": "mg/dL",
        },
        {
            "raw_units": ["mmol/l"],
            "factor": 88.57,
            "canonical_unit": "mg/dL",
        },
    ],

    # ── Liver enzymes ─────────────────────────────────────────────
    # Canonical: U/L
    "sgpt_alt": [
        {
            "raw_units": ["u/l", "iu/l", "u/ml", "µkat/l"],
            "factor": 1.0,
            "canonical_unit": "U/L",
        },
    ],
    "sgot_ast": [
        {
            "raw_units": ["u/l", "iu/l", "u/ml"],
            "factor": 1.0,
            "canonical_unit": "U/L",
        },
    ],
    "alp": [
        {
            "raw_units": ["u/l", "iu/l"],
            "factor": 1.0,
            "canonical_unit": "U/L",
        },
    ],
    "ggt": [
        {
            "raw_units": ["u/l", "iu/l"],
            "factor": 1.0,
            "canonical_unit": "U/L",
        },
    ],

    # ── Bilirubin ─────────────────────────────────────────────────
    # Canonical: mg/dL
    "bilirubin": [
        {
            "raw_units": ["mg/dl", "mg%"],
            "factor": 1.0,
            "canonical_unit": "mg/dL",
        },
        {
            "raw_units": ["µmol/l", "umol/l"],
            "factor": 1 / 17.1,
            "canonical_unit": "mg/dL",
        },
    ],

    # ── Proteins ──────────────────────────────────────────────────
    # Canonical: g/dL
    "total_protein": [
        {
            "raw_units": ["g/dl", "gm/dl", "g%"],
            "factor": 1.0,
            "canonical_unit": "g/dL",
        },
    ],
    "albumin": [
        {
            "raw_units": ["g/dl", "gm/dl", "g%"],
            "factor": 1.0,
            "canonical_unit": "g/dL",
        },
    ],
    "globulin": [
        {
            "raw_units": ["g/dl", "gm/dl"],
            "factor": 1.0,
            "canonical_unit": "g/dL",
        },
    ],

    # ── Electrolytes ──────────────────────────────────────────────
    # Canonical: mEq/L  (= mmol/L for monovalent ions)
    "sodium": [
        {
            "raw_units": ["meq/l", "mmol/l", "mequiv/l"],
            "factor": 1.0,
            "canonical_unit": "mEq/L",
        },
    ],
    "potassium": [
        {
            "raw_units": ["meq/l", "mmol/l", "mequiv/l"],
            "factor": 1.0,
            "canonical_unit": "mEq/L",
        },
    ],
    "chloride": [
        {
            "raw_units": ["meq/l", "mmol/l"],
            "factor": 1.0,
            "canonical_unit": "mEq/L",
        },
    ],

    # ── Blood pressure ────────────────────────────────────────────
    # Canonical: mmHg  (no conversion; included for completeness)
    "bp_systolic": [
        {
            "raw_units": ["mmhg", "mm hg"],
            "factor": 1.0,
            "canonical_unit": "mmHg",
        },
    ],
    "bp_diastolic": [
        {
            "raw_units": ["mmhg", "mm hg"],
            "factor": 1.0,
            "canonical_unit": "mmHg",
        },
    ],

    # ── Troponin ──────────────────────────────────────────────────
    # Canonical: ng/mL
    "troponin": [
        {
            "raw_units": ["ng/ml", "µg/l"],
            "factor": 1.0,
            "canonical_unit": "ng/mL",
        },
        {
            "raw_units": ["ng/l", "pg/ml"],
            "factor": 1e-3,       # ng/L ÷ 1000 → ng/mL
            "canonical_unit": "ng/mL",
        },
    ],

    # ── CRP ───────────────────────────────────────────────────────
    # Canonical: mg/L
    "crp": [
        {
            "raw_units": ["mg/l"],
            "factor": 1.0,
            "canonical_unit": "mg/L",
        },
        {
            "raw_units": ["mg/dl"],
            "factor": 10.0,       # mg/dL × 10 → mg/L
            "canonical_unit": "mg/L",
        },
    ],

    # ── ESR ───────────────────────────────────────────────────────
    # Canonical: mm/hr
    "esr": [
        {
            "raw_units": ["mm/hr", "mm/hour", "mm/h"],
            "factor": 1.0,
            "canonical_unit": "mm/hr",
        },
    ],

    # ── Vitamin D ─────────────────────────────────────────────────
    # Canonical: ng/mL
    "vitamin_d": [
        {
            "raw_units": ["ng/ml", "ng/dl"],
            "factor": 1.0,
            "canonical_unit": "ng/mL",
        },
        {
            "raw_units": ["nmol/l"],
            "factor": 0.401,      # nmol/L × 0.401 → ng/mL
            "canonical_unit": "ng/mL",
        },
    ],

    # ── Vitamin B12 ───────────────────────────────────────────────
    # Canonical: pg/mL
    "vitamin_b12": [
        {
            "raw_units": ["pg/ml", "ng/l"],
            "factor": 1.0,
            "canonical_unit": "pg/mL",
        },
        {
            "raw_units": ["pmol/l"],
            "factor": 1.355,      # pmol/L × 1.355 → pg/mL
            "canonical_unit": "pg/mL",
        },
    ],

    # ── Folate ────────────────────────────────────────────────────
    # Canonical: ng/mL
    "folate": [
        {
            "raw_units": ["ng/ml"],
            "factor": 1.0,
            "canonical_unit": "ng/mL",
        },
        {
            "raw_units": ["nmol/l"],
            "factor": 0.441,
            "canonical_unit": "ng/mL",
        },
    ],

    # ── Ferritin ──────────────────────────────────────────────────
    # Canonical: ng/mL  (= µg/L — numerically identical)
    "ferritin": [
        {
            "raw_units": ["ng/ml", "µg/l", "ug/l"],
            "factor": 1.0,
            "canonical_unit": "ng/mL",
        },
    ],

    # ── Iron ──────────────────────────────────────────────────────
    # Canonical: µg/dL
    "iron": [
        {
            "raw_units": ["µg/dl", "ug/dl", "mcg/dl"],
            "factor": 1.0,
            "canonical_unit": "µg/dL",
        },
        {
            "raw_units": ["µmol/l", "umol/l"],
            "factor": 5.585,      # µmol/L × 5.585 → µg/dL
            "canonical_unit": "µg/dL",
        },
    ],

    # ── CBC indices ───────────────────────────────────────────────
    "mcv": [
        {
            "raw_units": ["fl", "fL"],
            "factor": 1.0,
            "canonical_unit": "fL",
        },
    ],
    "mch": [
        {
            "raw_units": ["pg"],
            "factor": 1.0,
            "canonical_unit": "pg",
        },
    ],
    "mchc": [
        {
            "raw_units": ["g/dl", "gm/dl"],
            "factor": 1.0,
            "canonical_unit": "g/dL",
        },
    ],
    "rdw": [
        {
            "raw_units": ["%"],
            "factor": 1.0,
            "canonical_unit": "%",
        },
    ],

    # ── SpO2 ──────────────────────────────────────────────────────
    "spo2": [
        {
            "raw_units": ["%", "percent"],
            "factor": 1.0,
            "canonical_unit": "%",
        },
    ],
}


# ═══════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════

def _resolve_canonical_key(raw_key: str) -> str:
    """
    Resolve a raw parser key to its canonical registry key.

    Steps:
      1. Lowercase + strip.
      2. Direct lookup in _KEY_ALIASES.
      3. If not found, return the normalized key as-is (may already
         be canonical, or may be unknown — the caller handles both).

    Never raises.
    """
    if not raw_key:
        return ""
    k = str(raw_key).lower().strip()
    return _KEY_ALIASES.get(k, k)


def _normalize_unit_string(unit: str | None) -> str:
    """
    Normalize a raw unit string for registry lookup:
      - Lowercase
      - Strip surrounding whitespace
      - Unify unicode micro sign (µ / μ) → µ
      - Collapse internal whitespace

    Returns "" for None or empty input.
    """
    if not unit:
        return ""
    u = str(unit).lower().strip()
    # Unify both Unicode micro signs to a single canonical form
    u = u.replace("\u03bc", "\u00b5")   # Greek small letter mu → micro sign
    u = u.replace(" ", "")              # "mm /hr" → "mm/hr" (no spaces)
    return u


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════

def normalize_for_comparison(
    canonical_key: str,
    value: float | int | None,
    unit: str | None = None,
) -> NormalizedValue:
    """
    Normalize a biomarker value to canonical clinical units for
    threshold comparison.

    Args:
        canonical_key: The biomarker key as used in threshold registries
                       (e.g. "platelets", "neutrophils_abs", "glucose").
                       Raw parser keys are resolved via _resolve_canonical_key
                       before registry lookup, so passing "plt" or
                       "platelet_count" works identically to "platelets".
        value:         The raw value from the lab report (numeric).
                       May be int, float, or str-coercible to float.
                       Returns NormalizedValue(None) for non-numeric input.
        unit:          The raw unit string from the lab report
                       (e.g. "/cumm", "K/µL", "mg/dL", None).
                       If None or empty: no conversion is applied and
                       the raw numeric value is returned as-is.

    Returns:
        NormalizedValue with:
          .value             float in canonical units, or None on error
          .canonical_unit    str  (e.g. "K/µL")
          .original_value    the raw input value
          .original_unit     the raw unit string
          .conversion_factor the factor applied (1.0 if no conversion)
          .was_converted     True if a non-trivial factor was applied

    Behavior:
      - Unknown key, unknown unit-for-known-key: value returned as-is
        (raw float). This is safe because unknown keys means we have no
        threshold to compare against, and unknown units means we assume
        the value is already in canonical form (common for well-behaved
        parsers).
      - None or non-numeric value: NormalizedValue(None) returned.
        Callers should skip comparisons when .value is None.

    Fail-safe: any internal error returns NormalizedValue(None) and
    logs a warning. Never raises.

    Examples:
        >>> normalize_for_comparison("platelets", 147000, "/cumm")
        NormalizedValue(value=147.0, canonical_unit='K/µL', ...)

        >>> normalize_for_comparison("platelets", 147, "K/µL")
        NormalizedValue(value=147.0, canonical_unit='K/µL', ...)

        >>> normalize_for_comparison("glucose", 95, "mg/dL")
        NormalizedValue(value=95.0, canonical_unit='mg/dL', ...)

        >>> normalize_for_comparison("tsh", 5.2, "µIU/mL")
        NormalizedValue(value=5.2, canonical_unit='µIU/mL', ...)

        >>> normalize_for_comparison("platelets", None, "/cumm")
        NormalizedValue(value=None, ...)
    """
    raw_unit = str(unit or "")

    # ── Step 1: Parse raw value to float ──────────────────────────────
    if value is None:
        return _null_result(value, raw_unit)

    try:
        raw_float = float(value)
    except (TypeError, ValueError):
        logger.debug(
            "unit_normalizer · non-numeric value skipped",
            key=canonical_key,
            value=value,
        )
        return _null_result(value, raw_unit)

    # ── Step 2: Resolve key to canonical form ─────────────────────────
    resolved_key = _resolve_canonical_key(str(canonical_key or ""))

    # ── Step 3: Look up conversion rules ─────────────────────────────
    rules = _UNIT_REGISTRY.get(resolved_key)
    if not rules:
        # Key not in registry — return raw value as-is; no conversion.
        # This is safe: the caller compares against thresholds that were
        # written for whatever unit this key uses natively.
        return NormalizedValue(
            value=raw_float,
            canonical_unit="",
            original_value=value,
            original_unit=raw_unit,
            conversion_factor=1.0,
            was_converted=False,
        )

    # ── Step 4: No unit provided — return raw value as-is ────────────
    unit_norm = _normalize_unit_string(raw_unit)
    if not unit_norm:
        # No unit information: assume value is already in canonical units.
        canonical_unit = rules[0]["canonical_unit"] if rules else ""
        return NormalizedValue(
            value=raw_float,
            canonical_unit=canonical_unit,
            original_value=value,
            original_unit=raw_unit,
            conversion_factor=1.0,
            was_converted=False,
        )

    # ── Step 5: Match unit against registry rules ─────────────────────
    try:
        for rule in rules:
            rule_units = [
                _normalize_unit_string(u) for u in (rule.get("raw_units") or [])
            ]
            if unit_norm in rule_units:
                factor = float(rule["factor"])
                converted = raw_float * factor
                return NormalizedValue(
                    value=converted,
                    canonical_unit=rule["canonical_unit"],
                    original_value=value,
                    original_unit=raw_unit,
                    conversion_factor=factor,
                    was_converted=(factor != 1.0),
                )

        # Unit recognized in structure but not matched in any rule —
        # return raw value. Handles parsers that emit unusual but
        # already-canonical unit strings.
        canonical_unit = rules[0]["canonical_unit"] if rules else ""
        logger.debug(
            "unit_normalizer · unit not matched in registry; returning raw value",
            key=resolved_key,
            unit=raw_unit,
            unit_normalized=unit_norm,
        )
        return NormalizedValue(
            value=raw_float,
            canonical_unit=canonical_unit,
            original_value=value,
            original_unit=raw_unit,
            conversion_factor=1.0,
            was_converted=False,
        )

    except Exception:
        logger.warning(
            "unit_normalizer · conversion error; returning raw value",
            key=resolved_key,
            value=value,
            unit=raw_unit,
        )
        return _null_result(value, raw_unit)


def normalize_display_value(
    canonical_key: str,
    value: float | int | None,
    unit: str | None = None,
) -> NormalizedValue:
    """
    Like normalize_for_comparison but also applies conversion when the
    value magnitude is implausible for the canonical unit — handles the
    case where the PDF parser stored a raw instrument value (e.g. 147000)
    but already labelled it with a canonical unit (K/µL) by mistake,
    or where unit resolution lost the original raw unit string.

    Magnitude thresholds per canonical key:
      If value > magnitude_threshold AND canonical unit matches,
      apply the known conversion factor.

    This is DISPLAY-ONLY — never use for threshold comparisons.
    """
    # First try standard normalization
    result = normalize_for_comparison(canonical_key, value, unit)
    # Only return early if already converted or value is invalid.
    # Do NOT return early on was_converted=False — the magnitude check
    # below must still run for values stored in instrument scale
    # (e.g. 147000) even when the unit string is already canonical.
    if result.value is None:
        return result
    if result.was_converted:
        return result

    # Magnitude-based fallback: raw value looks like instrument units
    # even though the unit string appears canonical.
    resolved_key = _resolve_canonical_key(str(canonical_key or ""))

    _MAGNITUDE_THRESHOLDS: dict[str, tuple[float, float, str]] = {
        # key: (threshold, factor, canonical_unit)
        # If value > threshold, multiply by factor to get display value
        "platelets":      (10_000, 1e-3, "K/µL"),
        "wbc":            (1_000,  1e-3, "K/µL"),
        "neutrophils_abs":(1_000,  1e-3, "K/µL"),
        "rbc":            (1_000,  1e-6, "M/µL"),
    }

    if resolved_key in _MAGNITUDE_THRESHOLDS:
        threshold, factor, canon_unit = _MAGNITUDE_THRESHOLDS[resolved_key]
        raw_float = result.original_value
        try:
            raw_float = float(raw_float)
        except (TypeError, ValueError):
            return result
        if raw_float > threshold:
            converted = raw_float * factor
            return NormalizedValue(
                value=converted,
                canonical_unit=canon_unit,
                original_value=value,
                original_unit=str(unit or ""),
                conversion_factor=factor,
                was_converted=True,
            )

    return result

# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE ALIAS — matches the import name used in
# patient_context_adapter.py's top-level import
# ═══════════════════════════════════════════════════════════════════
normalize_value_for_comparison = normalize_for_comparison


__all__ = [
    "normalize_for_comparison",
    "normalize_value_for_comparison",   # alias
    "normalize_display_value",
    "_resolve_canonical_key",           # semi-public; used by matchers
    "NormalizedValue",
]