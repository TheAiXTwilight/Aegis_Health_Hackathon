"""
tools/lab_thresholds.py — All numeric lab threshold constants + reference ranges.

Single authoritative location for every clinical cutoff value used
in Aegis Health. Three clearly separated categories:

    ABNORMAL_*        — detection thresholds used by LabReportParser.
                        A measurement crossing these thresholds is flagged
                        in LabReportResult.abnormal_values and triggers
                        RULE_ABNORMAL_LAB_ANY (MEDIUM severity).

    CRITICAL_*        — severity thresholds used by SeverityScorer.
                        A measurement crossing these thresholds triggers
                        a HIGH severity rule directly.

        REFERENCE_RANGES  — normal clinical ranges for display/UI.
                        Shipped to the frontend so it can show ranges
                        without hardcoding anything.

                        Optional per-entry metadata flags:
                          "upper_is_nominal": True
                             → display as "≥ low" instead of "low–high"
                             (use when the upper bound is an artificial
                              reference ceiling, e.g. Vitamin D 30–100)
                          "lower_is_nominal": True
                             → display as "≤ high" instead of "low–high"
                             (use when the lower bound is nominal,
                              e.g. some inflammatory markers)
                        Flags affect DISPLAY ONLY. Classification logic
                        continues to use both numeric bounds.

    CANONICAL_UNITS   — canonical unit per lab key. Used only as fallback
                        when the parser couldn't extract a unit from the
                        actual PDF text (which is the preferred source).

Import pattern:
    LabReportParser   → from tools.lab_thresholds import ABNORMAL_*, REFERENCE_RANGES, CANONICAL_UNITS
    SeverityScorer    → from tools.lab_thresholds import CRITICAL_*

Units are documented in constant names to prevent unit-confusion errors.
Changing a threshold or range requires editing only this file.

Troponin note: ABNORMAL and CRITICAL share the same value (0.04).
They are intentionally separate constants — they may diverge in
future clinical updates, and sharing one constant would couple
parser and scorer to the same value inadvertently.
"""

# ── Abnormal detection thresholds (LabReportParser) ───────────────
# Crossing these flags a value as abnormal in the report.
# Lower bar than CRITICAL — catches clinically notable but
# not immediately life-threatening values.

ABNORMAL_LOW_HAEMOGLOBIN_G_DL   = 12.0   # Hb < 12.0 g/dL      → abnormal
ABNORMAL_HIGH_GLUCOSE_MG_DL    = 180.0  # Glucose > 180 mg/dL  → abnormal
ABNORMAL_HIGH_POTASSIUM_MMOL_L =   5.5  # K+ > 5.5 mmol/L      → abnormal
ABNORMAL_HIGH_TROPONIN_NG_ML   =   0.04 # Troponin > 0.04 ng/mL → abnormal


# ── Critical severity thresholds (SeverityScorer) ─────────────────
# Crossing these triggers a HIGH severity rule.
# Higher clinical urgency than ABNORMAL thresholds.

CRITICAL_HAEMOGLOBIN_G_DL   = 7.0    # Hb < 7.0 g/dL   → RULE_CRITICAL_LAB_HAEMOGLOBIN
CRITICAL_POTASSIUM_MMOL_L  = 6.5    # K+ > 6.5 mmol/L  → RULE_CRITICAL_LAB_POTASSIUM
CRITICAL_TROPONIN_NG_ML    = 0.04   # Troponin > 0.04   → RULE_CRITICAL_LAB_TROPONIN


# ── Reference ranges for display (LabReportParser → Frontend) ─────
# Complete normal-range dictionary keyed by canonical lab key.
# Shipped verbatim in LabReportResult.reference_ranges so the frontend
# never has to hardcode these values.
#
# Format: { canonical_key: { "low": float|None, "high": float|None } }
# - Both bounds present → two-sided range (e.g. Hb 12.0–17.5)
# - Only "high" present → upper-limit-only (e.g. cholesterol < 200)
# - Only "low" present  → lower-limit-only (e.g. HDL > 40)
#
# Values here mirror the ABNORMAL_* / CRITICAL_* constants above
# and standard clinical reference intervals for values that don't
# have hard thresholds.

REFERENCE_RANGES: dict[str, dict[str, float | None]] = {
    # Blood — thresholds from ABNORMAL_* constants where applicable
    "haemoglobin":       {"low": ABNORMAL_LOW_HAEMOGLOBIN_G_DL, "high": 17.5},
    "wbc":               {"low": 4.0,   "high": 11.0},
    "rbc":               {"low": 4.2,   "high": 5.9},
    "platelets":         {"low": 150,   "high": 450},
    "hematocrit":        {"low": 36,    "high": 50},

    # Metabolic
    "glucose":           {"low": 70,    "high": ABNORMAL_HIGH_GLUCOSE_MG_DL},
    "hba1c":             {"low": 4.0,   "high": 5.6},

    # Electrolytes
    "potassium":         {"low": 3.5,   "high": ABNORMAL_HIGH_POTASSIUM_MMOL_L},
    "sodium":            {"low": 135,   "high": 145},

    # Cardiac
    "troponin":          {"low": 0,     "high": ABNORMAL_HIGH_TROPONIN_NG_ML},

    # Kidney / Liver
    "creatinine":        {"low": 0.6,   "high": 1.3},
    "urea":              {"low": 15,    "high": 45},
    "bun":               {"low": 7,     "high": 20},
    "uric_acid":         {"low": 3.4,   "high": 7.0},
    "sgpt_alt":          {"low": 7,     "high": 56},
    "sgot_ast":          {"low": 8,     "high": 48},
    "bilirubin":         {"low": 0.1,   "high": 1.2},

    # Iron
    "iron":              {"low": 60,    "high": 170},

    # Lipids — thresholds from _detect_abnormal() in lab_report_parser.py
    "total_cholesterol": {"low": None,  "high": 200},   # desirable < 200
    "ldl_cholesterol":   {"low": None,  "high": 130},   # borderline/high from 130
    "hdl_cholesterol":   {"low": 40,    "high": None},  # low below 40
    "vldl_cholesterol":  {"low": 5,     "high": 40},
    "triglycerides":     {"low": None,  "high": 150},   # normal < 150

    # Vitamins
    # Vitamin D: upper bound (100) is a nominal ceiling — clinical guidance
    # is defined by the ≥ 30 ng/mL sufficiency threshold. Display as "≥ 30".
    "vitamin_d":         {"low": 30,    "high": 100, "upper_is_nominal": True},
    # Vitamin B12: upper bound (900) is a nominal reference ceiling; the
    # clinically meaningful cutoff is the lower deficiency threshold.
    "vitamin_b12":       {"low": 200,   "high": 900, "upper_is_nominal": True},

    # Thyroid
    "tsh":               {"low": 0.4,   "high": 4.5},
    "t3":                {"low": 80,    "high": 200},
    "t4":                {"low": 4.5,   "high": 12.0},

    # Vitals (used by frontend when vitals come through as measurements)
    "bp_systolic":       {"low": 90,    "high": 120},
    "bp_diastolic":      {"low": 60,    "high": 80},
    "heart_rate":        {"low": 60,    "high": 100},
    "spo2":              {"low": 95,    "high": 100},
    "temperature":       {"low": 97.0,  "high": 99.5},
    "respiratory_rate":  {"low": 12,    "high": 20},
}


# ── Canonical units per lab key (fallback only) ───────────────────
# Used ONLY when the parser could not extract a unit from the PDF text.
# The extracted unit from the raw report ALWAYS wins if present.

CANONICAL_UNITS: dict[str, str] = {
    # Blood
    "haemoglobin":       "g/dL",
    "wbc":               "K/µL",
    "rbc":               "M/µL",
    "platelets":         "K/µL",
    "hematocrit":        "%",

    # Metabolic
    "glucose":           "mg/dL",
    "hba1c":             "%",

    # Electrolytes
    "potassium":         "mmol/L",
    "sodium":            "mmol/L",

    # Cardiac
    "troponin":          "ng/mL",

    # Kidney / Liver
    "creatinine":        "mg/dL",
    "urea":              "mg/dL",
    "bun":               "mg/dL",
    "uric_acid":         "mg/dL",
    "sgpt_alt":          "U/L",
    "sgot_ast":          "U/L",
    "bilirubin":         "mg/dL",

    # Iron
    "iron":              "µg/dL",

    # Lipids
    "total_cholesterol": "mg/dL",
    "hdl_cholesterol":   "mg/dL",
    "ldl_cholesterol":   "mg/dL",
    "vldl_cholesterol":  "mg/dL",
    "triglycerides":     "mg/dL",

    # Vitamins
    "vitamin_d":         "ng/mL",
    "vitamin_b12":       "pg/mL",

    # Thyroid
    "tsh":               "µIU/mL",
    "t3":                "ng/dL",
    "t4":                "µg/dL",

    # Vitals
    "bp_systolic":       "mmHg",
    "bp_diastolic":      "mmHg",
    "heart_rate":        "bpm",
    "spo2":              "%",
    "temperature":       "°F",
    "respiratory_rate":  "/min",
}