"""
tools/lab_thresholds.py — All numeric lab threshold constants.

Single authoritative location for every clinical cutoff value used
in Aegis Health. Two clearly separated categories:

    ABNORMAL_*  — detection thresholds used by LabReportParser.
                  A measurement crossing these thresholds is flagged
                  in LabReportResult.abnormal_values and triggers
                  RULE_ABNORMAL_LAB_ANY (MEDIUM severity).

    CRITICAL_*  — severity thresholds used by SeverityScorer.
                  A measurement crossing these thresholds triggers
                  a HIGH severity rule directly.

Import pattern:
    LabReportParser   → from tools.lab_thresholds import ABNORMAL_*
    SeverityScorer    → from tools.lab_thresholds import CRITICAL_*

Units are documented in constant names to prevent unit-confusion
errors. Changing a threshold requires editing only this file.

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