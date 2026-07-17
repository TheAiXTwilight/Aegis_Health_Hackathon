"""
tools/lab_constants.py — Canonical lab measurement key constants.

These are the authoritative string keys used in:
    LabReportResult.measurements         (populated by LabReportParser)
    LabReportResult.extra_measurements   (unrecognised keys, preserved)
    SeverityScorer check functions       (read measurements by key)

All keys use British spelling and lowercase. "haemoglobin" is the
canonical form; the parser's alias map accepts US "hemoglobin" and
normalises it to this canonical key.

The parser is responsible for normalising every alias (haemoglobin,
hemoglobin, Hb, HGB, etc.) to these canonical keys at storage time.
No downstream code should ever use raw alias strings — always import
from here.

Alias normalisation map lives in tools/lab_report_parser.py because
normalisation is a parsing concern, not a schema concern.
"""

LAB_KEY_HAEMOGLOBIN = "haemoglobin"
LAB_KEY_TROPONIN    = "troponin"
LAB_KEY_POTASSIUM   = "potassium"
LAB_KEY_SODIUM      = "sodium"
LAB_KEY_CREATININE  = "creatinine"
LAB_KEY_GLUCOSE     = "glucose"
LAB_KEY_WBC         = "wbc"
LAB_KEY_PLATELETS   = "platelets"