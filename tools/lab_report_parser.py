"""
tools/lab_report_parser.py — Lab report parser (Step 2).

Placeholder implementation. Real implementation uses:
    PyMuPDF → pdfminer.six → EasyOCR (opt-in)
without changing this module's public interface.

Changes from original:
    - PDF magic byte detection: returns ToolError for real PDFs.
    - Full alias normalization map: all recognized variants → canonical key.
    - Unknown measurements preserved in extra_measurements.
    - Abnormal detection uses ABNORMAL_* thresholds from lab_thresholds.py.
    - Removes internal state.lab_result assignment (pipeline owns state).
    - Uses tool_names and lab_constants throughout.

Note on repeated canonical keys:
    If a fixture contains the same measurement twice under different aliases
    (e.g. "Hb: 11" and "Haemoglobin: 12"), the last parsed value wins.
    Production parser should define an explicit policy (first wins / last
    wins / validation error). Placeholder behavior is acceptable for
    text fixtures.
"""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

from schemas.errors import ToolError
from schemas.lab import LabReportResult
from schemas.state import AegisState
from tools.lab_constants import (
    LAB_KEY_CREATININE,
    LAB_KEY_GLUCOSE,
    LAB_KEY_HAEMOGLOBIN,
    LAB_KEY_PLATELETS,
    LAB_KEY_POTASSIUM,
    LAB_KEY_SODIUM,
    LAB_KEY_TROPONIN,
    LAB_KEY_WBC,
)
from tools.lab_thresholds import (
    ABNORMAL_HIGH_GLUCOSE_MG_DL,
    ABNORMAL_HIGH_POTASSIUM_MMOL_L,
    ABNORMAL_HIGH_TROPONIN_NG_ML,
    ABNORMAL_LOW_HAEMOGLOBIN_G_DL,
)
from tools.tool_names import TOOL_LAB_REPORT_PARSER


# ── PDF detection ─────────────────────────────────────────────────

_PDF_MAGIC = b"%PDF"


def _is_real_pdf(path: Path) -> bool:
    """Return True if file starts with PDF magic bytes."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == _PDF_MAGIC
    except OSError:
        return False


# ── Alias normalization map ───────────────────────────────────────
# Keys: all recognized alias strings (lowercased, stripped).
# Values: canonical lab key constants from lab_constants.py.
# Parser normalizes at storage time — downstream code sees only
# canonical keys and never deals with aliases.

_ALIAS_MAP: dict[str, str] = {
    # Haemoglobin
    "haemoglobin":        LAB_KEY_HAEMOGLOBIN,
    "hb":                 LAB_KEY_HAEMOGLOBIN,
    "hgb":                LAB_KEY_HAEMOGLOBIN,
    "h b":                LAB_KEY_HAEMOGLOBIN,

    # Troponin
    "troponin":          LAB_KEY_TROPONIN,
    "troponin i":        LAB_KEY_TROPONIN,
    "troponin t":        LAB_KEY_TROPONIN,
    "trop-i":            LAB_KEY_TROPONIN,
    "trop-t":            LAB_KEY_TROPONIN,
    "trop i":            LAB_KEY_TROPONIN,
    "trop t":            LAB_KEY_TROPONIN,
    "ctni":              LAB_KEY_TROPONIN,
    "ctnt":              LAB_KEY_TROPONIN,

    # Potassium
    "potassium":         LAB_KEY_POTASSIUM,
    "k":                 LAB_KEY_POTASSIUM,
    "k+":                LAB_KEY_POTASSIUM,

    # Sodium
    "sodium":            LAB_KEY_SODIUM,
    "na":                LAB_KEY_SODIUM,
    "na+":               LAB_KEY_SODIUM,

    # Creatinine
    "creatinine":        LAB_KEY_CREATININE,
    "cr":                LAB_KEY_CREATININE,
    "creat":             LAB_KEY_CREATININE,

    # Glucose
    "glucose":           LAB_KEY_GLUCOSE,
    "glu":               LAB_KEY_GLUCOSE,
    "blood glucose":     LAB_KEY_GLUCOSE,
    "fasting glucose":   LAB_KEY_GLUCOSE,

    # WBC
    "wbc":               LAB_KEY_WBC,
    "white blood cell":  LAB_KEY_WBC,
    "white blood cells": LAB_KEY_WBC,
    "leukocytes":        LAB_KEY_WBC,
    "total wbc":         LAB_KEY_WBC,

    # Platelets
    "platelets":         LAB_KEY_PLATELETS,
    "plt":               LAB_KEY_PLATELETS,
    "platelet count":    LAB_KEY_PLATELETS,
    "thrombocytes":      LAB_KEY_PLATELETS,
}


# ── Abnormal value detection ──────────────────────────────────────

def _detect_abnormal(key: str, value: float) -> str | None:
    """
    Return a human-readable abnormal finding string, or None if normal.
    Uses ABNORMAL_* thresholds from lab_thresholds.py.
    """
    if key == LAB_KEY_HAEMOGLOBIN and value < ABNORMAL_LOW_HAEMOGLOBIN_G_DL:
        return (
            f"Low haemoglobin: {value} g/dL "
            f"(threshold < {ABNORMAL_LOW_HAEMOGLOBIN_G_DL})"
        )
    if key == LAB_KEY_GLUCOSE and value > ABNORMAL_HIGH_GLUCOSE_MG_DL:
        return (
            f"High glucose: {value} mg/dL "
            f"(threshold > {ABNORMAL_HIGH_GLUCOSE_MG_DL})"
        )
    if key == LAB_KEY_POTASSIUM and value > ABNORMAL_HIGH_POTASSIUM_MMOL_L:
        return (
            f"High potassium: {value} mmol/L "
            f"(threshold > {ABNORMAL_HIGH_POTASSIUM_MMOL_L})"
        )
    if key == LAB_KEY_TROPONIN and value > ABNORMAL_HIGH_TROPONIN_NG_ML:
        return (
            f"Elevated troponin: {value} ng/mL "
            f"(threshold > {ABNORMAL_HIGH_TROPONIN_NG_ML})"
        )
    return None


# ── Parser ────────────────────────────────────────────────────────

class LabReportParser:
    """
    Parses laboratory reports into structured LabReportResult.
    Does not write to state — pipeline owns state mutation.
    """

    TOOL_NAME = TOOL_LAB_REPORT_PARSER

    async def run(
        self,
        state: AegisState,
    ) -> LabReportResult | ToolError:

        try:
            if not state.lab_pdf_path:
                return ToolError(
                    tool=TOOL_LAB_REPORT_PARSER,
                    reason="No laboratory report path supplied.",
                    fatal=False,
                )

            path = Path(state.lab_pdf_path)

            if not path.is_file():
                return ToolError(
                    tool=TOOL_LAB_REPORT_PARSER,
                    reason=f"Lab report file not found: {state.lab_pdf_path}",
                    fatal=False,
                )

            # Detect real PDF — placeholder cannot parse binary PDF.
            if _is_real_pdf(path):
                logger.warning(
                    "lab_report_parser · real PDF detected · "
                    "placeholder only supports text fixtures",
                    path=str(path),
                )
                return ToolError(
                    tool=TOOL_LAB_REPORT_PARSER,
                    reason=(
                        "PDF parsing requires PyMuPDF — "
                        "placeholder only supports text fixtures."
                    ),
                    fatal=False,
                )

            text = path.read_text(encoding="utf-8", errors="ignore").lower()

            measurements:       dict[str, float] = {}
            extra_measurements: dict[str, float] = {}
            abnormal_values:    list[str]         = []

            pattern = re.compile(
                r"([a-zA-Z][a-zA-Z0-9 _\-]*?)"
                r"\s*[:=]\s*"
                r"([0-9]+(?:\.[0-9]+)?)"
            )

            for match in pattern.finditer(text):
                raw_key = match.group(1).strip().lower()
                value   = float(match.group(2))

                canonical = _ALIAS_MAP.get(raw_key)

                if canonical is not None:
                    measurements[canonical] = value
                    finding = _detect_abnormal(canonical, value)
                    if finding:
                        abnormal_values.append(finding)
                else:
                    extra_measurements[raw_key] = value
                    logger.debug(
                        "lab_report_parser · unrecognized lab key preserved",
                        raw_key=raw_key,
                        value=value,
                    )

            return LabReportResult(
                abnormal_values=abnormal_values,
                measurements=measurements,
                extra_measurements=extra_measurements,
            )

        except Exception as exc:
            return ToolError(
                tool=TOOL_LAB_REPORT_PARSER,
                reason=str(exc),
                fatal=False,
            )


async def parse(state: AegisState) -> LabReportResult | ToolError:
    """Canonical functional entrypoint."""
    return await LabReportParser().run(state)