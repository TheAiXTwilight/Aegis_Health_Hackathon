from __future__ import annotations

from pydantic import BaseModel, Field


class LabReportResult(BaseModel):
    """
    Output of LabReportParser (Step 2).

    abnormal_values    — human-readable findings for report text
    measurements       — structured numeric values (canonical keys only)
    extra_measurements — unrecognized lab keys preserved for future use
    units              — per-key unit strings extracted from the PDF
    reference_ranges   — per-key normal range extracted from the PDF
    text_findings      — non-numeric interpretive lines (morphology,
                         peripheral smear, impressions, notes, etc.)

    Canonical keys for `measurements` are defined in tools/lab_constants.py.
    All numeric thresholds live in tools/lab_thresholds.py.
    """

    abnormal_values: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable abnormal lab findings surfaced in the report."
        ),
    )

    measurements: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Structured numeric lab measurements keyed by canonical name. "
            "Used by SeverityScorer for threshold-based rules."
        ),
    )

    extra_measurements: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Preserved lab measurements not yet recognized by the canonical "
            "lab key set. Future rules may consume these without reparsing."
        ),
    )

    units: dict[str, str] = Field(
        default_factory=dict,
        description="Canonical key -> unit string extracted from PDF (best-effort).",
    )
    reference_ranges: dict[str, dict[str, float | None]] = Field(
        default_factory=dict,
        description="Canonical key -> {low, high} normal reference range.",
    )

    text_findings: list[str] = Field(
        default_factory=list,
        description=(
            "Non-numeric interpretive findings extracted from the lab report — "
            "morphology descriptions, peripheral smear examinations, clinical "
            "impressions, and other free-text observations. Each item is a "
            "single self-contained sentence like "
            "'RBC Morphology: Red cells are normocytic and normochromic'."
        ),
    )

    schema_version: str = "1.0"