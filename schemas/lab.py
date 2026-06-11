from __future__ import annotations

from pydantic import BaseModel, Field


class LabReportResult(BaseModel):
    """
    Output of LabReportParser (Step 2).

    abnormal_values  — human-readable findings for report text
    measurements     — structured numeric values for SeverityScorer rule evaluation

    Canonical keys for `measurements` are defined in tools/lab_constants.py
    and documented in docs/lab_keys.md. Producers and consumers must use
    those constants — never hard-coded strings.
    """

    abnormal_values: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable abnormal lab values surfaced in the report. "
            "Populated during extraction."
        ),
    )

    measurements: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Structured numeric lab measurements keyed by standardised test name. "
            "Used by SeverityScorer for threshold-based rules (e.g. troponin, haemoglobin, potassium). "
            "Canonical keys live in tools/lab_constants.py."
        ),
    )

    schema_version: str = "1.0"