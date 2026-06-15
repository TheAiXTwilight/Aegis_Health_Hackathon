from __future__ import annotations

from pydantic import BaseModel, Field


class LabReportResult(BaseModel):
    """
    Output of LabReportParser (Step 2).

    abnormal_values    — human-readable findings for report text
    measurements       — structured numeric values (canonical keys only)
    extra_measurements — unrecognized lab keys preserved for future use

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

    schema_version: str = "1.0"