"""
schemas/validation.py — RuleValidatorResult schema.

Produced by RuleValidator (Step 9) after ReportGenerator completes.

RuleValidator compares the deterministic severity level from
SeverityScorer against the severity level expressed in the LLM
narrative (extracted from the ### Severity section of the report
using whole-word regex matching).

Three-state output:

    agreement   SLM narrative and deterministic rules concur.

    warning     Minor discrepancy or unextractable narrative level.
                Flagged for clinician review. No automatic correction.

    override    Safety-critical conflict: deterministic rules require
                HIGH but the narrative does not express HIGH.
                TriageReport.severity is already set from the
                deterministic result — override surfaces the conflict
                for the UI to display a safety banner.

Override condition:
    deterministic_level == "HIGH" and slm_narrative_level != "HIGH"

    The safety risk is false reassurance: a clinician reading LOW or
    MEDIUM when critical conditions were detected by rules.
    The reverse (overclaiming HIGH when rules say LOW) is not a
    clinical safety hazard — it produces WARNING not OVERRIDE.

Stored on AegisState.rule_validator_result (full structured result).
String value stored on TriageReport.validation_status.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class ValidationStatus(str, Enum):
    """
    Three-state output of RuleValidator.

    String enum — .value gives the string directly for serialisation
    into TriageReport.validation_status without an extra .value call.
    """

    AGREEMENT = "agreement"
    WARNING   = "warning"
    OVERRIDE  = "override"


class RuleValidatorResult(BaseModel):
    """
    Output of RuleValidator (Step 9).

    deterministic_level
        Level from SeverityResult — always authoritative.
        Equals state.severity_result.level at validation time.

    slm_narrative_level
        Level extracted from ### Severity section of the report.
        "HIGH", "MEDIUM", "LOW", or None when extraction fails.
        None triggers WARNING status.

    disagreement_reason
        Human-readable explanation of the discrepancy.
        None when status == AGREEMENT.

    overridden
        True only when status == OVERRIDE. Redundant with status but
        simplifies downstream checks:
            if result.overridden: show_safety_banner()
    """

    status:              ValidationStatus
    deterministic_level: Literal["LOW", "MEDIUM", "HIGH"]
    slm_narrative_level: str | None = None
    disagreement_reason: str | None = None
    overridden:          bool       = False
    schema_version:      str        = "1.0"