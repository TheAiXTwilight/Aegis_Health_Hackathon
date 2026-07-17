"""
tools/rule_validator.py — RuleValidator (Step 9).

Compares deterministic severity (SeverityScorer) against the severity
expressed in the LLM narrative (ReportGenerator output).

Extraction method:
    Whole-word regex (\bHIGH\b, \bMEDIUM\b, \bLOW\b) applied to the
    ### Severity section of state.report.text. Deterministic, no LLM.

Three-state output:
    agreement  — levels match
    warning    — minor mismatch or unextractable narrative level
    override   — deterministic=HIGH and narrative≠HIGH (safety-critical)

Override condition:
    deterministic_level == "HIGH" and slm_narrative_level != "HIGH"

    False reassurance is the safety risk: a clinician reading LOW or
    MEDIUM when critical conditions were detected by rules. The reverse
    (overclaiming HIGH when rules say LOW) is not a clinical safety
    hazard — it produces WARNING not OVERRIDE.

TriageReport.severity is already set from the deterministic result by
ReportGenerator. Override surfaces the conflict without rewriting the
report text. The UI shows a safety banner when result.overridden=True.

Interface: async def run(self, state) -> RuleValidatorResult | ToolError
Consistent with all other tools. Logic is synchronous.
"""

from __future__ import annotations

import re

from loguru import logger

from schemas.errors import ToolError
from schemas.state import AegisState
from schemas.validation import RuleValidatorResult, ValidationStatus
from tools.tool_names import TOOL_RULE_VALIDATOR


class RuleValidator:
    """
    Synchronous comparison logic wrapped in async interface.
    Does not call Ollama. Does not modify state.
    """

    TOOL_NAME = TOOL_RULE_VALIDATOR

    async def run(
        self,
        state: AegisState,
    ) -> RuleValidatorResult | ToolError:

        # ── Guard conditions ──────────────────────────────────────
        if state.report is None:
            return ToolError(
                tool   = TOOL_RULE_VALIDATOR,
                reason = "No report available to validate.",
                fatal  = False,
            )

        if (
            state.severity_result is None
            or isinstance(state.severity_result, ToolError)
        ):
            return ToolError(
                tool   = TOOL_RULE_VALIDATOR,
                reason = "No severity result available for comparison.",
                fatal  = False,
            )

        deterministic_level = state.severity_result.level
        narrative_level     = _extract_narrative_level(state.report.text)
        result              = _classify(deterministic_level, narrative_level)

        logger.info(
            "rule_validator · complete",
            session_id=getattr(state, "session_id", None),
            status=result.status.value,
            deterministic_level=deterministic_level,
            slm_narrative_level=narrative_level,
            overridden=result.overridden,
        )

        return result


# ── Narrative level extraction ────────────────────────────────────

def _extract_narrative_level(report_text: str) -> str | None:
    """
    Extract severity level from the ### Severity section.

    Algorithm:
        1. Find ### Severity header.
        2. Extract text from that header to the next ### header.
        3. Search HIGH, MEDIUM, LOW as whole words in priority order.
        4. First match wins.

    Word boundary matching (\b) prevents:
        "highest priority" matching HIGH
        "medium-term"      matching MEDIUM
        "low-level"        matching LOW

    Returns "HIGH", "MEDIUM", "LOW", or None.
    Extraction is case-sensitive (uppercase only) matching the
    required section content format from ReportGenerator.
    """
    start = report_text.find("### Severity")
    if start == -1:
        return None

    end     = report_text.find("###", start + len("### Severity"))
    section = (
        report_text[start:end]
        if end != -1
        else report_text[start:]
    )

    for level in ("HIGH", "MEDIUM", "LOW"):
        if re.search(rf"\b{level}\b", section):
            return level

    return None


# ── Classification logic ──────────────────────────────────────────

def _classify(
    deterministic: str,
    narrative: str | None,
) -> RuleValidatorResult:
    """
    Apply three-state classification.

    Override only when: deterministic=HIGH and narrative≠HIGH.
    All other mismatches and extraction failures → WARNING.
    """
    if narrative is None:
        return RuleValidatorResult(
            status              = ValidationStatus.WARNING,
            deterministic_level = deterministic,  # type: ignore[arg-type]
            slm_narrative_level = None,
            disagreement_reason = (
                "Unable to extract severity level from narrative. "
                "The ### Severity section may be missing or malformed."
            ),
            overridden          = False,
        )

    if narrative == deterministic:
        return RuleValidatorResult(
            status              = ValidationStatus.AGREEMENT,
            deterministic_level = deterministic,  # type: ignore[arg-type]
            slm_narrative_level = narrative,
            disagreement_reason = None,
            overridden          = False,
        )

    # Safety-critical: rules say HIGH, narrative does not
    if deterministic == "HIGH" and narrative != "HIGH":
        return RuleValidatorResult(
            status              = ValidationStatus.OVERRIDE,
            deterministic_level = "HIGH",
            slm_narrative_level = narrative,
            disagreement_reason = (
                f"Deterministic rules require HIGH severity; "
                f"narrative expresses {narrative}. "
                "Deterministic result is authoritative."
            ),
            overridden          = True,
        )

    # Non-safety-critical mismatch
    if deterministic == "LOW" and narrative == "HIGH":
        reason = (
            "Narrative overclaims HIGH severity; "
            "deterministic rules produced LOW."
        )
    else:
        reason = (
            f"Minor disagreement: "
            f"deterministic={deterministic}, narrative={narrative}."
        )

    return RuleValidatorResult(
        status              = ValidationStatus.WARNING,
        deterministic_level = deterministic,  # type: ignore[arg-type]
        slm_narrative_level = narrative,
        disagreement_reason = reason,
        overridden          = False,
    )