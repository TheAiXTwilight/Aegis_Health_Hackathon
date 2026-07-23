"""
schemas/report.py — TriageReport schema.

Phase 2.5 additions:
    execution_plan_summary  — human-readable plan artifact
    validation_status       — RuleValidator three-state outcome string

Both default to None and are populated by the pipeline after their
respective steps complete.

execution_plan_summary
    Written by _run_report_generator after state.report is assigned.
    Built by _build_plan_summary(plan, state).
    Format: mandatory tools from input state + optional tools from plan.
    None only if execution_plan is somehow absent (cannot occur in
    normal pipeline flow).

validation_status
    Written by _run_rule_validator after RuleValidatorResult obtained.
    String value of ValidationStatus: "agreement", "warning", "override".
    None when RuleValidator did not run or returned ToolError.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TriageReport(BaseModel):
    """
    Final output of the Aegis pipeline (Step 8 — ReportGenerator).

    Pure data contract. All business logic lives in tools/.
    confidence is set to 0.0 by ReportGenerator as a placeholder and
    injected by AegisPipeline via calculate_confidence() after streaming.
    """

    severity: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        description=(
            "Matches the level from SeverityResult. Always deterministic."
        )
    )

    confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Pipeline confidence score. Set to 0.0 by ReportGenerator "
            "as placeholder. Injected by AegisPipeline via "
            "calculate_confidence() after ReportGenerator completes."
        ),
    )

    text: str = Field(
        description=(
            "Generated markdown report containing all six required sections."
        )
    )

    citations: list[str] = Field(
        default_factory=list,
        description="MedlinePlus citations used in the Evidence section.",
    )

    disclaimer: str = Field(
        description="Standard medical disclaimer appended to the report."
    )

    knowledge_base_version: str | None = Field(
        default=None,
        description=(
            "Git commit hash or version identifier of the RAG corpus."
        ),
    )

    knowledge_base_date: str | None = Field(
        default=None,
        description="Snapshot date of the RAG corpus.",
    )

    # ── Phase 2.5 additions ───────────────────────────────────────

    execution_plan_summary: str | None = Field(
        default=None,
        description=(
            "Human-readable execution plan summary. Reports mandatory tools "
            "from input state and optional tools from ExecutionPlan. "
            "Format: 'Mandatory: ✓ Tool ... | Optional: ✓ Tool ... "
            "[FALLBACK] [REPAIRED] | reasoning'. "
            "Written by AegisPipeline after ReportGenerator completes."
        ),
    )

    validation_status: str | None = Field(
        default=None,
        description=(
            "RuleValidator outcome: 'agreement', 'warning', or 'override'. "
            "None when RuleValidator did not run or returned ToolError."
        ),
    )

    schema_version: str = "1.0"