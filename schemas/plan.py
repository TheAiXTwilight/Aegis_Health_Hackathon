"""
schemas/plan.py — ExecutionPlan schema.

The ExecutionPlan is produced by ExecutionPlanner (Step -1) and
validated/repaired by PlanValidator before any clinical tool runs.

Design principle — Planner Authority Invariant:
    The ExecutionPlanner may choose ONLY optional enrichment capabilities.
    Mandatory input-driven tools (VoiceTranscriber, SymptomExtractor,
    LabReportParser, XRayProcessor, DrugInteractionChecker) are
    determined entirely by input presence. They are pipeline invariants,
    not planner decisions. The plan contains no fields for them.

    This schema contains exactly one tool boolean: use_rag.
    That is the only enrichment tool the planner controls today.
    Future optional enrichment tools would add fields here.

Field semantics:

    use_rag
        The planner's single optional enrichment decision.
        PlanValidator may override to True when safety signals are
        present (see tools/planner_constants.py), but respects False
        when no safety signals exist. This is the planner's genuine
        authority.

    reasoning
        Audit metadata only. It is never interpreted programmatically
        and never affects pipeline execution. May contain inaccurate
        or hallucinated content from the 1B planner. Useful for
        debugging, audit trails, and examiner review.

    was_repaired
        True when PlanValidator changed use_rag from False to True
        due to safety signals. The plan is always executable after
        PlanValidator runs — was_repaired is a quality signal about
        the planner, not a gate on execution.

    is_fallback
        True when the planner failed entirely (ToolError or unhandled
        exception after all retries) and _make_fallback_plan() was used.
        No planner output existed.

    validation_errors
        Human-readable reasons for each repair PlanValidator made.
        Empty when was_repaired=False.

Invariant (enforced by model_validator):
    is_fallback and was_repaired cannot both be True.
    A fallback plan has no planner output to attribute a repair to.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ExecutionPlan(BaseModel):
    """
    Structured execution plan produced by ExecutionPlanner and
    validated/repaired by PlanValidator.

    Always executable after PlanValidator runs.
    Stored on AegisState.execution_plan.
    Summarised in TriageReport.execution_plan_summary.
    """

    schema_version: int = 1

    # ── Optional enrichment decision (planner authority) ──────────
    use_rag: bool

    # ── Audit metadata ────────────────────────────────────────────
    reasoning: str = Field(
        description=(
            "Audit metadata only. It is never interpreted programmatically "
            "and never affects pipeline execution. May contain inaccurate "
            "or hallucinated content from the 1B planner."
        )
    )

    # ── Plan provenance ───────────────────────────────────────────
    was_repaired:      bool      = False
    validation_errors: list[str] = Field(default_factory=list)
    is_fallback:       bool      = False

    # ── Invariant ─────────────────────────────────────────────────

    @model_validator(mode="after")
    def _check_fallback_repair_invariant(self) -> "ExecutionPlan":
        """
        is_fallback and was_repaired cannot both be True.

        is_fallback=True means no planner output existed.
        was_repaired=True means planner output existed but was corrected.
        Both True simultaneously indicates a construction bug.

        Defense-in-depth: _make_fallback_plan() hardcodes was_repaired=False.
        PlanValidator only sets was_repaired=True on non-fallback plans.
        This validator catches any future regression.
        """
        if self.is_fallback and self.was_repaired:
            raise ValueError(
                "ExecutionPlan invariant violated: "
                "is_fallback and was_repaired cannot both be True. "
                "A fallback plan has no planner output to repair."
            )
        return self