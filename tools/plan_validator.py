"""
tools/plan_validator.py — PlanValidator: RAG safety-override enforcement.

Validates and optionally repairs the use_rag field of an ExecutionPlan.
This is the only field PlanValidator controls — mandatory tool execution
is determined entirely by input presence in the pipeline.

Single responsibility:
    Force use_rag=True when safety signals indicate evidence retrieval
    should run regardless of the planner's decision.

NOT a pipeline tool. No TOOL_NAME. No _run_step integration.
Called synchronously inside _run_execution_planner.
Always returns a valid ExecutionPlan. Never raises. Never returns ToolError.

Mechanism vs configuration:
    This module owns the repair mechanism (when and how to repair).
    tools/planner_constants.py owns the configuration (what triggers repair).
    Changing clinical thresholds requires editing planner_constants.py only.
"""

from __future__ import annotations

from loguru import logger

from schemas.plan import ExecutionPlan
from schemas.state import AegisState
from tools.planner_constants import (
    RAG_FORCE_POLYPHARMACY_THRESHOLD,
    RAG_FORCE_SYMPTOM_TERMS,
    RAG_FORCE_XRAY_FINDINGS,
)


class PlanValidator:
    """
    Synchronous single-responsibility validator.

    validate() checks whether the planner's use_rag decision should
    be overridden by safety policy, repairs it if so, and returns a
    fresh ExecutionPlan. The input plan is never mutated.
    """

    def validate(
        self,
        raw_plan: ExecutionPlan,
        state: AegisState,
    ) -> ExecutionPlan:
        """
        Apply RAG safety override and return a corrected ExecutionPlan.

        If the planner said use_rag=False but safety signals are present,
        force use_rag=True and record the repair.

        Preserves is_fallback from the raw plan unchanged.
        Sets was_repaired=True only when use_rag was changed.
        Returns a fresh ExecutionPlan instance — never mutates input.
        """
        use_rag  = raw_plan.use_rag
        errors   = list(raw_plan.validation_errors)
        repaired = False

        if not use_rag and self._rag_should_be_forced(state):
            use_rag  = True
            repaired = True
            errors.append(
                "use_rag forced True: "
                "critical clinical indicators detected in input."
            )

        validated = ExecutionPlan(
            use_rag           = use_rag,
            reasoning         = raw_plan.reasoning,
            was_repaired      = repaired,
            validation_errors = errors,
            is_fallback       = raw_plan.is_fallback,
        )

        if repaired:
            logger.info(
                "plan_validator · use_rag repaired to True",
                session_id=getattr(state, "session_id", None),
                is_fallback=raw_plan.is_fallback,
            )
        else:
            logger.debug(
                "plan_validator · plan accepted without repair",
                session_id=getattr(state, "session_id", None),
                use_rag=use_rag,
            )

        return validated

    def _rag_should_be_forced(self, state: AegisState) -> bool:
        """
        Return True when any safety signal indicates RAG should run.

        Three independent conditions from planner_constants.py.
        Any one condition is sufficient.
        """
        # Condition 1: critical symptom terms in raw text
        text = (state.raw_symptoms_text or "").lower()
        if any(term in text for term in RAG_FORCE_SYMPTOM_TERMS):
            return True

        # Condition 2: critical X-ray findings
        findings_lower = [f.lower() for f in state.xray_findings_raw]
        if any(
            term in finding
            for term in RAG_FORCE_XRAY_FINDINGS
            for finding in findings_lower
        ):
            return True

        # Condition 3: polypharmacy threshold exceeded
        if len(state.medications_raw) > RAG_FORCE_POLYPHARMACY_THRESHOLD:
            return True

        return False