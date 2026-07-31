"""
agents/pipeline.py — AegisPipeline orchestrator.

Phase 2.5 changes from Phase 2:
    Step -1  ExecutionPlanner added — decides use_rag only
    Step  0  PlanValidator called synchronously inside _run_execution_planner
    Step  5  MedicalRAGSearch gated on state.execution_plan.use_rag
    Step  9  RuleValidator added — compares deterministic vs narrative severity

Mandatory tool gates (Steps 1–4, 6) are driven by input presence only.
No plan involvement. This implements the Planner Authority Invariant:
    The planner controls only optional enrichment (use_rag).
    Mandatory tools run whenever their input exists.

Pipeline invariant (Phase 2.5):
    After _run_execution_planner(state), state.execution_plan is always
    a validated ExecutionPlan. No downstream method branches on planner
    success vs fallback — all code reads state.execution_plan only.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import AsyncGenerator, Awaitable, Callable, TypeVar
from loguru import logger

from schemas.errors import FatalPipelineError, ToolError
from schemas.plan import ExecutionPlan
from schemas.state import AegisState
from schemas.validation import RuleValidatorResult
from tools.confidence import calculate_confidence
from tools.drug_checker import DrugInteractionChecker
from tools.execution_planner import ExecutionPlanner, _make_fallback_plan
from tools.lab_report_parser import LabReportParser
from tools.medical_rag_search import MedicalRAGSearch
from tools.plan_validator import PlanValidator
from tools.report_generator import ReportGenerator
from tools.rule_validator import RuleValidator
from tools.severity_scorer import SeverityScorer
from tools.symptom_extractor import SymptomExtractor
from tools.tool_names import (
    TOOL_DRUG_INTERACTION_CHECKER,
    TOOL_EXECUTION_PLANNER,
    TOOL_LAB_REPORT_PARSER,
    TOOL_MEDICAL_RAG_SEARCH,
    TOOL_REPORT_GENERATOR,
    TOOL_RULE_VALIDATOR,
    TOOL_SEVERITY_SCORER,
    TOOL_SYMPTOM_EXTRACTOR,
    TOOL_VOICE_TRANSCRIBER,
    TOOL_XRAY_PROCESSOR,
)
from tools.voice_transcriber import VoiceTranscriber
from vision.xray_processor import XRayProcessor

T = TypeVar("T")
UI_MIN_STEP_SECONDS = float(os.getenv("AEGIS_UI_STEP_SECONDS", "0.25"))


# ── Plan summary builder ──────────────────────────────────────────

def _build_plan_summary(plan: ExecutionPlan, state: AegisState) -> str:
    """
    Build a comprehensive human-readable execution plan summary.

    Two sections:
        Mandatory tools — derived from input presence (not planner)
        Optional tools  — derived from ExecutionPlan (planner decision)

    Suffix list uses accumulation rather than elif so the formatter
    remains correct if the mutual-exclusivity invariant ever relaxes.

    Format:
        "Mandatory: ✓ ToolA ✗ ToolB | Optional: ✓ ToolC [FALLBACK] | reasoning"
    """
    mandatory: list[tuple[str, bool]] = [
        (
            "VoiceTranscriber",
            bool(state.audio_file_path),
        ),
        (
            "SymptomExtractor",
            bool(state.raw_symptoms_text) or bool(state.audio_file_path),
        ),
        (
            "LabReportParser",
            bool(state.lab_pdf_path),
        ),
        (
            "XRayProcessor",
            bool(state.xray_image_path)
            or bool(state.xray_findings_raw)
            or bool(state.xray_free_text_raw),
        ),
        (
            "DrugInteractionChecker",
            bool(state.medications_raw),
        ),
    ]

    optional: list[tuple[str, bool]] = [
        ("MedicalRAGSearch", plan.use_rag),
    ]

    def _marks(tools: list[tuple[str, bool]]) -> str:
        return " | ".join(
            f"{'✓' if enabled else '✗'} {name}"
            for name, enabled in tools
        )

    # Future-proof suffix: accumulate rather than elif
    suffixes: list[str] = []
    if plan.is_fallback:
        suffixes.append("[FALLBACK]")
    if plan.was_repaired:
        suffixes.append("[REPAIRED]")
    suffix = " " + " ".join(suffixes) if suffixes else ""

    return (
        f"Mandatory: {_marks(mandatory)} | "
        f"Optional: {_marks(optional)}"
        f"{suffix} | {plan.reasoning}"
    )


# ── Pipeline ──────────────────────────────────────────────────────

class AegisPipeline:
    """
    Sequential async orchestrator. One instance, application-lifetime.

    Step -1: ExecutionPlanner  (always — fallback on failure)
    Step  0: PlanValidator     (always — called inside _run_execution_planner)
    Step  1: VoiceTranscriber  (if audio_file_path)
    Step  2: SymptomExtractor  (if symptoms text or voice available)
    Step  3: LabReportParser   (if lab_pdf_path)
    Step  4: XRayProcessor     (if xray input)
    Step  5: MedicalRAGSearch  (if state.execution_plan.use_rag)
    Step  6: DrugInteractionChecker (if medications_raw)
    Step  7: SeverityScorer    (always)
    Step  8: ReportGenerator   (always — yields tokens)
    Step  9: RuleValidator     (always — guard: state.report is not None)
    """

    def __init__(self) -> None:
        self._execution_planner        = ExecutionPlanner()
        self._plan_validator           = PlanValidator()
        self._voice_transcriber        = VoiceTranscriber()
        self._symptom_extractor        = SymptomExtractor()
        self._lab_report_parser        = LabReportParser()
        self._medical_rag_search       = MedicalRAGSearch()
        self._drug_interaction_checker = DrugInteractionChecker()
        self._severity_scorer          = SeverityScorer()
        self._report_generator         = ReportGenerator()
        self._rule_validator           = RuleValidator()
        self._xray_processor           = XRayProcessor()

    async def run(
        self, state: AegisState
    ) -> AsyncGenerator[str, None]:
        state.pipeline_start_ms = time.perf_counter() * 1000
        logger.info("Pipeline started", session_id=state.session_id)

        try:
            await self._run_execution_planner(state)
            await self._run_voice_transcriber(state)
            await self._run_symptom_extractor(state)
            await self._run_lab_report_parser(state)
            await self._run_xray_processor(state)
            await self._run_medical_rag_search(state)
            await self._run_drug_interaction_checker(state)
            await self._run_severity_scorer(state)

            async for token in self._run_report_generator(state):
                yield token

            await self._run_rule_validator(state)

            # Confidence is computed AFTER rule validation so the
            # narrative-vs-rules agreement signal is available.
            # See the confidence module (varies 0.5-0.97, never flat 1.0).
            if state.report is not None:
                state.report.confidence = calculate_confidence(state)

        finally:
            state.current_tool      = None
            state.pipeline_end_ms   = time.perf_counter() * 1000
            state.pipeline_complete = True

            logger.info(
                "Pipeline finished",
                session_id=state.session_id,
                duration_ms=state.pipeline_end_ms - state.pipeline_start_ms,
                tools_run=state.tools_run,
                tools_failed=state.tools_failed,
                has_report=state.report is not None,
                has_plan=state.execution_plan is not None,
                validation_status=(
                    state.rule_validator_result.status.value
                    if state.rule_validator_result is not None
                    else None
                ),
            )

    # ── Central step helper ───────────────────────────────────────

    async def _run_step(
        self,
        name: str,
        tool_fn: Callable[[AegisState], Awaitable[T | ToolError]],
        state: AegisState,
    ) -> T | ToolError | None:
        state.current_tool = name
        start = time.perf_counter()

        try:
            result = await tool_fn(state)

            if isinstance(result, ToolError):
                state.tools_failed.append(name)
                if result.fatal:
                    logger.warning(
                        "Fatal pipeline failure",
                        step=name,
                        reason=result.reason,
                    )
                    raise FatalPipelineError(result)
                logger.info(
                    "Non-fatal tool error",
                    step=name,
                    reason=result.reason,
                )
                return result

            state.tools_run.append(name)
            return result

        except FatalPipelineError:
            raise

        except Exception:
            state.tools_failed.append(name)
            logger.exception(
                "Unhandled exception in step",
                step=name,
                session_id=state.session_id,
            )
            return None

        finally:
            elapsed_seconds = time.perf_counter() - start

            state.step_durations_ms[name] = elapsed_seconds * 1000

            remaining_seconds = UI_MIN_STEP_SECONDS - elapsed_seconds

            if remaining_seconds > 0:
                await asyncio.sleep(remaining_seconds)

            state.current_tool = None

    # ── Step -1 — ExecutionPlanner + PlanValidator ────────────────

    async def _run_execution_planner(self, state: AegisState) -> None:
        """
        Single normalisation point (Decision 51).

        Guarantees: state.execution_plan is a validated ExecutionPlan
        after this method returns, regardless of planner success/failure.

        Flow:
            planner result (ExecutionPlan | ToolError | None)
                ↓
            _make_fallback_plan(state) if not ExecutionPlan
                ↓
            PlanValidator.validate()
                ↓
            state.execution_plan
        """
        result = await self._run_step(
            TOOL_EXECUTION_PLANNER,
            self._execution_planner.run,
            state,
        )

        raw_plan = (
            result
            if isinstance(result, ExecutionPlan)
            else _make_fallback_plan(state)
        )

        state.execution_plan = self._plan_validator.validate(raw_plan, state)

        logger.info(
            "execution_planner · plan set",
            session_id=state.session_id,
            use_rag=state.execution_plan.use_rag,
            is_fallback=state.execution_plan.is_fallback,
            was_repaired=state.execution_plan.was_repaired,
        )

    # ── Step 1 — VoiceTranscriber ─────────────────────────────────

    async def _run_voice_transcriber(self, state: AegisState) -> None:
        if state.audio_file_path is None:
            return

        result = await self._run_step(
            TOOL_VOICE_TRANSCRIBER,
            self._voice_transcriber.run,
            state,
        )
        if result is not None:
            state.voice_result = result

    # ── Step 2 — SymptomExtractor ─────────────────────────────────

    async def _run_symptom_extractor(self, state: AegisState) -> None:
        if state.raw_symptoms_text is None and (
            state.voice_result is None
            or isinstance(state.voice_result, ToolError)
        ):
            return
        result = await self._run_step(
            TOOL_SYMPTOM_EXTRACTOR,
            self._symptom_extractor.run,
            state,
        )
        if result is not None:
            state.symptom_result = result

    # ── Step 3 — LabReportParser ──────────────────────────────────

    async def _run_lab_report_parser(self, state: AegisState) -> None:
        if state.lab_pdf_path is None:
            return
        result = await self._run_step(
            TOOL_LAB_REPORT_PARSER,
            self._lab_report_parser.run,
            state,
        )
        if result is not None:
            state.lab_result = result

    # ── Step 4 — XRayProcessor ────────────────────────────

    async def _run_xray_processor(self, state: AegisState) -> None:
        if state.xray_image_path is None:
            return
        result = await self._run_step(
            TOOL_XRAY_PROCESSOR,
            self._xray_processor.run,
            state,
        )
        if result is not None:
            state.xray_result = result

    # ── Step 5 — MedicalRAGSearch ─────────────────────────────────

    async def _run_medical_rag_search(self, state: AegisState) -> None:
        # Only optional tool gated on the execution plan
        if not state.execution_plan or not state.execution_plan.use_rag:
            return
        result = await self._run_step(
            TOOL_MEDICAL_RAG_SEARCH,
            self._medical_rag_search.run,
            state,
        )
        if result is not None:
            state.rag_result = result

    # ── Step 6 — DrugInteractionChecker ──────────────────────────

    async def _run_drug_interaction_checker(self, state: AegisState) -> None:
        if not state.medications_raw:
            return
        result = await self._run_step(
            TOOL_DRUG_INTERACTION_CHECKER,
            self._drug_interaction_checker.run,
            state,
        )
        if result is not None:
            state.drug_result = result

    # ── Step 7 — SeverityScorer ───────────────────────────────────

    async def _run_severity_scorer(self, state: AegisState) -> None:
        result = await self._run_step(
            TOOL_SEVERITY_SCORER,
            self._severity_scorer.score,
            state,
        )
        if result is not None:
            state.severity_result = result

    # ── Step 8 — ReportGenerator ──────────────────────────────────

    async def _run_report_generator(
        self, state: AegisState
    ) -> AsyncGenerator[str, None]:
        state.current_tool = TOOL_REPORT_GENERATOR
        start = time.perf_counter()

        try:
            async for token in self._report_generator.run(state):
                yield token

            state.tools_run.append(TOOL_REPORT_GENERATOR)

            # Write execution plan summary now that state.report exists
            if state.report is not None and state.execution_plan is not None:
                state.report.execution_plan_summary = _build_plan_summary(
                    state.execution_plan, state
                )

        except FatalPipelineError:
            state.tools_failed.append(TOOL_REPORT_GENERATOR)
            raise

        except Exception as exc:
            state.tools_failed.append(TOOL_REPORT_GENERATOR)
            logger.exception(
                "ReportGenerator failed",
                session_id=state.session_id,
            )
            raise FatalPipelineError(
                ToolError(
                    tool   = TOOL_REPORT_GENERATOR,
                    reason = str(exc),
                    fatal  = True,
                )
            )

        finally:
            elapsed_seconds = time.perf_counter() - start

            state.step_durations_ms[TOOL_REPORT_GENERATOR] = elapsed_seconds * 1000

            remaining_seconds = UI_MIN_STEP_SECONDS - elapsed_seconds

            if remaining_seconds > 0:
                await asyncio.sleep(remaining_seconds)

            state.current_tool = None

    # ── Step 9 — RuleValidator ────────────────────────────────────

    async def _run_rule_validator(self, state: AegisState) -> None:
        """
        Compare deterministic severity against LLM narrative severity.

        Writes to state.rule_validator_result.
        Writes result.status.value to state.report.validation_status.
        Non-fatal on ToolError — report remains valid without validation.
        """
        result = await self._run_step(
            TOOL_RULE_VALIDATOR,
            self._rule_validator.run,
            state,
        )
        if isinstance(result, RuleValidatorResult):
            state.rule_validator_result = result
            if state.report is not None:
                state.report.validation_status = result.status.value
