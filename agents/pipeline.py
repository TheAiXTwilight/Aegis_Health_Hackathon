"""
agents/pipeline.py — AegisPipeline orchestrator.

Changes from original:
    - Uses tool name constants from tools.tool_names.
    - Real tool classes wired in place of stubs.
    - SymptomExtractor skip condition handles ToolError voice_result.
    - calculate_confidence() called unconditionally after ReportGenerator.
    - _run_step return type corrected to object | ToolError | None.
    - ToolFn type alias updated to Awaitable[object | ToolError].
    - SeverityScorer.score is async — satisfies _run_step contract.
    - Async generator return annotations corrected to AsyncGenerator.
"""

from __future__ import annotations

import time
from typing import AsyncGenerator, Awaitable, Callable, TypeVar

from loguru import logger

from schemas.errors import FatalPipelineError, ToolError
from schemas.state import AegisState
from schemas.xray import XRayResult
from tools.confidence import calculate_confidence
from tools.drug_checker import DrugInteractionChecker
from tools.lab_report_parser import LabReportParser
from tools.medical_rag_search import MedicalRAGSearch
from tools.report_generator import ReportGenerator
from tools.severity_scorer import SeverityScorer
from tools.symptom_extractor import SymptomExtractor
from tools.tool_names import (
    TOOL_DRUG_INTERACTION_CHECKER,
    TOOL_LAB_REPORT_PARSER,
    TOOL_MEDICAL_RAG_SEARCH,
    TOOL_REPORT_GENERATOR,
    TOOL_SEVERITY_SCORER,
    TOOL_SYMPTOM_EXTRACTOR,
    TOOL_VOICE_TRANSCRIBER,
    TOOL_XRAY_PROCESSOR,
)
from tools.voice_transcriber import VoiceTranscriber

T = TypeVar("T")

class AegisPipeline:
    """
    Sequential async orchestrator. One instance, application-lifetime.
    Steps 0–6 mutate state silently.
    Step 7 yields raw report tokens only.
    """

    def __init__(self) -> None:
        self._voice_transcriber        = VoiceTranscriber()
        self._symptom_extractor        = SymptomExtractor()
        self._lab_report_parser        = LabReportParser()
        self._medical_rag_search       = MedicalRAGSearch()
        self._drug_interaction_checker = DrugInteractionChecker()
        self._severity_scorer          = SeverityScorer()
        self._report_generator         = ReportGenerator()

    async def run(
        self, state: AegisState
    ) -> AsyncGenerator[str, None]:
        state.pipeline_start_ms = time.perf_counter() * 1000
        logger.info("Pipeline started", session_id=state.session_id)

        try:
            await self._run_voice_transcriber(state)
            await self._run_symptom_extractor(state)
            await self._run_lab_report_parser(state)
            await self._run_xray_processor(state)
            await self._run_medical_rag_search(state)
            await self._run_drug_interaction_checker(state)
            await self._run_severity_scorer(state)

            async for token in self._run_report_generator(state):
                yield token

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
            )

    # ── Central step helper ───────────────────────────────────────

    async def _run_step(
        self,
        name: str,
        tool_fn: Callable[[AegisState], Awaitable[T | ToolError]],
        state: AegisState,
    ) -> T | ToolError | None:
        """
        Run one tool, capturing timing, errors, and lifecycle.

        Returns:
            Tool result on success.
            ToolError on non-fatal failure.
            None on unhandled exception.

        Raises:
            FatalPipelineError when tool returns ToolError(fatal=True).
        """
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
            state.step_durations_ms[name] = (
                (time.perf_counter() - start) * 1000
            )
            state.current_tool = None

    # ── Step 0 — VoiceTranscriber ─────────────────────────────────

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

    # ── Step 1 — SymptomExtractor ─────────────────────────────────

    async def _run_symptom_extractor(self, state: AegisState) -> None:
        # Skip when no text is available AND voice transcription failed
        # or was not attempted. A ToolError voice_result means
        # raw_symptoms_text was not populated from audio.
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

    # ── Step 2 — LabReportParser ──────────────────────────────────

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

    # ── Step 3 — XRayProcessor (stub) ────────────────────────────

    async def _run_xray_processor(self, state: AegisState) -> None:
        if state.xray_image_path is None:
            return
        result = await self._run_step(
            TOOL_XRAY_PROCESSOR,
            _xray_processor_stub,
            state,
        )
        if result is not None:
            state.xray_result = result

    # ── Step 4 — MedicalRAGSearch ─────────────────────────────────

    async def _run_medical_rag_search(self, state: AegisState) -> None:
        result = await self._run_step(
            TOOL_MEDICAL_RAG_SEARCH,
            self._medical_rag_search.run,
            state,
        )
        if result is not None:
            state.rag_result = result

    # ── Step 5 — DrugInteractionChecker ──────────────────────────

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

    # ── Step 6 — SeverityScorer ───────────────────────────────────

    async def _run_severity_scorer(self, state: AegisState) -> None:
        result = await self._run_step(
            TOOL_SEVERITY_SCORER,
            self._severity_scorer.score,
            state,
        )
        if result is not None:
            state.severity_result = result

    # ── Step 7 — ReportGenerator ──────────────────────────────────

    async def _run_report_generator(
        self, state: AegisState
    ) -> AsyncGenerator[str, None]:
        state.current_tool = TOOL_REPORT_GENERATOR
        start = time.perf_counter()

        try:
            async for token in self._report_generator.run(state):
                yield token

            state.tools_run.append(TOOL_REPORT_GENERATOR)

            # Compute confidence unconditionally so it is always
            # available for diagnostics even if state.report failed.
            # Assign into report only if it was successfully created.
            confidence = calculate_confidence(state)
            if state.report is not None:
                state.report.confidence = confidence

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
                    tool=TOOL_REPORT_GENERATOR,
                    reason=str(exc),
                    fatal=True,
                )
            )

        finally:
            state.step_durations_ms[TOOL_REPORT_GENERATOR] = (
                (time.perf_counter() - start) * 1000
            )
            state.current_tool = None


# ── XRayProcessor stub ────────────────────────────────────────────
# Replaced by real XRayProcessor in Week 2.

async def _xray_processor_stub(state: AegisState) -> XRayResult | ToolError | None:
    return None