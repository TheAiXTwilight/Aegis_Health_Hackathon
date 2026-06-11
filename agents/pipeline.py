"""
agents/pipeline.py — AegisPipeline orchestrator.

Sequential, deterministic, async. Wall-clock bounded by the worker.

Topology:
    Step 0 — VoiceTranscriber       (optional, skipped if no audio)
    Step 1 — SymptomExtractor
    Step 2 — LabReportParser
    Step 3 — XRayProcessor
    Step 4 — MedicalRAGSearch
    Step 5 — DrugInteractionChecker
    Step 6 — SeverityScorer
    Step 7 — ReportGenerator        (only step that yields tokens)

Streaming protocol
──────────────────
GET /queue/stream/{job_id}  — yields raw report tokens only (plain text,
    no framing). Consumed by st.write_stream. Nothing else goes into this
    stream — no events, no status, no protocol markers.

GET /queue/status/{job_id}  — returns job status plus live pipeline state
    for the sidebar: current_tool, tools_run, tools_failed,
    step_durations_ms. Frontend polls every 1–2 seconds. No stream parsing
    required.

These two endpoints serve two distinct concerns and are never mixed.

pipeline_complete and pipeline_end_ms are set in finally so they record
even for failed runs. pipeline_complete means "pipeline finished"
(success or failure) — success is indicated by state.report being set.
"""
from __future__ import annotations

import time
from typing import AsyncIterator, Awaitable, Callable

from loguru import logger

from schemas.errors import FatalPipelineError, ToolError
from schemas.state import AegisState
from tools.report_generator import ReportGenerator


ToolFn = Callable[[AegisState], Awaitable[object]]


class AegisPipeline:
    """
    Sequential async orchestrator. One instance, application-lifetime.

    Each AegisState flows through Steps 0-7 in order. Steps 0-6 mutate
    state silently — progress is observable via AegisState fields
    (current_tool, tools_run, tools_failed, step_durations_ms) which
    GET /queue/status polls. Step 7 yields raw report tokens only.
    """

    def __init__(self) -> None:
        self._report_generator = ReportGenerator()

    async def run(self, state: AegisState) -> AsyncIterator[str]:
        """
        Execute the full pipeline against the given state.

        Yields raw string tokens from Step 7 (ReportGenerator) only.
        All other steps mutate state in place and return nothing to the
        caller — progress is readable from state fields.

        On normal completion:
            state.report is set, state.pipeline_complete is True.

        On fatal failure:
            FatalPipelineError propagates to backend/queue.py.
            state.pipeline_complete is True, state.report may be None.

        On infrastructure timeout:
            asyncio.TimeoutError propagates; state.pipeline_complete is True.
        """
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
            state.current_tool       = None
            state.pipeline_end_ms    = time.perf_counter() * 1000
            state.pipeline_complete  = True

            logger.info(
                "Pipeline finished",
                session_id=state.session_id,
                duration_ms=state.pipeline_end_ms - state.pipeline_start_ms,
                tools_run=state.tools_run,
                tools_failed=state.tools_failed,
                has_report=state.report is not None,
            )

    # ── Central step execution helper ─────────────────────────────

    async def _run_step(
        self,
        name: str,
        tool_fn: ToolFn,
        state: AegisState,
    ) -> object | None:
        """
        Run one tool, capturing timing, errors, and lifecycle.

        Sets state.current_tool for the duration so /queue/status can
        report which tool is active without any stream-side signalling.

        Returns:
            The tool's result on success, or
            ToolError(fatal=False) on non-fatal failure, or
            None when the tool raised an unhandled exception.

        Raises:
            FatalPipelineError when the tool returns ToolError(fatal=True).

        State mutations:
            - Appends name to tools_run on success.
            - Appends name to tools_failed on any failure.
            - Records step_durations_ms[name] always.
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
                        tool=result.tool,
                    )
                    raise FatalPipelineError(result)
                logger.info(
                    "Non-fatal tool error",
                    step=name,
                    reason=result.reason,
                    tool=result.tool,
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

    # ── Step 0 — VoiceTranscriber (optional) ──────────────────────

    async def _run_voice_transcriber(self, state: AegisState) -> None:
        if state.audio_file_path is None:
            return
        result = await self._run_step(
            "VoiceTranscriber", _voice_transcriber_stub, state
        )
        if result is not None and not isinstance(result, ToolError):
            state.voice_result = result
        elif isinstance(result, ToolError):
            state.voice_result = result

    # ── Step 1 — SymptomExtractor ─────────────────────────────────

    async def _run_symptom_extractor(self, state: AegisState) -> None:
        if state.raw_symptoms_text is None and state.voice_result is None:
            return
        result = await self._run_step(
            "SymptomExtractor", _symptom_extractor_stub, state
        )
        if result is not None and not isinstance(result, ToolError):
            state.symptom_result = result
        elif isinstance(result, ToolError):
            state.symptom_result = result

    # ── Step 2 — LabReportParser ──────────────────────────────────

    async def _run_lab_report_parser(self, state: AegisState) -> None:
        if state.lab_pdf_path is None:
            return
        result = await self._run_step(
            "LabReportParser", _lab_report_parser_stub, state
        )
        if result is not None and not isinstance(result, ToolError):
            state.lab_result = result
        elif isinstance(result, ToolError):
            state.lab_result = result

    # ── Step 3 — XRayProcessor ────────────────────────────────────

    async def _run_xray_processor(self, state: AegisState) -> None:
        if state.xray_image_path is None:
            return
        result = await self._run_step(
            "XRayProcessor", _xray_processor_stub, state
        )
        if result is not None and not isinstance(result, ToolError):
            state.xray_result = result
        elif isinstance(result, ToolError):
            state.xray_result = result

    # ── Step 4 — MedicalRAGSearch ─────────────────────────────────

    async def _run_medical_rag_search(self, state: AegisState) -> None:
        result = await self._run_step(
            "MedicalRAGSearch", _medical_rag_search_stub, state
        )
        if result is not None and not isinstance(result, ToolError):
            state.rag_result = result
        elif isinstance(result, ToolError):
            state.rag_result = result

    # ── Step 5 — DrugInteractionChecker ───────────────────────────

    async def _run_drug_interaction_checker(self, state: AegisState) -> None:
        if not state.medications_raw:
            return
        result = await self._run_step(
            "DrugInteractionChecker", _drug_interaction_checker_stub, state
        )
        if result is not None and not isinstance(result, ToolError):
            state.drug_result = result
        elif isinstance(result, ToolError):
            state.drug_result = result

    # ── Step 6 — SeverityScorer ───────────────────────────────────

    async def _run_severity_scorer(self, state: AegisState) -> None:
        result = await self._run_step(
            "SeverityScorer", _severity_scorer_stub, state
        )
        if result is not None and not isinstance(result, ToolError):
            state.severity_result = result
        elif isinstance(result, ToolError):
            state.severity_result = result

    # ── Step 7 — ReportGenerator ──────────────────────────────────

    async def _run_report_generator(
        self, state: AegisState
    ) -> AsyncIterator[str]:
        """
        Stream raw report tokens and write state.report on completion.

        Delegates to ReportGenerator.run() which is an async generator:
            - yields plain string tokens (no framing)
            - writes state.report once the stream is exhausted
            - raises FatalPipelineError on any failure

        tools_run / tools_failed and step timing are managed here,
        not inside ReportGenerator, consistent with all other steps.
        """
        state.current_tool = "ReportGenerator"
        start = time.perf_counter()

        try:
            async for token in self._report_generator.run(state):
                yield token

            state.tools_run.append("ReportGenerator")

        except FatalPipelineError:
            state.tools_failed.append("ReportGenerator")
            raise

        except Exception as exc:
            state.tools_failed.append("ReportGenerator")
            logger.exception(
                "ReportGenerator failed",
                session_id=state.session_id,
            )
            raise FatalPipelineError(
                ToolError(
                    tool="ReportGenerator",
                    reason=str(exc),
                    fatal=True,
                )
            )

        finally:
            state.step_durations_ms["ReportGenerator"] = (
                (time.perf_counter() - start) * 1000
            )
            state.current_tool = None


# ── Tool stubs ────────────────────────────────────────────────────
# Placeholders replaced by real tool imports as they land in tools/*.
# The orchestration above does not change when stubs are replaced.

async def _voice_transcriber_stub(state: AegisState) -> object:
    return None


async def _symptom_extractor_stub(state: AegisState) -> object:
    return None


async def _lab_report_parser_stub(state: AegisState) -> object:
    return None


async def _xray_processor_stub(state: AegisState) -> object:
    return None


async def _medical_rag_search_stub(state: AegisState) -> object:
    return None


async def _drug_interaction_checker_stub(state: AegisState) -> object:
    return None


async def _severity_scorer_stub(state: AegisState) -> object:
    return None