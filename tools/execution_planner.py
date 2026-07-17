"""
tools/execution_planner.py — ExecutionPlanner (Step -1).

SLM-powered planning step. Runs before all clinical tools.
Reads input metadata and a short symptom preview (≤200 chars).
Emits an ExecutionPlan with a single tool decision: use_rag.

The planner decides ONLY optional enrichment. Mandatory tools
(VoiceTranscriber, SymptomExtractor, LabReportParser, XRayProcessor,
DrugInteractionChecker) are determined by input presence in the
pipeline — this tool has no authority over them.

Retry policy:
    Attempt 1: full metadata prompt → json.loads() → ExecutionPlan
    Attempt 2: simplified prompt → same parse/validate
    Attempt 3: return ToolError(fatal=False)

The pipeline (_run_execution_planner) handles fallback construction.
This tool only attempts LLM-based planning.

Interface: async def run(self, state) -> ExecutionPlan | ToolError
Consistent with all other tools. Never raises.

Ollama request:
    Non-streaming (stream=False).
    Temperature 0.0 — routing decision, not creative output.
    num_predict 128 — plan JSON is two fields, small budget suffices.

Fallback rationale (Decision 50):
    _make_fallback_plan() sets use_rag=True as a safety-first policy.
    When planner reasoning is unavailable, evidence retrieval defaults
    to enabled. Retrieving unnecessary evidence is acceptable;
    omitting evidence in a planner-failure scenario is not.
"""

from __future__ import annotations

import json
import os

import httpx
from loguru import logger

from schemas.errors import ToolError
from schemas.plan import ExecutionPlan
from schemas.state import AegisState
from tools.tool_names import TOOL_EXECUTION_PLANNER


# ── Ollama configuration ──────────────────────────────────────────

OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_STREAM_URL = OLLAMA_BASE_URL + "/api/generate"
MODEL_TAG         = "aegis-llama"

_SYMPTOM_PREVIEW_CHARS = 200
_NUM_PREDICT           = 128
_NUM_CTX               = 4096


# ── Prompt templates ──────────────────────────────────────────────

_PROMPT_ATTEMPT_1 = """\
You are a routing assistant. Decide whether to retrieve medical evidence \
for a clinical triage session.

Input summary:
- symptoms_present: {symptoms_present}
- symptom_preview: "{symptom_preview}"
- lab_pdf_present: {lab_pdf_present}
- xray_findings_present: {xray_findings_present}
- medications_present: {medications_present}
- medication_count: {medication_count}

MedicalRAGSearch retrieves supporting medical evidence passages.

Return ONLY valid JSON. No explanation. No markdown. No other text:
{{"use_rag": true or false, "reasoning": "one sentence"}}
"""

_PROMPT_ATTEMPT_2 = """\
Return only JSON. No text before or after.

{{"use_rag": true, "reasoning": "reason here"}}

Should medical evidence be retrieved?
symptoms={symptoms_present}, lab={lab_pdf_present}, \
xray={xray_findings_present}, meds={medications_present} \
(count={medication_count})

Return: {{"use_rag": true or false, "reasoning": "one sentence"}}
"""


# ── Fallback plan ─────────────────────────────────────────────────

def _make_fallback_plan(state: AegisState) -> ExecutionPlan:
    """
    Construct a safe fallback plan when the planner fails.

    use_rag=True is unconditional — safety-first policy (Decision 50).
    When planner reasoning is unavailable, evidence retrieval defaults
    to enabled. Retrieving unnecessary evidence is acceptable; omitting
    evidence in a planner-failure scenario is not.

    is_fallback=True and was_repaired=False satisfy the model_validator
    invariant on ExecutionPlan.

    Imported by agents/pipeline.py.
    """
    return ExecutionPlan(
        use_rag      = True,
        reasoning    = "Fallback plan: planner failed after retries.",
        is_fallback  = True,
        was_repaired = False,
    )


# ── Input metadata builder ────────────────────────────────────────

def _build_prompt_vars(state: AegisState) -> dict[str, object]:
    """Extract metadata and preview for prompt substitution."""
    raw     = state.raw_symptoms_text or ""
    preview = raw[:_SYMPTOM_PREVIEW_CHARS] if raw else "none"

    return {
        "symptoms_present":      str(bool(raw or state.audio_file_path)).lower(),
        "symptom_preview":       preview,
        "lab_pdf_present":       str(bool(state.lab_pdf_path)).lower(),
        "xray_findings_present": str(bool(state.xray_findings_raw)).lower(),
        "medications_present":   str(bool(state.medications_raw)).lower(),
        "medication_count":      len(state.medications_raw),
    }


# ── Ollama call ───────────────────────────────────────────────────

async def _call_ollama(prompt: str) -> str:
    """
    Non-streaming Ollama request. Returns raw response string.
    Raises on HTTP or network failure — caller handles retry logic.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            OLLAMA_STREAM_URL,
            json={
                "model":  MODEL_TAG,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": _NUM_PREDICT,
                    "num_ctx":     _NUM_CTX,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")


# ── Plan parser ───────────────────────────────────────────────────

def _parse_plan(raw_response: str) -> ExecutionPlan:
    """
    Parse LLM response string into ExecutionPlan.

    Strips markdown code fences if present.
    Raises json.JSONDecodeError or pydantic.ValidationError on failure.
    Caller treats both as parse failure and proceeds to next attempt.
    """
    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(
            line for line in lines
            if not line.startswith("```")
        ).strip()

    data = json.loads(text)
    return ExecutionPlan(**data)


# ── Tool ──────────────────────────────────────────────────────────

class ExecutionPlanner:
    """
    SLM-powered planning step.

    Decides use_rag only — mandatory tools are pipeline invariants.
    Attempts two LLM calls before returning ToolError.
    Does not construct fallback plans — that is the pipeline's job.
    Does not validate business rules — that is PlanValidator's job.
    """

    TOOL_NAME = TOOL_EXECUTION_PLANNER

    async def run(
        self,
        state: AegisState,
    ) -> ExecutionPlan | ToolError:

        vars_ = _build_prompt_vars(state)

        # ── Attempt 1 ─────────────────────────────────────────────
        try:
            raw  = await _call_ollama(_PROMPT_ATTEMPT_1.format(**vars_))
            plan = _parse_plan(raw)
            logger.info(
                "execution_planner · attempt 1 succeeded",
                session_id=state.session_id,
                use_rag=plan.use_rag,
            )
            return plan

        except Exception as exc:
            logger.warning(
                "execution_planner · attempt 1 failed",
                session_id=state.session_id,
                error=str(exc),
            )

        # ── Attempt 2 ─────────────────────────────────────────────
        try:
            raw  = await _call_ollama(_PROMPT_ATTEMPT_2.format(**vars_))
            plan = _parse_plan(raw)
            logger.info(
                "execution_planner · attempt 2 succeeded",
                session_id=state.session_id,
                use_rag=plan.use_rag,
            )
            return plan

        except Exception as exc:
            logger.warning(
                "execution_planner · attempt 2 failed · returning ToolError",
                session_id=state.session_id,
                error=str(exc),
            )

        return ToolError(
            tool   = TOOL_EXECUTION_PLANNER,
            reason = "ExecutionPlanner failed after 2 attempts.",
            fatal  = False,
        )