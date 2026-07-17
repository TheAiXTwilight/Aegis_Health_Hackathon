"""
tools/symptom_extractor.py — Symptom extraction (Step 2).

Phase 3 replacement: LLM-based extractor using aegis-llama via Ollama.
Replaces the rule-based regex/keyword implementation.

Architecture:
    - Non-streaming Ollama call (same as ExecutionPlanner)
    - Temperature 0.0 — structured extraction, not creative output
    - 3-attempt retry with simplified prompt on each failure
    - Returns ToolError(fatal=False) after 3 consecutive failures
    - JSON response validated via Pydantic (SymptomExtractionResult)
    - Markdown fence stripping (same pattern as _parse_plan in execution_planner.py)

Public interface unchanged:
    async def run(self, state: AegisState) -> SymptomExtractionResult | ToolError

Response schema (what the LLM must return):

    {
      "symptoms": ["headache", "nausea"],
      "duration": "2 days",
      "severity_indicators": ["severe", "persistent"],
      "medical_entities": ["head", "stomach"],
      "negations": ["no fever"]
    }

    All fields are required in the JSON. Empty list [] is valid for
    list fields. Empty string "" is valid for duration when not mentioned.

Prompt policy:
    Attempt 1: full structured extraction prompt with field definitions
    Attempt 2: simplified prompt with explicit JSON-only instruction
    Attempt 3: minimal prompt — just the schema and the text
    Return ToolError after attempt 3 fails.

Exception handling:
    Each attempt catches only expected failure modes:
        json.JSONDecodeError       — LLM emitted non-JSON output
        pydantic.ValidationError   — JSON parsed but schema mismatch
        httpx.HTTPStatusError      — Ollama returned 4xx/5xx
        httpx.RequestError         — network / timeout / connection
    Unexpected exceptions (AttributeError, TypeError, etc.) are NOT
    caught per-attempt — they propagate to the outer try/except in
    run() and become a ToolError with the real error message, never
    silently retried. This prevents programming bugs masquerading
    as LLM failures.

Input truncation:
    Input text is truncated to _MAX_INPUT_CHARS (4000) before being
    injected into any prompt. Prevents context-window overflow on
    Ollama when a patient pastes unusually long text.

JSON extraction:
    Uses a brace-counting parser to extract the first balanced JSON
    object from the response. This handles nested objects and multiple
    JSON objects in the output correctly.

Coercion policy:
    - duration: None → "" (absent field)
    - list fields: None → [] (absent field)
    - list fields: str → [str] (LLM returned a scalar instead of list)
      Preserves recoverable information — "headache" → ["headache"]
      rather than silently dropping it.

Fallback note:
    There is no rule-based fallback. If all 3 LLM attempts fail, the
    pipeline continues without symptom extraction (state.symptom_result
    = ToolError). SeverityScorer still runs on raw text fields, and
    ReportGenerator uses raw_symptoms_text directly. Clinical safety is
    preserved by the mandatory stages.

Input priority:
    Voice transcript takes precedence over raw_symptoms_text when
    both are available (preserves Phase 2.5 behaviour).
"""

from __future__ import annotations

import json
import os

import httpx
from loguru import logger
from pydantic import ValidationError

from schemas.errors import ToolError
from schemas.state import AegisState
from schemas.symptom import SymptomExtractionResult
from tools.tool_names import TOOL_SYMPTOM_EXTRACTOR


# ── Ollama configuration ──────────────────────────────────────────

OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_STREAM_URL = OLLAMA_BASE_URL + "/api/generate"
MODEL_TAG         = "aegis-llama"

_NUM_PREDICT    = 256   # symptom extraction is small — 256 tokens is enough
_NUM_CTX        = 4096
_TIMEOUT_S      = 30
_MAX_INPUT_CHARS = 4000  # truncate input before prompt injection


# ── Prompt templates ──────────────────────────────────────────────
#
# Using .replace() with %%TEXT%% — never .format() on patient input.
# Prevents KeyError from curly braces in patient-supplied text.

_PROMPT_ATTEMPT_1 = """\
You are a clinical NLP assistant. Extract structured clinical entities \
from the patient symptom text below.

Return ONLY a single valid JSON object. No explanation. No markdown. \
No text before or after the JSON object.

Required JSON structure (all fields are required):
{
  "symptoms": ["list of individual symptom phrases"],
  "duration": "duration string or empty string if not mentioned",
  "severity_indicators": ["severity words found: severe, mild, moderate, intense, etc"],
  "medical_entities": ["body parts and organs mentioned: chest, heart, head, etc"],
  "negations": ["negation phrases: no fever, denies cough, etc"]
}

Rules:
- symptoms: split compound sentences into individual symptom phrases
- duration: extract exactly as stated (e.g. "2 days", "3 weeks") or "" if absent
- severity_indicators: only words that describe symptom severity
- medical_entities: only anatomical terms and organs
- negations: full negation phrases, not just the negation word
- All list values must be strings
- duration must be a string, not null

Patient symptom text:
%%TEXT%%

Return the JSON object now:"""

_PROMPT_ATTEMPT_2 = """\
Extract clinical entities from this text. Return ONLY JSON. No other text.

Text: %%TEXT%%

Return this exact structure with no extra fields:
{"symptoms":[],"duration":"","severity_indicators":[],"medical_entities":[],"negations":[]}

Fill each list with items found in the text. Use "" for duration if not mentioned.
Return JSON only:"""

_PROMPT_ATTEMPT_3 = """\
JSON only. No explanation.

%%TEXT%%

{"symptoms":[],"duration":"","severity_indicators":[],"medical_entities":[],"negations":[]}"""


_PROMPTS = [_PROMPT_ATTEMPT_1, _PROMPT_ATTEMPT_2, _PROMPT_ATTEMPT_3]

# Exceptions that indicate expected LLM/network failure modes.
# Caught per-attempt to allow retry. Anything outside this set
# (AttributeError, TypeError, KeyError, etc.) propagates — it is
# a programming bug, not an LLM failure.
_RETRY_EXCEPTIONS = (
    json.JSONDecodeError,
    ValidationError,
    httpx.HTTPStatusError,
    httpx.RequestError,
)


# ── Ollama call ───────────────────────────────────────────────────

async def _call_ollama(prompt: str) -> str:
    """
    Non-streaming Ollama request. Returns raw response string.

    Raises httpx.HTTPStatusError on 4xx/5xx.
    Raises httpx.RequestError on network / timeout / connection failure.
    Caller handles retry logic.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
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


# ── Response parser ───────────────────────────────────────────────

def _extract_json_object(text: str) -> str:
    r"""
    Extract the first brace-balanced JSON object from text.

    Uses a brace-counting parser rather than regex. This correctly
    handles both failure modes that regex approaches suffer from:

        Greedy (r'\{.*\}', DOTALL): matches first '{' to LAST '}'
            — merges two separate objects into one unparseable blob.

        Non-greedy (r'\{.*?\}', DOTALL): matches first '{' to FIRST '}'
            — truncates nested objects at the first closing brace.

    Algorithm:
        Scan forward from the first '{'. Increment depth on '{',
        decrement on '}'. Return the substring when depth returns
        to zero — that is the first complete, balanced object.

    Returns the matched substring, or the original text if no '{'
    is found (json.loads will then raise JSONDecodeError → retry).
    """
    start = text.find("{")
    if start == -1:
        return text  # no object found — let json.loads raise

    depth     = 0
    in_string = False
    escape    = False

    for i, ch in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # Unbalanced braces — return from start to end and let json.loads raise.
    return text[start:]


def _coerce_field(value: object, field: str) -> object:
    """
    Coerce a single field value to the expected Python type.

    duration:      None → "" (absent); non-string → str(value)
    list fields:   None → []  (absent)
                   str  → [str] (scalar — preserves recoverable data)
                   other non-list → []
    """
    if field == "duration":
        if value is None:
            return ""
        if not isinstance(value, str):
            return str(value)
        return value

    # list fields: symptoms, severity_indicators, medical_entities, negations
    if value is None:
        return []
    if isinstance(value, str):
        # LLM returned a scalar instead of a list — wrap and preserve it.
        return [value]
    if not isinstance(value, list):
        return []
    return value


def _parse_response(raw: str) -> SymptomExtractionResult:
    """
    Parse LLM JSON response into SymptomExtractionResult.

    Steps:
        1. Strip markdown code fences (same pattern as execution_planner)
        2. Extract JSON object via regex (safe: avoids spurious braces)
        3. json.loads → dict
        4. Per-field coercion (None, scalar → correct Python type)
        5. SymptomExtractionResult(**data) — Pydantic validates

    Raises json.JSONDecodeError or pydantic.ValidationError on failure.
    Caller (SymptomExtractor.run) catches these as _RETRY_EXCEPTIONS.
    """
    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(
            line for line in lines
            if not line.startswith("```")
        ).strip()

    text = _extract_json_object(text)
    data = json.loads(text)

    # Coerce fields individually — preserve recoverable values
    data["duration"]             = _coerce_field(data.get("duration"),             "duration")
    data["symptoms"]             = _coerce_field(data.get("symptoms"),             "symptoms")
    data["severity_indicators"]  = _coerce_field(data.get("severity_indicators"),  "severity_indicators")
    data["medical_entities"]     = _coerce_field(data.get("medical_entities"),     "medical_entities")
    data["negations"]            = _coerce_field(data.get("negations"),            "negations")

    return SymptomExtractionResult(**data)


# ── Tool ──────────────────────────────────────────────────────────

class SymptomExtractor:
    """
    LLM-based symptom extractor using aegis-llama (non-streaming).

    3 attempts with progressively simplified prompts.
    Per-attempt exception handling catches only expected failure modes
    (JSON/validation/HTTP/network) — unexpected exceptions propagate
    to the outer handler and surface as ToolError immediately.
    Does not write to state — pipeline owns state mutation.
    """

    TOOL_NAME = TOOL_SYMPTOM_EXTRACTOR

    async def run(
        self,
        state: AegisState,
    ) -> SymptomExtractionResult | ToolError:

        try:
            # Input priority: voice transcript > raw_symptoms_text
            text = state.raw_symptoms_text or ""

            if (
                state.voice_result is not None
                and not isinstance(state.voice_result, ToolError)
                and state.voice_result.transcript
            ):
                text = state.voice_result.transcript

            text = text.strip()

            if not text:
                return ToolError(
                    tool=TOOL_SYMPTOM_EXTRACTOR,
                    reason="No symptom text available for extraction.",
                    fatal=False,
                )

            # Truncate to avoid context-window overflow on unusually
            # long inputs (e.g. copy-pasted clinical notes).
            if len(text) > _MAX_INPUT_CHARS:
                logger.warning(
                    "symptom_extractor · input truncated",
                    session_id=state.session_id,
                    original_len=len(text),
                    truncated_to=_MAX_INPUT_CHARS,
                )
                text = text[:_MAX_INPUT_CHARS]

            # ── 3-attempt retry ────────────────────────────────────
            last_error: str = ""

            for attempt_num, prompt_template in enumerate(_PROMPTS, start=1):
                prompt = prompt_template.replace("%%TEXT%%", text)

                try:
                    raw    = await _call_ollama(prompt)
                    result = _parse_response(raw)

                    logger.info(
                        "symptom_extractor · attempt succeeded",
                        session_id=state.session_id,
                        attempt=attempt_num,
                        n_symptoms=len(result.symptoms),
                        duration=result.duration,
                    )
                    return result

                except _RETRY_EXCEPTIONS as exc:
                    # Expected failure — retry with next prompt.
                    last_error = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "symptom_extractor · attempt failed (retryable)",
                        session_id=state.session_id,
                        attempt=attempt_num,
                        error=last_error,
                    )
                # Non-retryable exceptions (programming bugs) propagate
                # to the outer except and surface as ToolError immediately.

            # All 3 attempts exhausted
            logger.error(
                "symptom_extractor · all 3 attempts failed",
                session_id=state.session_id,
                last_error=last_error,
            )
            return ToolError(
                tool=TOOL_SYMPTOM_EXTRACTOR,
                reason=(
                    f"SymptomExtractor failed after 3 attempts. "
                    f"Last error: {last_error}"
                ),
                fatal=False,
            )

        except Exception as exc:
            # Catches unexpected exceptions from setup code or
            # non-retryable exceptions propagated from the retry loop.
            return ToolError(
                tool=TOOL_SYMPTOM_EXTRACTOR,
                reason=f"{type(exc).__name__}: {exc}",
                fatal=False,
            )


async def extract(state: AegisState) -> SymptomExtractionResult | ToolError:
    """Canonical functional entrypoint."""
    return await SymptomExtractor().run(state)