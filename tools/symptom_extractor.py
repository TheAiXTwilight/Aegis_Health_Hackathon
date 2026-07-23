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

Symptom vs. lab-value contract:
    The `symptoms` list must contain only symptom PHRASES (e.g. "chest
    tightness", "dizzy", "fever") — never lab measurements or vital
    readings (e.g. "glucose 245", "bp 140/90", "hba1c 7.2"). Those
    belong in the LabTool's output, not here.

    Enforcement is two-layer:
      1. Prompt-level  — all three prompts explicitly list lab values,
                         vitals, and biomarker readings as NOT symptoms.
                         Teaches the model but only affects new outputs.
      2. Validation-level — _strip_lab_values_from_symptoms() runs after
                         JSON parsing, before Pydantic validation. Kills
                         any "<biomarker> <number>" pattern that slipped
                         through the prompt (e.g. LLM misclassification,
                         or an older model that hasn't seen the updated
                         prompt yet).

    Reports already persisted in the DB with the old (broken) extraction
    remain broken — result_json is frozen. Only reports generated after
    this patch will have clean symptoms.

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
import re

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
#
# All three prompts now include an explicit "NOT symptoms" section
# that excludes lab values, vitals, and biomarker readings. This
# teaches the model that "glucose 245" is data ABOUT the patient,
# not a symptom the patient reported.

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

What COUNTS as a symptom:
- Any descriptive phrase describing how the patient feels or what
  they are experiencing, taken ONLY from the patient text below.
- A body part + a sensation word taken from the patient text
  (e.g. "<body part> pain", "<body part> ache").

What is NOT a symptom (never include these in "symptoms"):
- Lab values or biomarker readings, e.g. a test/biomarker name
  followed by a number (such as a glucose, HbA1c, TSH, vitamin,
  cholesterol, or creatinine reading).
- Vital signs with numbers, e.g. blood pressure, heart rate,
  oxygen saturation, or temperature readings.
- Any "<test or biomarker name> <number>" phrase — that is data,
  not a symptom. Skip it entirely. It belongs to a different tool.

CRITICAL: Only extract symptoms that are actually written in the
"Patient symptom text" section below. Do not invent, assume, or add
any symptom that is not explicitly present in that text. If the
patient text mentions only one or two symptoms, return only those
one or two symptoms — never pad the list with other symptoms.

Patient symptom text:
%%TEXT%%

Return the JSON object now:"""

_PROMPT_ATTEMPT_2 = """\
Extract clinical entities from this text. Return ONLY JSON. No other text.

Text: %%TEXT%%

Return this exact structure with no extra fields:
{"symptoms":[],"duration":"","severity_indicators":[],"medical_entities":[],"negations":[]}

Rules:
- symptoms are descriptive phrases taken ONLY from the text above —
  never invent or add symptoms not present in the text
- do NOT include lab values or vitals in symptoms (a biomarker or
  vital-sign name followed by a number is NOT a symptom — skip it)
- fill each list with items found in the text; if the text mentions
  only one symptom, return only that one symptom
- use "" for duration if not mentioned

Return JSON only:"""

_PROMPT_ATTEMPT_3 = """\
JSON only. No explanation. Symptoms are descriptive phrases taken \
only from the text below, never lab values (a biomarker name plus \
a number). Do not add symptoms that are not in the text.

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


# ── Lab-value filter ──────────────────────────────────────────────
#
# Post-parse guard. Even with the updated prompts, an older aegis-llama
# checkpoint or an off-day sampling can still emit "glucose 245" as a
# symptom. This regex + filter enforces the contract at the boundary
# between the LLM and SymptomExtractionResult.
#
# Match rule: a candidate is a "lab-value phrase" (NOT a symptom) if
# it contains one of a whitelist of biomarker/vital names followed by
# a number, OR if it matches a bare "<any word> <number>[units]" shape
# where the number is the only content besides units. A candidate is
# never dropped just because it contains a number — "fever 101" is
# borderline, but genuine descriptive symptoms rarely include numbers,
# so bare "<word> <number>" is treated as a measurement.
#
# Words that indicate a lab/vital reading rather than a symptom.
# Keep this list conservative — a false-positive here silently drops
# a real symptom.

_LAB_KEYWORDS = (
    # Blood chemistry
    "glucose", "sugar", "hba1c", "a1c", "cholesterol", "ldl", "hdl",
    "triglyceride", "creatinine", "urea", "bun", "sodium", "potassium",
    "calcium", "magnesium", "chloride", "bicarbonate", "albumin",
    # Endocrine
    "tsh", "t3", "t4", "insulin", "cortisol",
    # Hematology
    "hemoglobin", "haemoglobin", "hgb", "hct", "hematocrit",
    "wbc", "rbc", "platelet", "platelets",
    # Cardiac markers
    "troponin", "ck", "ck-mb", "bnp", "nt-probnp",
    # Vitamins & minerals
    "vitamin", "vit", "ferritin", "iron", "b12", "folate",
    # Vitals
    "bp", "blood pressure", "systolic", "diastolic",
    "heart rate", "hr", "pulse", "spo2", "oxygen saturation",
    "temperature", "temp", "respiratory rate", "rr",
    # Other
    "creatine", "bilirubin", "alt", "ast", "alp", "ggt",
)

# A bare-measurement candidate is anything of shape
# "<word> <optional-punctuation> <number>[optional-unit]" with nothing
# else meaningful. Used as a secondary check when a lab keyword doesn't
# match but the phrase is clearly a measurement.
#
# Example matches: "glucose 245", "bp 140/90", "tsh: 6.05", "hba1c=7.2"
# Example non-matches: "chest pain", "dizzy for 2 days" (has other words),
#                       "3 days" (no leading word)

_BARE_MEASUREMENT_RE = re.compile(
    r"""
    ^\s*
    [a-z][a-z0-9\-\s]{0,25}?    # leading name (letters, digits, dashes, spaces)
    \s*[:=]?\s*                  # optional : or =
    \d+(?:\.\d+)?                # number
    (?:\s*/\s*\d+(?:\.\d+)?)?    # optional /number (e.g. 140/90)
    \s*
    [a-zA-Zµ%/]{0,10}            # optional unit (mg/dL, %, mmol/L, µg, etc.)
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _looks_like_lab_value(phrase: str) -> bool:
    """
    True if a candidate symptom phrase actually describes a lab value,
    vital sign, or biomarker reading rather than a symptom.

    Two-part check:
      1. Contains a known lab/vital keyword AND at least one digit.
         This catches "glucose 245", "TSH 6.05", "vitamin d 12".
      2. Matches the bare "<word> <number>[unit]" shape with nothing
         extra. Catches things not in the keyword list, like a rare
         biomarker "urate 8.5" the whitelist doesn't cover.

    Conservative on purpose — genuine descriptive symptoms like
    "chest pain", "shortness of breath", or "severe headache" cannot
    match either rule because they contain no digit AND don't fit the
    bare-measurement shape.
    """
    if not phrase:
        return False

    lowered = phrase.lower().strip()

    # Rule 1: known lab/vital keyword + a digit
    has_digit = any(ch.isdigit() for ch in lowered)
    if has_digit:
        for kw in _LAB_KEYWORDS:
            # Word-boundary match so "hr" doesn't match inside "hurt"
            if re.search(r"\b" + re.escape(kw) + r"\b", lowered):
                return True

    # Rule 2: bare "<word> <number>[unit]" shape
    if _BARE_MEASUREMENT_RE.match(lowered):
        return True

    return False


def _strip_lab_values_from_symptoms(symptoms: list) -> list:
    """
    Return the symptoms list with lab-value phrases removed. Logs a
    warning for each dropped phrase so misclassifications are visible
    in production logs.

    Accepts a list (already coerced by _coerce_field). Non-string
    items are dropped silently — Pydantic would have rejected them
    anyway.
    """
    cleaned: list[str] = []
    dropped: list[str] = []
    for item in symptoms:
        if not isinstance(item, str):
            continue
        if _looks_like_lab_value(item):
            dropped.append(item)
            continue
        cleaned.append(item)

    if dropped:
        logger.warning(
            "symptom_extractor · dropped lab-value phrases from symptoms",
            dropped=dropped,
        )

    return cleaned


def _normalize_words(text: str) -> set[str]:
    """Lowercase word set for lexical overlap checks (ignores punctuation)."""
    return set(re.findall(r"[a-z]+", text.lower()))


def _filter_ungrounded_symptoms(symptoms: list, source_text: str) -> list:
    """
    Drop symptom phrases that share no words with the actual patient
    input. Guards against the LLM parroting few-shot example phrases
    (or otherwise inventing symptoms) instead of extracting from the
    real text. Logs a warning for each dropped phrase.

    A phrase is kept if at least one of its content words (length > 2,
    to skip stray "a"/"is"/etc.) appears in the source text. This is
    intentionally permissive — it only needs to catch wholesale
    fabrication, not lightly-paraphrased extraction.
    """
    source_words = _normalize_words(source_text)
    if not source_words:
        return symptoms

    kept: list[str] = []
    dropped: list[str] = []
    for item in symptoms:
        if not isinstance(item, str):
            continue
        phrase_words = {w for w in _normalize_words(item) if len(w) > 2}
        if not phrase_words or phrase_words & source_words:
            kept.append(item)
        else:
            dropped.append(item)

    if dropped:
        logger.warning(
            "symptom_extractor · dropped ungrounded symptom phrases "
            "(not present in source text)",
            dropped=dropped,
        )

    return kept


def _dedupe_overlapping_symptoms(symptoms: list) -> list:
    """
    Collapse near-duplicate symptom phrases the model sometimes emits for
    a single real symptom — e.g. "chest pain" and "chest pain ache" both
    extracted from one "I have chest pain" input. Both phrases are
    individually grounded (share words with the source text), so
    _filter_ungrounded_symptoms correctly keeps both; this is a
    different problem — overlap between the *extracted* phrases
    themselves, not fabrication.

    If one phrase's word set is a subset of another's, keep only the
    longer (more specific) phrase. This is intentionally conservative:
    it only collapses true subset/superset pairs, not merely related
    phrases, so e.g. "chest pain" and "shortness of breath" (both real,
    distinct symptoms) are never merged.
    """
    if len(symptoms) < 2:
        return symptoms

    parsed = [
        (item, {w for w in _normalize_words(item) if len(w) > 2})
        for item in symptoms
        if isinstance(item, str)
    ]

    drop_indices: set[int] = set()
    for i, (text_i, words_i) in enumerate(parsed):
        if i in drop_indices or not words_i:
            continue
        for j, (text_j, words_j) in enumerate(parsed):
            if i == j or j in drop_indices or not words_j:
                continue
            if words_i < words_j:
                # i is a strict subset of j — i is redundant, drop it.
                drop_indices.add(i)
                break
            if words_i == words_j and j > i:
                # Identical word sets (e.g. differ only in stray
                # punctuation/casing already normalized away) — keep
                # the first occurrence, drop the later duplicate.
                drop_indices.add(j)

    return [item for idx, (item, _) in enumerate(parsed) if idx not in drop_indices]


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


def _parse_response(raw: str, source_text: str = "") -> SymptomExtractionResult:
    """
    Parse LLM JSON response into SymptomExtractionResult.

    Steps:
        1. Strip markdown code fences (same pattern as execution_planner)
        2. Extract JSON object via brace-counting parser
        3. json.loads → dict
        4. Per-field coercion (None, scalar → correct Python type)
        5. Filter lab-value phrases out of `symptoms` (guardrail
           against LLM misclassifying "glucose 245" as a symptom)
        6. Filter out symptoms with no lexical grounding in the actual
           patient text (guardrail against the LLM parroting prompt
           few-shot examples instead of extracting from the real input)
        7. SymptomExtractionResult(**data) — Pydantic validates

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

    # Contract enforcement: drop lab-value phrases that the LLM
    # misclassified as symptoms. Only touches `symptoms` — lab
    # readings never belonged there and this is where they get
    # filtered out before persistence.
    data["symptoms"] = _strip_lab_values_from_symptoms(data["symptoms"])

    # Contract enforcement: drop symptom phrases the model invented
    # rather than extracted (e.g. echoed prompt examples). Only runs
    # when we have source text to check against.
    if source_text:
        data["symptoms"] = _filter_ungrounded_symptoms(data["symptoms"], source_text)

    # Contract enforcement: collapse near-duplicate/overlapping symptom
    # phrases (e.g. "chest pain" + "chest pain ache" from one real
    # symptom) so downstream text like Section 10's recommendation line
    # never reads "chest pain and chest pain ache".
    data["symptoms"] = _dedupe_overlapping_symptoms(data["symptoms"])

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
                    result = _parse_response(raw, source_text=text)

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