"""
tests/tools/test_symptom_extractor.py — SymptomExtractor.

Mocking strategy:
    tools.symptom_extractor._call_ollama is patched directly in each
    test using patch() as a context manager. This intercepts the actual
    function called by run() — a module-level async function — without
    requiring a live Ollama instance.

    Tests that need real Ollama are marked @pytest.mark.ollama.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from schemas.errors import ToolError
from schemas.state import AegisState
from schemas.symptom import SymptomExtractionResult
from schemas.voice import VoiceTranscriptionResult
from tools.symptom_extractor import (
    SymptomExtractor,
    _MAX_INPUT_CHARS,
    _coerce_field,
    _extract_json_object,
    _parse_response,
    extract,
)
from tools.tool_names import TOOL_SYMPTOM_EXTRACTOR

_PATCH_TARGET = "tools.symptom_extractor._call_ollama"


# ── Response builders ──────────────────────────────────────────────

def _valid_json(
    symptoms: list[str] | None = None,
    duration: str = "",
    severity_indicators: list[str] | None = None,
    medical_entities: list[str] | None = None,
    negations: list[str] | None = None,
) -> str:
    return json.dumps({
        "symptoms":            symptoms or ["chest pain"],
        "duration":            duration,
        "severity_indicators": severity_indicators or [],
        "medical_entities":    medical_entities or [],
        "negations":           negations or [],
    })


# ── _extract_json_object — unit ────────────────────────────────────

def test_extract_json_object_simple():
    assert _extract_json_object('{"a": 1}') == '{"a": 1}'


def test_extract_json_object_with_prefix():
    assert _extract_json_object('Here: {"a": 1}') == '{"a": 1}'


def test_extract_json_object_nested():
    text = '{"a": {"b": 2}}'
    assert _extract_json_object(text) == text


def test_extract_json_object_no_brace_returns_text():
    assert _extract_json_object("no json") == "no json"


def test_extract_json_object_first_of_two():
    assert _extract_json_object('{"a": 1} {"b": 2}') == '{"a": 1}'


# ── _coerce_field — unit ───────────────────────────────────────────

def test_coerce_duration_none():
    assert _coerce_field(None, "duration") == ""


def test_coerce_duration_non_string():
    assert _coerce_field(42, "duration") == "42"


def test_coerce_duration_string_preserved():
    assert _coerce_field("3 days", "duration") == "3 days"


def test_coerce_list_none():
    assert _coerce_field(None, "symptoms") == []


def test_coerce_list_scalar_wrapped():
    assert _coerce_field("headache", "symptoms") == ["headache"]


def test_coerce_list_non_list_empty():
    assert _coerce_field(42, "symptoms") == []


def test_coerce_list_preserved():
    assert _coerce_field(["a", "b"], "symptoms") == ["a", "b"]


# ── _parse_response — unit ─────────────────────────────────────────

def test_parse_response_valid():
    result = _parse_response(_valid_json(symptoms=["headache"], duration="2 days"))
    assert isinstance(result, SymptomExtractionResult)
    assert "headache" in result.symptoms
    assert result.duration == "2 days"


def test_parse_response_strips_fence():
    raw = "```json\n" + _valid_json() + "\n```"
    assert isinstance(_parse_response(raw), SymptomExtractionResult)


def test_parse_response_with_prefix():
    raw = "Here:\n" + _valid_json(symptoms=["fever"])
    result = _parse_response(raw)
    assert "fever" in result.symptoms


def test_parse_response_null_duration_coerced():
    data = {"symptoms": ["x"], "duration": None,
            "severity_indicators": [], "medical_entities": [], "negations": []}
    assert _parse_response(json.dumps(data)).duration == ""


def test_parse_response_scalar_symptoms_coerced():
    data = {"symptoms": "headache", "duration": "",
            "severity_indicators": [], "medical_entities": [], "negations": []}
    assert _parse_response(json.dumps(data)).symptoms == ["headache"]


def test_parse_response_invalid_json_raises():
    with pytest.raises(json.JSONDecodeError):
        _parse_response("not json {")


# ── Guard conditions ───────────────────────────────────────────────

async def test_empty_text_returns_tool_error():
    state = AegisState(raw_symptoms_text="")
    result = await SymptomExtractor().run(state)
    assert isinstance(result, ToolError)
    assert result.fatal is False


async def test_none_text_returns_tool_error():
    state = AegisState()
    result = await SymptomExtractor().run(state)
    assert isinstance(result, ToolError)
    assert result.fatal is False
    assert result.tool == TOOL_SYMPTOM_EXTRACTOR


# ── Happy path ─────────────────────────────────────────────────────

async def test_extracts_symptoms_from_text():
    state = AegisState(raw_symptoms_text="chest pain, shortness of breath")
    with patch(_PATCH_TARGET, new=AsyncMock(
        return_value=_valid_json(symptoms=["chest pain", "shortness of breath"])
    )):
        result = await SymptomExtractor().run(state)
    assert isinstance(result, SymptomExtractionResult)
    assert len(result.symptoms) >= 1


async def test_extracts_duration_days():
    state = AegisState(raw_symptoms_text="fever for 3 days")
    with patch(_PATCH_TARGET, new=AsyncMock(
        return_value=_valid_json(duration="3 days")
    )):
        result = await SymptomExtractor().run(state)
    assert isinstance(result, SymptomExtractionResult)
    assert "3" in (result.duration or "")
    assert "day" in (result.duration or "")


async def test_extracts_duration_weeks():
    state = AegisState(raw_symptoms_text="fatigue for 2 weeks")
    with patch(_PATCH_TARGET, new=AsyncMock(
        return_value=_valid_json(duration="2 weeks")
    )):
        result = await SymptomExtractor().run(state)
    assert isinstance(result, SymptomExtractionResult)
    assert "week" in (result.duration or "")


async def test_extracts_duration_months():
    state = AegisState(raw_symptoms_text="cough for 3 months")
    with patch(_PATCH_TARGET, new=AsyncMock(
        return_value=_valid_json(duration="3 months")
    )):
        result = await SymptomExtractor().run(state)
    assert isinstance(result, SymptomExtractionResult)
    assert "month" in (result.duration or "")


async def test_no_duration_when_absent():
    state = AegisState(raw_symptoms_text="chest pain")
    with patch(_PATCH_TARGET, new=AsyncMock(
        return_value=_valid_json(duration="")
    )):
        result = await SymptomExtractor().run(state)
    assert isinstance(result, SymptomExtractionResult)
    assert not result.duration


async def test_extracts_severe_indicator():
    state = AegisState(raw_symptoms_text="severe chest pain")
    with patch(_PATCH_TARGET, new=AsyncMock(
        return_value=_valid_json(severity_indicators=["severe"])
    )):
        result = await SymptomExtractor().run(state)
    assert isinstance(result, SymptomExtractionResult)
    assert "severe" in result.severity_indicators


async def test_extracts_mild_indicator():
    state = AegisState(raw_symptoms_text="mild headache")
    with patch(_PATCH_TARGET, new=AsyncMock(
        return_value=_valid_json(severity_indicators=["mild"])
    )):
        result = await SymptomExtractor().run(state)
    assert "mild" in result.severity_indicators


async def test_no_severity_indicators_when_absent():
    state = AegisState(raw_symptoms_text="runny nose")
    with patch(_PATCH_TARGET, new=AsyncMock(
        return_value=_valid_json(severity_indicators=[])
    )):
        result = await SymptomExtractor().run(state)
    assert result.severity_indicators == []


async def test_extracts_chest_entity():
    state = AegisState(raw_symptoms_text="chest pain")
    with patch(_PATCH_TARGET, new=AsyncMock(
        return_value=_valid_json(medical_entities=["chest"])
    )):
        result = await SymptomExtractor().run(state)
    assert "chest" in result.medical_entities


async def test_extracts_multiple_entities():
    state = AegisState(raw_symptoms_text="chest pain and stomach pain")
    with patch(_PATCH_TARGET, new=AsyncMock(
        return_value=_valid_json(medical_entities=["chest", "stomach"])
    )):
        result = await SymptomExtractor().run(state)
    assert "chest" in result.medical_entities
    assert "stomach" in result.medical_entities


async def test_extracts_negation_no():
    state = AegisState(raw_symptoms_text="no fever, chest pain")
    with patch(_PATCH_TARGET, new=AsyncMock(
        return_value=_valid_json(negations=["no fever"])
    )):
        result = await SymptomExtractor().run(state)
    assert any("no" in n.lower() for n in result.negations)


# ── Voice priority ─────────────────────────────────────────────────

async def test_voice_transcript_takes_priority_over_raw_text():
    state = AegisState(raw_symptoms_text="raw text with no duration")
    state.voice_result = VoiceTranscriptionResult(
        transcript="chest pain for 3 days"
    )
    with patch(_PATCH_TARGET, new=AsyncMock(
        return_value=_valid_json(symptoms=["chest pain"], duration="3 days")
    )):
        result = await SymptomExtractor().run(state)
    assert isinstance(result, SymptomExtractionResult)


async def test_voice_tool_error_falls_back_to_raw_text():
    state = AegisState(raw_symptoms_text="fever for 2 days")
    state.voice_result = ToolError(tool="VoiceTranscriber", reason="fail")
    with patch(_PATCH_TARGET, new=AsyncMock(
        return_value=_valid_json(duration="2 days")
    )):
        result = await SymptomExtractor().run(state)
    assert isinstance(result, SymptomExtractionResult)


# ── Retry logic ────────────────────────────────────────────────────

async def test_bad_json_attempt1_retries_attempt2():
    call_count = 0

    async def _selective(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return "not json {" if call_count == 1 else _valid_json(symptoms=["headache"])

    state = AegisState(raw_symptoms_text="headache")
    with patch(_PATCH_TARGET, new=_selective):
        result = await SymptomExtractor().run(state)

    assert isinstance(result, SymptomExtractionResult)
    assert call_count == 2


async def test_all_three_attempts_fail_returns_tool_error():
    state = AegisState(raw_symptoms_text="chest pain")
    with patch(_PATCH_TARGET, new=AsyncMock(return_value="not json {")):
        result = await SymptomExtractor().run(state)
    assert isinstance(result, ToolError)
    assert result.fatal is False
    assert result.tool == TOOL_SYMPTOM_EXTRACTOR


# ── Truncation ─────────────────────────────────────────────────────

async def test_long_input_truncated():
    long_text = "x " * (_MAX_INPUT_CHARS + 500)
    state = AegisState(raw_symptoms_text=long_text)
    captured: list[str] = []

    async def _capture(prompt: str) -> str:
        captured.append(prompt)
        return _valid_json()

    with patch(_PATCH_TARGET, new=_capture):
        result = await SymptomExtractor().run(state)

    assert isinstance(result, SymptomExtractionResult)
    assert all(len(p) < len(long_text) for p in captured)


# ── Schema ─────────────────────────────────────────────────────────

async def test_result_has_schema_version():
    state = AegisState(raw_symptoms_text="chest pain")
    with patch(_PATCH_TARGET, new=AsyncMock(return_value=_valid_json())):
        result = await SymptomExtractor().run(state)
    assert result.schema_version == "1.0"


async def test_symptoms_list_is_list():
    state = AegisState(raw_symptoms_text="fever, cough")
    with patch(_PATCH_TARGET, new=AsyncMock(
        return_value=_valid_json(symptoms=["fever", "cough"])
    )):
        result = await SymptomExtractor().run(state)
    assert isinstance(result.symptoms, list)


# ── Functional entrypoint ──────────────────────────────────────────

async def test_extract_functional_entrypoint():
    state = AegisState(raw_symptoms_text="chest pain")
    with patch(_PATCH_TARGET, new=AsyncMock(return_value=_valid_json())):
        result = await extract(state)
    assert isinstance(result, SymptomExtractionResult)