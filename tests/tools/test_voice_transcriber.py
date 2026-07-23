"""
tests/tools/test_voice_transcriber.py — VoiceTranscriber.

Test structure:
    Unit tests (always run in CI)
        Mock WhisperModel via patch.
        Verify all logic paths: guards, error branches, success contract.
        No model download, no network, deterministic.

    Integration tests (@pytest.mark.whisper)
        Real Faster-Whisper model loaded from AEGIS_WHISPER_DIR.
        Real WAV fixture generated via stdlib wave module.
        Not run in standard CI — require model to be present.
        Run with: pytest -m whisper

Key behaviours under test:
    - No audio path            → ToolError(fatal=False)
    - Missing file             → ToolError(fatal=False)
    - Non-WAV file             → ToolError(fatal=False)
    - Empty transcript         → ToolError(fatal=False)
    - Whisper raises           → ToolError(fatal=False)
    - Model load fails         → ToolError(fatal=False)
    - Valid WAV + transcript   → VoiceTranscriptionResult
    - state.raw_symptoms_text  written on success
    - state.voice_result       NOT written (pipeline owns it)
    - Tool attribution         always TOOL_VOICE_TRANSCRIBER
    - _get_model_dir()         reads AEGIS_WHISPER_DIR env var
    - _get_model_dir()         falls back to default path

Segment mock uses SimpleNamespace (not MagicMock) to behave like
the real faster_whisper.transcribe.Segment dataclass. Accessing
an attribute not declared here will raise AttributeError immediately,
making contract drift visible rather than silently passing.

Singleton patch note:
    WhisperModel is imported lazily inside _load_model() with a local
    `from faster_whisper import WhisperModel`. It is NOT a module-level
    name in tools.voice_transcriber, so patching
    "tools.voice_transcriber.WhisperModel" would raise AttributeError.
    The correct target is "faster_whisper.WhisperModel", which intercepts
    the import at the source module level.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from schemas.errors import ToolError
from schemas.state import AegisState
from schemas.voice import VoiceTranscriptionResult
from tools.tool_names import TOOL_VOICE_TRANSCRIBER
from tools.voice_transcriber import (
    VoiceTranscriber,
    _get_model_dir,
    _is_real_wav,
    transcribe,
)


# ── Helpers ────────────────────────────────────────────────────────

def _make_segment(text: str) -> SimpleNamespace:
    """
    Mimics faster_whisper.transcribe.Segment dataclass.

    Fields match Segment exactly (faster-whisper 1.2.1):
        id, seek, start, end, text, tokens, avg_logprob,
        compression_ratio, no_speech_prob, words, temperature
    """
    return SimpleNamespace(
        id=0,
        seek=0,
        start=0.0,
        end=2.5,
        text=text,
        tokens=[],
        avg_logprob=-0.3,
        compression_ratio=1.2,
        no_speech_prob=0.05,
        words=None,
        temperature=0.0,
    )


def _make_wav(path: Path, n_frames: int = 16000) -> Path:
    """
    Write a valid mono 16-bit 16 kHz WAV file at path.

    n_frames=16000 → 1 second of silence.
    Returns the path for convenience.
    """
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))
    return path


def _mock_model(segments: list[SimpleNamespace]) -> MagicMock:
    """
    Return a MagicMock WhisperModel whose .transcribe() yields segments.
    info is a MagicMock (we discard it in production code).
    """
    model = MagicMock()
    model.transcribe.return_value = (segments, MagicMock())
    return model


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def wav_path(tmp_path) -> Path:
    """Valid 1-second silent WAV."""
    return _make_wav(tmp_path / "audio.wav")


@pytest.fixture
def short_wav_path(tmp_path) -> Path:
    """Minimal valid WAV (1 frame — used for guard tests)."""
    return _make_wav(tmp_path / "short.wav", n_frames=1)


@pytest.fixture
def non_wav_path(tmp_path) -> Path:
    """Plain text file — not a WAV."""
    p = tmp_path / "symptoms.txt"
    p.write_text("chest pain", encoding="utf-8")
    return p


@pytest.fixture
def empty_file_path(tmp_path) -> Path:
    """Zero-byte file."""
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    return p


# ── _get_model_dir ─────────────────────────────────────────────────

def test_get_model_dir_default(monkeypatch):
    monkeypatch.delenv("AEGIS_WHISPER_DIR", raising=False)
    result = _get_model_dir()
    assert result == Path("data/audio/whisper-tiny-en")


def test_get_model_dir_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("AEGIS_WHISPER_DIR", str(tmp_path))
    result = _get_model_dir()
    assert result == tmp_path


def test_get_model_dir_env_var_strips_whitespace(monkeypatch, tmp_path):
    monkeypatch.setenv("AEGIS_WHISPER_DIR", f"  {tmp_path}  ")
    result = _get_model_dir()
    assert result == tmp_path


def test_get_model_dir_empty_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("AEGIS_WHISPER_DIR", "")
    result = _get_model_dir()
    assert result == Path("data/audio/whisper-tiny-en")


def test_get_model_dir_whitespace_only_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("AEGIS_WHISPER_DIR", "   ")
    result = _get_model_dir()
    assert result == Path("data/audio/whisper-tiny-en")


# ── _is_real_wav ───────────────────────────────────────────────────

def test_is_real_wav_returns_true_for_valid_wav(wav_path):
    assert _is_real_wav(wav_path) is True


def test_is_real_wav_returns_false_for_text_file(non_wav_path):
    assert _is_real_wav(non_wav_path) is False


def test_is_real_wav_returns_false_for_empty_file(empty_file_path):
    assert _is_real_wav(empty_file_path) is False


def test_is_real_wav_returns_false_for_missing_file(tmp_path):
    assert _is_real_wav(tmp_path / "nonexistent.wav") is False


def test_is_real_wav_returns_false_for_truncated_header(tmp_path):
    p = tmp_path / "truncated.wav"
    p.write_bytes(b"RIFF")  # only 4 bytes — too short
    assert _is_real_wav(p) is False


# ── Guard: no audio path ───────────────────────────────────────────

async def test_no_audio_path_returns_tool_error():
    state = AegisState()
    result = await VoiceTranscriber().run(state)
    assert isinstance(result, ToolError)
    assert result.fatal is False
    assert result.tool == TOOL_VOICE_TRANSCRIBER


# ── Guard: missing file ────────────────────────────────────────────

async def test_missing_file_returns_tool_error():
    state = AegisState(audio_file_path="/nonexistent/audio.wav")
    result = await VoiceTranscriber().run(state)
    assert isinstance(result, ToolError)
    assert result.fatal is False
    assert result.tool == TOOL_VOICE_TRANSCRIBER


# ── Guard: non-WAV file ────────────────────────────────────────────

async def test_non_wav_file_returns_tool_error(non_wav_path):
    state = AegisState(audio_file_path=str(non_wav_path))
    result = await VoiceTranscriber().run(state)
    assert isinstance(result, ToolError)
    assert result.fatal is False
    assert result.tool == TOOL_VOICE_TRANSCRIBER


async def test_empty_file_returns_tool_error(empty_file_path):
    state = AegisState(audio_file_path=str(empty_file_path))
    result = await VoiceTranscriber().run(state)
    assert isinstance(result, ToolError)
    assert result.fatal is False
    assert result.tool == TOOL_VOICE_TRANSCRIBER


# ── Happy path (mocked Whisper) ────────────────────────────────────

async def test_valid_wav_returns_transcription_result(wav_path):
    segments = [_make_segment("chest pain since yesterday")]
    model = _mock_model(segments)

    with patch("tools.voice_transcriber._load_model", return_value=model):
        state = AegisState(audio_file_path=str(wav_path))
        result = await VoiceTranscriber().run(state)

    assert isinstance(result, VoiceTranscriptionResult)
    assert result.transcript == "chest pain since yesterday"


async def test_valid_wav_writes_raw_symptoms_text(wav_path):
    """VoiceTranscriber is the only tool permitted to write raw_symptoms_text."""
    segments = [_make_segment("fever and chills")]
    model = _mock_model(segments)

    with patch("tools.voice_transcriber._load_model", return_value=model):
        state = AegisState(audio_file_path=str(wav_path))
        await VoiceTranscriber().run(state)

    assert state.raw_symptoms_text == "fever and chills"


async def test_valid_wav_does_not_write_voice_result(wav_path):
    """Pipeline owns state.voice_result — tool must never assign it."""
    segments = [_make_segment("headache")]
    model = _mock_model(segments)

    with patch("tools.voice_transcriber._load_model", return_value=model):
        state = AegisState(audio_file_path=str(wav_path))
        await VoiceTranscriber().run(state)

    assert state.voice_result is None


async def test_multi_segment_transcript_joined_correctly(wav_path):
    """Multiple segments joined with single space, stripped."""
    segments = [
        _make_segment("  Patient reports  "),
        _make_segment("chest pain"),
        _make_segment("and shortness of breath.  "),
    ]
    model = _mock_model(segments)

    with patch("tools.voice_transcriber._load_model", return_value=model):
        state = AegisState(audio_file_path=str(wav_path))
        result = await VoiceTranscriber().run(state)

    assert isinstance(result, VoiceTranscriptionResult)
    assert result.transcript == "Patient reports chest pain and shortness of breath."


async def test_transcript_schema_version(wav_path):
    segments = [_make_segment("headache")]
    model = _mock_model(segments)

    with patch("tools.voice_transcriber._load_model", return_value=model):
        state = AegisState(audio_file_path=str(wav_path))
        result = await VoiceTranscriber().run(state)

    assert isinstance(result, VoiceTranscriptionResult)
    assert result.schema_version == "1.0"


# ── Guard: empty transcript ────────────────────────────────────────

async def test_empty_transcript_returns_tool_error(wav_path):
    """Whisper produced no text (silence, noise)."""
    model = _mock_model([])  # zero segments → empty join

    with patch("tools.voice_transcriber._load_model", return_value=model):
        state = AegisState(audio_file_path=str(wav_path))
        result = await VoiceTranscriber().run(state)

    assert isinstance(result, ToolError)
    assert result.fatal is False
    assert result.tool == TOOL_VOICE_TRANSCRIBER


async def test_whitespace_only_transcript_returns_tool_error(wav_path):
    """Segment produces only whitespace — strips to empty."""
    segments = [_make_segment("   "), _make_segment("  ")]
    model = _mock_model(segments)

    with patch("tools.voice_transcriber._load_model", return_value=model):
        state = AegisState(audio_file_path=str(wav_path))
        result = await VoiceTranscriber().run(state)

    assert isinstance(result, ToolError)
    assert result.fatal is False
    assert result.tool == TOOL_VOICE_TRANSCRIBER


# ── Error propagation: Whisper raises ─────────────────────────────

async def test_whisper_runtime_error_returns_tool_error(wav_path):
    """Any exception from Whisper transcribe is caught → ToolError."""
    model = MagicMock()
    model.transcribe.side_effect = RuntimeError("CUDA OOM")

    with patch("tools.voice_transcriber._load_model", return_value=model):
        state = AegisState(audio_file_path=str(wav_path))
        result = await VoiceTranscriber().run(state)

    assert isinstance(result, ToolError)
    assert result.fatal is False
    assert result.tool == TOOL_VOICE_TRANSCRIBER
    assert "RuntimeError" in result.reason


async def test_model_load_failure_returns_tool_error(wav_path):
    """If _load_model() raises (model not downloaded), return ToolError."""
    with patch(
        "tools.voice_transcriber._load_model",
        side_effect=Exception("Model directory not found"),
    ):
        state = AegisState(audio_file_path=str(wav_path))
        result = await VoiceTranscriber().run(state)

    assert isinstance(result, ToolError)
    assert result.fatal is False
    assert result.tool == TOOL_VOICE_TRANSCRIBER


async def test_tool_error_is_never_fatal(wav_path):
    """VoiceTranscriber never returns fatal=True — non-blocking step."""
    with patch(
        "tools.voice_transcriber._load_model",
        side_effect=Exception("anything"),
    ):
        state = AegisState(audio_file_path=str(wav_path))
        result = await VoiceTranscriber().run(state)

    assert isinstance(result, ToolError)
    assert result.fatal is False


# ── Functional entrypoint ──────────────────────────────────────────

async def test_transcribe_function_delegates_to_tool(wav_path):
    """transcribe() is a thin wrapper — same contract as VoiceTranscriber().run()."""
    segments = [_make_segment("shortness of breath")]
    model = _mock_model(segments)

    with patch("tools.voice_transcriber._load_model", return_value=model):
        state = AegisState(audio_file_path=str(wav_path))
        result = await transcribe(state)

    assert isinstance(result, VoiceTranscriptionResult)
    assert result.transcript == "shortness of breath"


# ── Singleton: model loaded only once ─────────────────────────────

def test_load_model_called_once_across_multiple_calls(monkeypatch):
    """
    WhisperModel constructor must be called exactly once regardless of
    how many times _load_model() is invoked.

    A regression here would silently reload the model on every
    transcription request (~40MB reload each time).

    Patch target: faster_whisper.WhisperModel (NOT tools.voice_transcriber.WhisperModel).

    WhisperModel is imported lazily inside _load_model() as a local name:
        from faster_whisper import WhisperModel
    It is never bound at module level in tools.voice_transcriber, so patching
    the module attribute would raise AttributeError. Patching the source
    module (faster_whisper.WhisperModel) intercepts the import correctly.
    """
    import tools.voice_transcriber as vt_module

    original_model = vt_module._MODEL
    vt_module._MODEL = None  # reset singleton for this test

    try:
        with patch("faster_whisper.WhisperModel") as mock_cls:
            mock_cls.return_value = MagicMock()

            from tools.voice_transcriber import _load_model
            _load_model()
            _load_model()
            _load_model()

            mock_cls.assert_called_once()
    finally:
        vt_module._MODEL = original_model  # restore


# ── State not mutated on failure ───────────────────────────────────

async def test_raw_symptoms_text_not_written_on_tool_error():
    """On any failure path, raw_symptoms_text must not be overwritten."""
    state = AegisState(
        raw_symptoms_text="original text",
        audio_file_path="/nonexistent/audio.wav",
    )
    await VoiceTranscriber().run(state)
    assert state.raw_symptoms_text == "original text"


async def test_raw_symptoms_text_not_written_on_empty_transcript(wav_path):
    model = _mock_model([])

    with patch("tools.voice_transcriber._load_model", return_value=model):
        state = AegisState(
            raw_symptoms_text="original text",
            audio_file_path=str(wav_path),
        )
        await VoiceTranscriber().run(state)

    assert state.raw_symptoms_text == "original text"


# ── Integration tests (require real model) ─────────────────────────

@pytest.mark.whisper
async def test_real_whisper_transcribes_wav(tmp_path, monkeypatch):
    """
    Integration: load real Faster-Whisper tiny.en and transcribe a
    generated WAV. Whisper on silence may return empty or hallucinate —
    we assert only on the return type, not transcript content.

    Requires: AEGIS_WHISPER_DIR set to a directory containing the
    downloaded faster-whisper tiny.en INT8 model files.
    """
    import tools.voice_transcriber as vt_module

    # Reset singleton so this test loads fresh
    vt_module._MODEL = None

    wav = _make_wav(tmp_path / "real.wav", n_frames=16000)
    state = AegisState(audio_file_path=str(wav))
    result = await VoiceTranscriber().run(state)

    # Silence may yield empty transcript → ToolError, or hallucinate text
    # → VoiceTranscriptionResult. Both are valid outcomes.
    assert isinstance(result, (VoiceTranscriptionResult, ToolError))
    if isinstance(result, ToolError):
        assert result.fatal is False
        assert result.tool == TOOL_VOICE_TRANSCRIBER

    # Cleanup singleton for other tests
    vt_module._MODEL = None