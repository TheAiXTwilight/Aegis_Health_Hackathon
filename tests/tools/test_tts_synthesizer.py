"""
tests/tools/test_tts_synthesizer.py — TTSSynthesizer (Piper, local TTS).

Test structure mirrors tests/tools/test_voice_transcriber.py:

    Unit tests (always run in CI)
        Mock PiperVoice via patch. No model download, no network,
        deterministic.

    Integration tests (@pytest.mark.piper)
        Real Piper voice loaded from AEGIS_TTS_MODEL_DIR. Not run in
        standard CI — require the voice model files to be present.
        Run with: pytest -m piper

Key behaviours under test:
    - Empty / whitespace-only text        → TTSError(reason="synthesis_failed")
    - Text exceeding max_chars            → TTSError(reason="text_too_long")
    - Missing model files                 → TTSError(reason="model_missing")
    - piper-tts not installed             → TTSError(reason="model_missing")
    - PiperVoice.load raises              → TTSError (generic)
    - synthesize_wav raises               → TTSError (generic)
    - Valid text + mocked voice           → non-empty WAV bytes returned
    - clean_text_for_speech               strips markdown formatting
    - _get_model_dir                      reads settings.AEGIS_TTS_MODEL_DIR
    - _get_voice_name                     reads settings.AEGIS_TTS_VOICE_NAME
    - Singleton                           PiperVoice loaded once, reused

Config source note:
    tools/tts_synthesizer.py reads AEGIS_TTS_MODEL_DIR / AEGIS_TTS_VOICE_NAME
    via app.settings.settings (a pydantic-settings singleton), NOT raw
    os.environ. Tests patch the `settings` object's attributes directly
    (monkeypatch.setattr) rather than monkeypatch.setenv, since setenv
    alone won't affect an already-constructed settings instance.

Singleton patch note:
    PiperVoice is imported lazily inside _load_voice() with a local
    `from piper import PiperVoice`. It is NOT a module-level name in
    tools.tts_synthesizer, so patching "tools.tts_synthesizer.PiperVoice"
    would raise AttributeError. The correct target is "piper.PiperVoice",
    which intercepts the import at the source module level.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.settings import settings
from tools.tts_synthesizer import (
    TTSError,
    _get_model_dir,
    _get_voice_name,
    _resolve_model_files,
    clean_text_for_speech,
    synthesize_speech,
)
import tools.tts_synthesizer as tts_module


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure the module-level _VOICE singleton doesn't leak across tests."""
    tts_module._VOICE = None
    yield
    tts_module._VOICE = None


@pytest.fixture
def model_dir(tmp_path, monkeypatch) -> Path:
    """A directory with valid (empty-content-ok) .onnx + .onnx.json files,
    named to match settings.AEGIS_TTS_VOICE_NAME at fixture time so
    _resolve_model_files() finds them regardless of which voice is
    configured as the default.
    """
    voice_name = settings.AEGIS_TTS_VOICE_NAME
    d = tmp_path / "voice"
    d.mkdir()
    (d / f"{voice_name}.onnx").write_bytes(b"fake-onnx-bytes")
    (d / f"{voice_name}.onnx.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(settings, "AEGIS_TTS_MODEL_DIR", str(d))
    return d


def _mock_voice() -> MagicMock:
    """
    Return a MagicMock PiperVoice whose synthesize_wav writes a few
    bytes of fake PCM data into the wave.Wave_write handed to it, so
    callers can assert non-empty output without a real model.
    """
    voice = MagicMock()

    def _fake_synthesize_wav(text, wav_file, *args, **kwargs):
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x01" * 100)

    voice.synthesize_wav.side_effect = _fake_synthesize_wav
    return voice


# ── clean_text_for_speech ────────────────────────────────────────

class TestCleanTextForSpeech:
    def test_strips_markdown_headers(self):
        assert clean_text_for_speech("## Summary") == "Summary"

    def test_strips_bold_markers(self):
        assert clean_text_for_speech("**important**") == "important"

    def test_strips_links_keeps_label(self):
        assert clean_text_for_speech("[click here](http://x)") == "click here"

    def test_strips_bullets(self):
        # The bullet character itself is removed, but the space after it
        # is preserved (matches the original client-side regex behaviour).
        assert clean_text_for_speech("- item one\n* item two\n• item three") == "item one\n item two\n item three"

    def test_plain_text_unchanged(self):
        assert clean_text_for_speech("Blood pressure is normal.") == "Blood pressure is normal."

    def test_strips_surrounding_whitespace(self):
        assert clean_text_for_speech("   hello   ") == "hello"

    # ── RAW_HTML block handling (report_generator.py embeds these) ──
    # Regression coverage for the bug where Piper read the literal
    # markup aloud (e.g. "html div 40 px") because the RAW_HTML_START/
    # END blocks emitted by tools/report_generator.py were being sent
    # to TTS verbatim instead of having their tags/styles stripped.

    def test_strips_raw_html_block_keeps_inner_text(self):
        text = (
            "## Findings\n\n"
            "<!--RAW_HTML_START-->"
            '<div style="padding: 40px; margin: 10px;">'
            '<p style="color:#333">Blood pressure: <strong>140/90 mmHg</strong> elevated</p>'
            "</div>"
            "<!--RAW_HTML_END-->\n\n"
            "Some **bold** text after."
        )
        cleaned = clean_text_for_speech(text)
        assert "div" not in cleaned.lower()
        assert "px" not in cleaned.lower()
        assert "style=" not in cleaned
        assert "<" not in cleaned and ">" not in cleaned
        assert "Blood pressure: 140/90 mmHg elevated" in cleaned
        assert "Some bold text after." in cleaned

    def test_strips_multiple_raw_html_blocks(self):
        text = (
            "<!--RAW_HTML_START--><div>First block</div><!--RAW_HTML_END-->\n"
            "<!--RAW_HTML_START--><div>Second block</div><!--RAW_HTML_END-->"
        )
        cleaned = clean_text_for_speech(text)
        assert "First block" in cleaned
        assert "Second block" in cleaned
        assert "<" not in cleaned

    def test_strips_html_entities_inside_raw_html_block(self):
        text = "<!--RAW_HTML_START--><p>A &amp; B &mdash; C</p><!--RAW_HTML_END-->"
        cleaned = clean_text_for_speech(text)
        assert "&amp;" not in cleaned
        assert "A & B" in cleaned

    def test_strips_stray_html_outside_raw_html_markers(self):
        # Defensive: even without RAW_HTML markers, tags should never
        # be read aloud if they slip into the text some other way.
        text = "Summary: <span style='color:red'>abnormal</span> result."
        cleaned = clean_text_for_speech(text)
        assert "<" not in cleaned and ">" not in cleaned
        assert "abnormal" in cleaned

    def test_block_tags_dont_glue_words_together(self):
        text = "<!--RAW_HTML_START--><div>Alpha</div><div>Beta</div><!--RAW_HTML_END-->"
        cleaned = clean_text_for_speech(text)
        assert "AlphaBeta" not in cleaned
        assert "Alpha" in cleaned and "Beta" in cleaned


# ── _get_model_dir / _get_voice_name ─────────────────────────────

class TestGetModelDir:
    def test_reads_current_settings_value(self):
        # Reflects whatever app/settings.py currently defines — this is
        # the single source of truth now, so we assert against it
        # directly rather than hardcoding a path here.
        assert _get_model_dir() == Path(settings.AEGIS_TTS_MODEL_DIR)

    def test_patched_settings_value(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "AEGIS_TTS_MODEL_DIR", str(tmp_path))
        assert _get_model_dir() == tmp_path


class TestGetVoiceName:
    def test_reads_current_settings_value(self):
        assert _get_voice_name() == settings.AEGIS_TTS_VOICE_NAME

    def test_patched_settings_value(self, monkeypatch):
        monkeypatch.setattr(settings, "AEGIS_TTS_VOICE_NAME", "en_GB-alan-medium")
        assert _get_voice_name() == "en_GB-alan-medium"


# ── _resolve_model_files ──────────────────────────────────────────

class TestResolveModelFiles:
    def test_missing_files_raises_model_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "AEGIS_TTS_MODEL_DIR", str(tmp_path / "does-not-exist"))
        with pytest.raises(TTSError) as exc_info:
            _resolve_model_files()
        assert exc_info.value.reason == "model_missing"

    def test_only_onnx_present_still_missing(self, tmp_path, monkeypatch):
        d = tmp_path / "voice"
        d.mkdir()
        (d / f"{settings.AEGIS_TTS_VOICE_NAME}.onnx").write_bytes(b"x")
        monkeypatch.setattr(settings, "AEGIS_TTS_MODEL_DIR", str(d))
        with pytest.raises(TTSError) as exc_info:
            _resolve_model_files()
        assert exc_info.value.reason == "model_missing"

    def test_both_files_present_resolves(self, model_dir):
        onnx_path, config_path = _resolve_model_files()
        assert onnx_path.exists()
        assert config_path.exists()


# ── synthesize_speech — input guards ──────────────────────────────

class TestSynthesizeSpeechGuards:
    def test_empty_text_raises(self):
        with pytest.raises(TTSError) as exc_info:
            synthesize_speech("")
        assert exc_info.value.reason == "synthesis_failed"

    def test_whitespace_only_text_raises(self):
        with pytest.raises(TTSError) as exc_info:
            synthesize_speech("   \n\t  ")
        assert exc_info.value.reason == "synthesis_failed"

    def test_markdown_only_text_raises_after_cleanup(self):
        # "- " strips to nothing once bullets/markdown are removed.
        with pytest.raises(TTSError) as exc_info:
            synthesize_speech("- ")
        assert exc_info.value.reason == "synthesis_failed"

    def test_text_over_max_chars_raises_text_too_long(self, model_dir):
        long_text = "a" * 100
        with pytest.raises(TTSError) as exc_info:
            synthesize_speech(long_text, max_chars=50)
        assert exc_info.value.reason == "text_too_long"

    def test_model_missing_propagates_as_model_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "AEGIS_TTS_MODEL_DIR", str(tmp_path / "nope"))
        with pytest.raises(TTSError) as exc_info:
            synthesize_speech("hello")
        assert exc_info.value.reason == "model_missing"


# ── synthesize_speech — success path (mocked PiperVoice) ──────────

class TestSynthesizeSpeechSuccess:
    def test_returns_nonempty_wav_bytes(self, model_dir):
        with patch("piper.PiperVoice") as MockPiperVoice:
            MockPiperVoice.load.return_value = _mock_voice()
            result = synthesize_speech("Your vitals are within normal range.")
        assert isinstance(result, bytes)
        assert len(result) > 0
        # RIFF/WAVE header present — this is real WAV framing, not raw bytes.
        assert result[:4] == b"RIFF"
        assert result[8:12] == b"WAVE"

    def test_voice_loaded_once_across_multiple_calls(self, model_dir):
        with patch("piper.PiperVoice") as MockPiperVoice:
            MockPiperVoice.load.return_value = _mock_voice()
            synthesize_speech("First call.")
            synthesize_speech("Second call.")
        # PiperVoice.load should only be invoked once — singleton reuse.
        assert MockPiperVoice.load.call_count == 1

    def test_markdown_stripped_before_synthesis(self, model_dir):
        with patch("piper.PiperVoice") as MockPiperVoice:
            mock_voice = _mock_voice()
            MockPiperVoice.load.return_value = mock_voice
            synthesize_speech("## Heading\n**bold** text")
        called_text = mock_voice.synthesize_wav.call_args[0][0]
        assert "#" not in called_text
        assert "*" not in called_text


# ── synthesize_speech — failure paths ──────────────────────────────

class TestSynthesizeSpeechFailures:
    def test_piper_not_installed_raises_model_missing(self, model_dir):
        with patch.dict("sys.modules", {"piper": None}):
            with pytest.raises(TTSError) as exc_info:
                synthesize_speech("hello")
            assert exc_info.value.reason == "model_missing"

    def test_voice_load_failure_raises_tts_error(self, model_dir):
        with patch("piper.PiperVoice") as MockPiperVoice:
            MockPiperVoice.load.side_effect = RuntimeError("corrupt model file")
            with pytest.raises(TTSError):
                synthesize_speech("hello")

    def test_synthesize_wav_failure_raises_tts_error(self, model_dir):
        with patch("piper.PiperVoice") as MockPiperVoice:
            broken_voice = MagicMock()
            broken_voice.synthesize_wav.side_effect = RuntimeError("inference failed")
            MockPiperVoice.load.return_value = broken_voice
            with pytest.raises(TTSError) as exc_info:
                synthesize_speech("hello")
            assert exc_info.value.reason == "synthesis_failed"


# ── Integration (real Piper model — requires model files) ─────────

@pytest.mark.piper
class TestSynthesizeSpeechIntegration:
    """
    Runs against the real Piper voice loaded from AEGIS_TTS_MODEL_DIR.
    Not run in standard CI — requires the voice model to be downloaded
    first (see tools/tts_synthesizer.py module docstring).

    Run with: pytest -m piper
    """

    def test_real_synthesis_produces_audible_wav(self):
        wav_bytes = synthesize_speech("This is a text to speech test.")
        assert len(wav_bytes) > 1000
        assert wav_bytes[:4] == b"RIFF"