"""Updated tests for the worker-process TTS synthesizer.

The old test file targeted the retired in-process PiperVoice singleton
(_get_model_dir, _get_voice_name, _resolve_model_files, _VOICE). The current
implementation is a local HTTP client/lifecycle manager for tools.piper_server,
so these tests cover its pure helpers, worker delegation, streaming, and error
mapping without spawning Piper.
"""
from __future__ import annotations

import struct
from contextlib import contextmanager
from unittest.mock import Mock

import pytest

import tools.tts_synthesizer as tts
from tools.tts_synthesizer import TTSError


@pytest.fixture(autouse=True)
def reset_worker_state(monkeypatch):
    monkeypatch.setattr(tts, "_worker_proc", None)
    monkeypatch.setattr(tts, "_last_activity_ts", 0.0)
    monkeypatch.setattr(tts, "_health_cache_ts", 0.0)
    monkeypatch.setattr(tts, "_health_cache_val", False)


def test_clean_text_for_speech_strips_markdown_links_and_html():
    raw = (
        "## Summary\n"
        "- **Blood pressure** is [elevated](https://example.test).\n"
        "<!--RAW_HTML_START--><div>Potassium: <strong>6.2 mmol/L</strong></div><!--RAW_HTML_END-->"
    )
    cleaned = tts.clean_text_for_speech(raw)
    assert "#" not in cleaned
    assert "**" not in cleaned
    assert "https://" not in cleaned
    assert "<" not in cleaned and ">" not in cleaned
    assert "Blood pressure" in cleaned
    assert "Potassium: 6.2 mmol/L" in cleaned


def test_validate_text_rejects_empty_and_limit_exceeded_content():
    with pytest.raises(TTSError) as empty:
        tts._validate_text("  \n", max_chars=10)
    assert empty.value.reason == "synthesis_failed"

    with pytest.raises(TTSError) as long:
        tts._validate_text("a" * 11, max_chars=10)
    assert long.value.reason == "text_too_long"

    assert tts._validate_text("## hello", max_chars=10) == "hello"


def test_segment_report_text_adds_sentence_terminal_and_pause_semantics():
    segments = tts._segment_report_text("Summary\nFirst finding\n\nthe second finding!")
    assert segments[0][0] == "Summary."
    assert segments[0][1] == tts._HEADING_PAUSE_SECS
    assert segments[1][0] == "First finding."
    # Blank line updates the preceding segment to the section pause.
    assert segments[1][1] == tts._SECTION_PAUSE_SECS
    assert segments[2] == ("the second finding!", tts._LINE_PAUSE_SECS)


def test_streaming_wav_header_has_expected_riff_fields():
    header = tts._make_wav_header(22050)
    assert len(header) == 44
    assert header[:4] == b"RIFF"
    assert header[8:12] == b"WAVE"
    assert header[12:16] == b"fmt "
    assert header[36:40] == b"data"
    # Header fields are little endian and intentionally use unknown data length.
    assert struct.unpack("<I", header[4:8])[0] == 0xFFFFFFFF
    assert struct.unpack("<I", header[40:44])[0] == 0xFFFFFFFF


def test_worker_url_uses_configured_local_port(monkeypatch):
    monkeypatch.setenv("AEGIS_PIPER_PORT", "12345")
    assert tts._worker_port() == 12345
    assert tts._worker_url("/health") == "http://127.0.0.1:12345/health"


def test_health_probe_uses_short_cache(monkeypatch):
    response = Mock(status_code=200)
    response.json.return_value = {"ok": True}
    get = Mock(return_value=response)
    monkeypatch.setattr(tts.httpx, "get", get)

    assert tts._health(force=True) is True
    assert tts._health() is True
    assert get.call_count == 1


def test_synthesize_speech_delegates_to_ready_worker(monkeypatch):
    monkeypatch.setattr(tts, "ensure_worker", Mock())
    response = Mock(status_code=200, content=b"RIFF....WAVEfake")
    post = Mock(return_value=response)
    monkeypatch.setattr(tts.httpx, "post", post)

    audio = tts.synthesize_speech("Hello report", max_chars=123)
    assert audio == b"RIFF....WAVEfake"
    assert post.call_args.args[0].endswith("/synthesize")
    assert post.call_args.kwargs["json"] == {"text": "Hello report", "max_chars": 123}


def test_synthesize_speech_maps_worker_network_error_to_tts_error(monkeypatch):
    monkeypatch.setattr(tts, "ensure_worker", Mock())
    monkeypatch.setattr(tts.httpx, "post", Mock(side_effect=RuntimeError("network down")))

    with pytest.raises(TTSError) as error:
        tts.synthesize_speech("Hello")
    assert error.value.reason == "synthesis_failed"
    assert "speech synthesis failed" in str(error.value)


def test_synthesize_speech_maps_non_200_worker_response_to_tts_error(monkeypatch):
    monkeypatch.setattr(tts, "ensure_worker", Mock())
    response = Mock(status_code=503, text="model missing")
    monkeypatch.setattr(tts.httpx, "post", Mock(return_value=response))

    with pytest.raises(TTSError) as error:
        tts.synthesize_speech("Hello")
    assert error.value.reason == "synthesis_failed"
    assert "worker error 503" in str(error.value)


def test_streaming_synthesis_yields_nonempty_chunks_in_order(monkeypatch):
    monkeypatch.setattr(tts, "ensure_worker", Mock())

    class FakeStreamResponse:
        status_code = 200

        def iter_bytes(self):
            yield b"RIFF-header"
            yield b"pcm-1"
            yield b""
            yield b"pcm-2"

    @contextmanager
    def fake_stream(*args, **kwargs):
        yield FakeStreamResponse()

    monkeypatch.setattr(tts.httpx, "stream", fake_stream)
    chunks = list(tts.synthesize_speech_stream("Hello stream", max_chars=321))
    assert chunks == [b"RIFF-header", b"pcm-1", b"pcm-2"]


def test_streaming_synthesis_non_200_before_chunks_raises_tts_error(monkeypatch):
    monkeypatch.setattr(tts, "ensure_worker", Mock())

    class BadResponse:
        status_code = 503

        def iter_bytes(self):
            return iter(())

    @contextmanager
    def fake_stream(*args, **kwargs):
        yield BadResponse()

    monkeypatch.setattr(tts.httpx, "stream", fake_stream)
    with pytest.raises(TTSError) as error:
        list(tts.synthesize_speech_stream("Hello"))
    assert error.value.reason == "synthesis_failed"


def test_evict_voice_is_safe_when_no_worker_exists(monkeypatch):
    monkeypatch.setattr(tts, "_health", lambda force=False: False)
    assert tts.evict_voice() is False
    assert tts.is_loaded() is False
