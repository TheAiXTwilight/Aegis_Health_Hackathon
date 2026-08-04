"""
tests/tools/test_uploads.py — backend/uploads.py validators.

Tests the pre-queue upload validation layer. All validators return
ToolError(fatal=True) on violation or None on acceptance.

File creation uses tmp_path (pytest built-in). No network calls.
No mocking — all validators are pure functions over file contents.
"""

from __future__ import annotations

import wave
from pathlib import Path


from backend.uploads import (
    MAX_AUDIO_BYTES,
    MAX_AUDIO_SECONDS,
    MAX_MEDICATIONS,
    MAX_PDF_BYTES,
    MAX_XRAY_BYTES,
    validate_audio,
    validate_lab_pdf,
    validate_medications,
    validate_xray_image,
)
from schemas.errors import ToolError


# ── Helpers ────────────────────────────────────────────────────────

def _make_file(
    tmp_path: Path,
    name: str,
    size: int,
    content: bytes = b"x",
) -> str:
    p = tmp_path / name
    data = (content * (size // len(content) + 1))[:size]
    p.write_bytes(data)
    return str(p)


def _make_wav(
    tmp_path: Path,
    duration_s: float,
    name: str = "audio.wav",
) -> str:
    """Create a real WAV file of the given duration (silence)."""
    p = tmp_path / name
    framerate = 16000
    n_frames = int(framerate * duration_s)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(framerate)
        w.writeframes(b"\x00\x00" * n_frames)
    return str(p)


def _make_riff_truncated(
    tmp_path: Path,
    name: str = "trunc.wav",
) -> str:
    """RIFF/WAVE magic bytes but no actual audio data — truncated WAV."""
    p = tmp_path / name
    p.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    return str(p)


# ── validate_lab_pdf ───────────────────────────────────────────────

def test_validate_lab_pdf_accepts_small_file(tmp_path):
    path = _make_file(tmp_path, "lab.pdf", 100)
    assert validate_lab_pdf(path) is None


def test_validate_lab_pdf_accepts_at_limit(tmp_path):
    path = _make_file(tmp_path, "lab.pdf", MAX_PDF_BYTES)
    assert validate_lab_pdf(path) is None


def test_validate_lab_pdf_rejects_missing_file():
    result = validate_lab_pdf("/nonexistent/lab.pdf")
    assert isinstance(result, ToolError)
    assert result.fatal is True


def test_validate_lab_pdf_accepts_any_readable_file(tmp_path):
    """
    validate_lab_pdf is a readability guard only after M3.
    Size is enforced during streaming in _save_upload().
    A file of any size that exists and is readable must pass.
    """
    path = _make_file(tmp_path, "lab.pdf", MAX_PDF_BYTES + 100)
    assert validate_lab_pdf(path) is None

# ── validate_xray_image ────────────────────────────────────────────

def test_validate_xray_image_accepts_small_file(tmp_path):
    path = _make_file(tmp_path, "xray.jpg", 100)
    assert validate_xray_image(path) is None


def test_validate_xray_image_accepts_at_limit(tmp_path):
    path = _make_file(tmp_path, "xray.jpg", MAX_XRAY_BYTES)
    assert validate_xray_image(path) is None


def test_validate_xray_image_rejects_missing_file():
    result = validate_xray_image("/nonexistent/xray.jpg")
    assert isinstance(result, ToolError)
    assert result.fatal is True


def test_validate_xray_image_accepts_any_readable_file(tmp_path):
    """
    validate_xray_image is a readability guard only after M3.
    Size is enforced during streaming in _save_upload().
    A file of any size that exists and is readable must pass.
    """
    path = _make_file(tmp_path, "xray.jpg", MAX_XRAY_BYTES + 100)
    assert validate_xray_image(path) is None


# ── validate_audio — size limit ────────────────────────────────────

def test_validate_audio_accepts_small_nonwav_file(tmp_path):
    """Non-WAV files under size limit are accepted; duration deferred."""
    path = _make_file(tmp_path, "audio.mp3", 100, content=b"\x00")
    assert validate_audio(path) is None


def test_validate_audio_non_wav_oversized_file_defers_size_to_streaming_save(tmp_path):
    """Post-write validation checks WAV duration/format only.

    _save_upload() in backend.main enforces the byte limit while the multipart
    body is streamed. A non-WAV file reaching this validator therefore defers
    duration/format handling to VoiceTranscriber instead of duplicating the
    retired file-size check.
    """
    path = _make_file(tmp_path, "audio.bin", MAX_AUDIO_BYTES + 1)
    assert validate_audio(path) is None


def test_validate_audio_rejects_missing_file():
    result = validate_audio("/nonexistent/audio.wav")
    assert isinstance(result, ToolError)
    assert result.fatal is True


# ── validate_audio — WAV duration ─────────────────────────────────

def test_validate_audio_accepts_wav_under_limit(tmp_path):
    path = _make_wav(tmp_path, duration_s=10.0)
    assert validate_audio(path) is None


def test_validate_audio_accepts_wav_at_limit(tmp_path):
    path = _make_wav(tmp_path, duration_s=float(MAX_AUDIO_SECONDS))
    assert validate_audio(path) is None


def test_validate_audio_rejects_wav_over_limit(tmp_path):
    path = _make_wav(tmp_path, duration_s=float(MAX_AUDIO_SECONDS) + 1.0)
    result = validate_audio(path)
    assert isinstance(result, ToolError)
    assert result.fatal is True


def test_validate_audio_duration_error_mentions_limit(tmp_path):
    path = _make_wav(tmp_path, duration_s=float(MAX_AUDIO_SECONDS) + 10.0)
    result = validate_audio(path)
    assert isinstance(result, ToolError)
    assert (
        str(MAX_AUDIO_SECONDS) in result.reason
        or "duration" in result.reason.lower()
    )


# ── validate_audio — truncated WAV ────────────────────────────────

def test_validate_audio_rejects_truncated_wav(tmp_path):
    """RIFF/WAVE magic present but file truncated → ToolError."""
    path = _make_riff_truncated(tmp_path)
    result = validate_audio(path)
    assert isinstance(result, ToolError)
    assert result.fatal is True


def test_validate_audio_non_wav_binary_defers_duration(tmp_path):
    """
    A binary file with no RIFF/WAVE header is treated as non-WAV.
    Duration check deferred to VoiceTranscriber — returns None here.
    """
    path = _make_file(tmp_path, "audio.ogg", 100, content=b"\xff\xfb")
    assert validate_audio(path) is None


# ── validate_medications ───────────────────────────────────────────

def test_validate_medications_accepts_empty_list():
    assert validate_medications([]) is None


def test_validate_medications_accepts_under_limit():
    assert validate_medications(["drug"] * (MAX_MEDICATIONS - 1)) is None


def test_validate_medications_accepts_at_limit():
    assert validate_medications(["drug"] * MAX_MEDICATIONS) is None


def test_validate_medications_rejects_over_limit():
    result = validate_medications(["drug"] * (MAX_MEDICATIONS + 1))
    assert isinstance(result, ToolError)
    assert result.fatal is True


def test_validate_medications_error_mentions_limit():
    result = validate_medications(["drug"] * (MAX_MEDICATIONS + 1))
    assert isinstance(result, ToolError)
    assert str(MAX_MEDICATIONS) in result.reason
