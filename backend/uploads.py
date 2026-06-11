"""
backend/uploads.py — Input bound enforcement.

Runs BEFORE a job enters the queue. Rejects inputs that exceed declared
limits without spinning up the pipeline.

Returns ToolError(fatal=True) on violation, None on acceptance.

Audio duration handling:
    All audio is subject to the 120-second duration limit.
    - WAV files: validated pre-queue when readable (fast rejection)
    - Malformed WAV: rejected pre-queue with a clear message
    - Non-WAV: duration check deferred to VoiceTranscriber (definitive)
"""
from __future__ import annotations

import wave
from pathlib import Path

from loguru import logger

from schemas.errors import ToolError


# ── Limits (exposed for tests and docs) ──────────────────────────
MAX_PDF_BYTES     = 25 * 1024 * 1024    # 25 MB
MAX_XRAY_BYTES    = 25 * 1024 * 1024    # 25 MB
MAX_AUDIO_BYTES   = 15 * 1024 * 1024    # 15 MB
MAX_AUDIO_SECONDS = 120                  # 2 minutes
MAX_MEDICATIONS   = 50

_TOOL_NAME = "input_validation"


# ── Helpers ──────────────────────────────────────────────────────
def _file_size(path: str) -> int:
    """
    Return file size in bytes.

    Raises:
        FileNotFoundError: path does not exist or is not a regular file
        OSError:           filesystem-level read failure
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    return p.stat().st_size


def _is_wav_header(path: str) -> bool:
    """
    Heuristic check: does the file *claim* to be WAV via its RIFF/WAVE magic?

    NOT a validator. NOT a parser. NOT a security boundary.
    Used solely to route error attribution between two equally-safe outcomes:

        True  + wave parse fails  → reject as "malformed WAV"
        False + wave parse fails  → defer to VoiceTranscriber (likely non-WAV)

    Both paths are safe. This function only improves the user-facing message.
    """
    try:
        with open(path, "rb") as f:
            header = f.read(12)
    except OSError:
        return False
    return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE"


def _reject(reason: str, **log_context: object) -> ToolError:
    """Build a fatal ToolError and log the rejection."""
    err = ToolError(tool=_TOOL_NAME, reason=reason, fatal=True)
    logger.warning("Upload rejected", reason=reason, **log_context)
    return err


# ── Validators ───────────────────────────────────────────────────
def validate_lab_pdf(path: str) -> ToolError | None:
    """Reject lab PDFs larger than MAX_PDF_BYTES."""
    try:
        size = _file_size(path)
    except (FileNotFoundError, OSError) as e:
        return _reject(f"Cannot read lab PDF: {e}", input="lab_pdf", path=path)

    if size > MAX_PDF_BYTES:
        return _reject(
            f"Lab PDF exceeds size limit: {size} bytes (max {MAX_PDF_BYTES} bytes / 25 MB)",
            input="lab_pdf", path=path, size=size, limit=MAX_PDF_BYTES,
        )
    return None


def validate_xray_image(path: str) -> ToolError | None:
    """Reject X-ray images larger than MAX_XRAY_BYTES."""
    try:
        size = _file_size(path)
    except (FileNotFoundError, OSError) as e:
        return _reject(f"Cannot read X-ray image: {e}", input="xray_image", path=path)

    if size > MAX_XRAY_BYTES:
        return _reject(
            f"X-ray image exceeds size limit: {size} bytes (max {MAX_XRAY_BYTES} bytes / 25 MB)",
            input="xray_image", path=path, size=size, limit=MAX_XRAY_BYTES,
        )
    return None


def validate_audio(path: str) -> ToolError | None:
    """
    Reject audio larger than MAX_AUDIO_BYTES.
    Reject WAV audio longer than MAX_AUDIO_SECONDS.
    Reject malformed WAV files (RIFF header present but wave parser fails).

    Non-WAV files are accepted here; their duration is verified definitively
    by VoiceTranscriber (Step 0) after the file is loaded for transcription.
    """
    try:
        size = _file_size(path)
    except (FileNotFoundError, OSError) as e:
        return _reject(f"Cannot read audio file: {e}", input="audio", path=path)

    if size > MAX_AUDIO_BYTES:
        return _reject(
            f"Audio exceeds size limit: {size} bytes (max {MAX_AUDIO_BYTES} bytes / 15 MB)",
            input="audio", path=path, size=size, limit=MAX_AUDIO_BYTES,
        )

    looks_like_wav = _is_wav_header(path)
    try:
        with wave.open(path, "rb") as w:
            frames = w.getnframes()
            rate   = w.getframerate()
            duration = frames / rate if rate else 0.0
    except (wave.Error, OSError) as e:
        if looks_like_wav:
            return _reject(
                f"Audio file appears to be WAV but is malformed: {e}",
                input="audio", path=path,
            )
        logger.info(
            "Audio duration check deferred (non-WAV format)",
            input="audio", path=path,
        )
        return None

    if duration > MAX_AUDIO_SECONDS:
        return _reject(
            f"Audio exceeds duration limit: {duration:.1f}s (max {MAX_AUDIO_SECONDS}s)",
            input="audio", path=path, duration=duration, limit=MAX_AUDIO_SECONDS,
        )
    return None


def validate_medications(medications: list[str]) -> ToolError | None:
    """Reject if medication list exceeds MAX_MEDICATIONS entries."""
    count = len(medications)
    if count > MAX_MEDICATIONS:
        return _reject(
            f"Medication list exceeds entry limit: {count} entries (max {MAX_MEDICATIONS})",
            input="medications", count=count, limit=MAX_MEDICATIONS,
        )
    return None