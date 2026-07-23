"""
backend/uploads.py — Input bound enforcement.

Runs BEFORE a job enters the queue. Rejects inputs that exceed declared
limits without spinning up the pipeline.

Returns ToolError(fatal=True) on violation, None on acceptance.

Size enforcement:
    File size limits are enforced during streaming in _save_upload()
    in backend/main.py. The validators here handle constraints that
    cannot be checked during the write:
        validate_audio()      — WAV duration and format
        validate_lab_pdf()    — file readability guard (supports single str or list[str])
        validate_xray_image() — file readability guard (supports single str or list[str])
        validate_medications()— entry count on deserialized JSON list

    validate_lab_pdf() and validate_xray_image() are retained as
    readability guards. If _save_upload() succeeds but the file is
    somehow unreadable afterward, they surface the failure before
    the pipeline starts.

Audio duration handling:
    All audio is subject to the 120-second duration limit.
    - WAV files: validated pre-queue when readable (fast rejection)
    - Malformed or truncated WAV: rejected pre-queue with a clear message
    - Non-WAV: duration check deferred to VoiceTranscriber (definitive)
"""
from __future__ import annotations

import wave
from pathlib import Path
from loguru import logger

from schemas.errors import ToolError
from tools.tool_names import TOOL_INPUT_VALIDATION


# ── Limits (exposed for tests, docs, and save_upload) ───────────
MAX_PDF_BYTES     = 25 * 1024 * 1024    # 25 MB
MAX_XRAY_BYTES    = 25 * 1024 * 1024    # 25 MB
MAX_AUDIO_BYTES   = 15 * 1024 * 1024    # 15 MB
MAX_AUDIO_SECONDS = 120                 # 2 minutes
MAX_MEDICATIONS   = 50


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
    return (
        len(header) >= 12
        and header[:4] == b"RIFF"
        and header[8:12] == b"WAVE"
    )


def _reject(
    reason: str,
    *,
    code: str,
    **log_context: object,
) -> ToolError:
    """Build a fatal ToolError and log the rejection."""
    err = ToolError(
        tool=TOOL_INPUT_VALIDATION,
        code=code,
        reason=reason,
        fatal=True,
    )
    logger.warning(
        "Upload rejected",
        code=code,
        reason=reason,
        **log_context,
    )
    return err


# ── Validators ───────────────────────────────────────────────────

def validate_lab_pdf(path: str | list[str]) -> ToolError | None:
    """
    Confirm the saved lab PDF(s) are readable.

    Size is enforced during streaming in _save_upload(). This validator
    is retained as a readability guard — if the file exists but cannot
    be stat'd after saving, it surfaces the failure before pipeline start.
    """
    if isinstance(path, list):
        for p in path:
            if err := validate_lab_pdf(p):
                return err
        return None

    try:
        _file_size(path)
    except FileNotFoundError as e:
        return _reject(
            f"Cannot read lab PDF: {e}",
            code="file_not_found",
            input="lab_pdf",
            path=path,
        )
    except OSError as e:
        return _reject(
            f"Cannot read lab PDF: {e}",
            code="internal_error",
            input="lab_pdf",
            path=path,
        )
    return None


def validate_xray_image(path: str | list[str]) -> ToolError | None:
    """
    Confirm the saved X-ray image(s) are readable.

    Size is enforced during streaming in _save_upload(). This validator
    is retained as a readability guard — if the file exists but cannot
    be stat'd after saving, it surfaces the failure before pipeline start.
    """
    if isinstance(path, list):
        for p in path:
            if err := validate_xray_image(p):
                return err
        return None

    try:
        _file_size(path)
    except FileNotFoundError as e:
        return _reject(
            f"Cannot read X-ray image: {e}",
            code="file_not_found",
            input="xray_image",
            path=path,
        )
    except OSError as e:
        return _reject(
            f"Cannot read X-ray image: {e}",
            code="internal_error",
            input="xray_image",
            path=path,
        )
    return None


def validate_audio(path: str) -> ToolError | None:
    """
    Validate WAV duration and format for a saved audio file.

    Size is enforced during streaming in _save_upload(). This validator
    handles constraints that cannot be checked during the write:
        - WAV duration (requires parsing the RIFF header)
        - Malformed WAV detection (RIFF magic present but parser fails)
        - Non-WAV format deferral (duration checked by VoiceTranscriber)
    """
    try:
        _file_size(path)
    except FileNotFoundError as e:
        return _reject(
            f"Cannot read audio file: {e}",
            code="file_not_found",
            input="audio",
            path=path,
        )
    except OSError as e:
        return _reject(
            f"Cannot read audio file: {e}",
            code="internal_error",
            input="audio",
            path=path,
        )

    looks_like_wav = _is_wav_header(path)

    try:
        with wave.open(path, "rb") as w:
            frames   = w.getnframes()
            rate     = w.getframerate()
            duration = frames / rate if rate else 0.0

    except (wave.Error, EOFError) as e:
        if looks_like_wav:
            return _reject(
                f"Audio file appears to be WAV but is malformed or truncated: {e}",
                code="unsupported_format",
                input="audio",
                path=path,
            )
        logger.info(
            "Audio duration check deferred (non-WAV format)",
            input="audio",
            path=path,
        )
        return None

    except OSError as e:
        return _reject(
            f"Cannot read audio file during WAV parse: {e}",
            code="internal_error",
            input="audio",
            path=path,
        )

    if duration > MAX_AUDIO_SECONDS:
        return _reject(
            f"Audio exceeds duration limit: {duration:.1f}s "
            f"(max {MAX_AUDIO_SECONDS}s)",
            code="invalid_input",
            input="audio",
            path=path,
            duration=duration,
            limit=MAX_AUDIO_SECONDS,
        )
    return None


def validate_medications(medications: list[str]) -> ToolError | None:
    """Reject if medication list exceeds MAX_MEDICATIONS entries."""
    count = len(medications)
    if count > MAX_MEDICATIONS:
        return _reject(
            f"Medication list exceeds entry limit: {count} entries "
            f"(max {MAX_MEDICATIONS})",
            code="invalid_input",
            input="medications",
            count=count,
            limit=MAX_MEDICATIONS,
        )
    return None