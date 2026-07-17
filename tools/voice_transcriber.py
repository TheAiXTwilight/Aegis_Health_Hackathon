"""

tools/voice_transcriber.py — Voice transcription (Step 0).


Real implementation using Faster-Whisper tiny.en INT8.


Architecture:
    - Singleton WhisperModel (_MODEL), lazy-loaded on first call.
    - Model directory resolved via _get_model_dir() — single source of truth.
    - CPU + INT8 only (Phase 3). CUDA detection deferred to Phase 4.
    - Input: WAV preferred; non-WAV (WebM/MP4) converted via ffmpeg automatically.
    - WAV-only validation (RIFF/WAVE magic byte) after conversion.
    - All failure modes return ToolError(fatal=False) — never raise.
    - state.raw_symptoms_text written on success (legitimate modality handoff).
    - state.voice_result NOT written — pipeline owns that assignment.


Environment variables:
    AEGIS_WHISPER_DIR   Path to Faster-Whisper model directory.
                        Default: data/audio/whisper-tiny-en/


Decisions (Phase 3 — locked):
    device       = "cpu"
    compute_type = "int8"
    language     = "en"
    beam_size    = 5

"""


from __future__ import annotations


import os
from pathlib import Path
from typing import TYPE_CHECKING


from loguru import logger


from schemas.errors import ToolError
from schemas.state import AegisState
from schemas.voice import VoiceTranscriptionResult
from tools.tool_names import TOOL_VOICE_TRANSCRIBER


if TYPE_CHECKING:
    from faster_whisper import WhisperModel as _WhisperModelType


# ── Constants ──────────────────────────────────────────────────────


_WAV_RIFF_MAGIC = b"RIFF"
_WAV_WAVE_MAGIC = b"WAVE"


_DEFAULT_WHISPER_DIR = Path("data/audio/whisper-tiny-en")
_WHISPER_DIR_ENV     = "AEGIS_WHISPER_DIR"


_DEVICE       = "cpu"
_COMPUTE_TYPE = "int8"
_LANGUAGE     = "en"
_BEAM_SIZE    = 5


def _ensure_wav(path: Path) -> Path:
    """Convert non-WAV audio to WAV via ffmpeg. Returns WAV path."""
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return path  # already WAV

    wav_path = path.with_suffix(".wav")
    # --- FIX FOR MACOS PATH ---
    # Manually add common Homebrew paths so FFmpeg is found
    env = os.environ.copy()
    env["PATH"] = env.get("PATH", "") + ":/opt/homebrew/bin:/usr/local/bin"

    try:
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path),
             "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
             str(wav_path)],
            capture_output=True, timeout=30, check=True, env=env  # Pass the custom env
        )
        return wav_path
    except Exception as exc:
        logger.warning(f"voice_transcriber · ffmpeg conversion failed: {exc}")
        return path


# ── Singleton ──────────────────────────────────────────────────────


_MODEL: _WhisperModelType | None = None


def _get_model_dir() -> Path:
    """
    Single source of truth for model directory resolution.

    Reads AEGIS_WHISPER_DIR; falls back to data/audio/whisper-tiny-en/.
    Relative paths are resolved relative to the working directory
    (project root in normal operation).
    """
    raw = os.environ.get(_WHISPER_DIR_ENV, "")
    if raw.strip():
        return Path(raw.strip())
    return _DEFAULT_WHISPER_DIR


def _load_model() -> _WhisperModelType:
    """
    Lazy-load the WhisperModel singleton.

    Called once on first transcription request.
    Subsequent calls return the cached instance.

    Raises:
        Exception — propagates to caller, which converts to ToolError.
    """
    global _MODEL

    if _MODEL is None:
        from faster_whisper import WhisperModel

        model_dir = _get_model_dir()

        logger.info(
            "voice_transcriber · loading Faster-Whisper model",
            model_dir=str(model_dir),
            device=_DEVICE,
            compute_type=_COMPUTE_TYPE,
        )

        _MODEL = WhisperModel(
            str(model_dir),  # Points to data/audio/whisper-tiny-en
            device=_DEVICE,
            compute_type=_COMPUTE_TYPE,
            local_files_only=True  # Force it to use the local files we just downloaded
        )

        logger.info("voice_transcriber · model loaded")

    return _MODEL


# ── WAV detection ──────────────────────────────────────────────────


def _is_real_wav(path: Path) -> bool:
    """
    Return True if file has RIFF....WAVE header (valid WAV audio).

    Reads first 12 bytes only. Matches the same heuristic used in
    backend/uploads.py. Returns False on any OSError.
    """
    try:
        with open(path, "rb") as f:
            header = f.read(12)
        return (
            len(header) >= 12
            and header[:4] == _WAV_RIFF_MAGIC
            and header[8:12] == _WAV_WAVE_MAGIC
        )
    except OSError:
        return False


# ── Transcription ──────────────────────────────────────────────────


def _transcribe_wav(path: Path) -> str:
    """
    Run Faster-Whisper on a WAV file. Returns stripped transcript string.

    May return empty string if Whisper produces no segments (silence,
    noise, very short clips). Caller converts empty string to ToolError.

    Raises:
        Exception — any model or I/O error propagates to caller.
    """
    model = _load_model()

    segments, _info = model.transcribe(
        str(path),
        language=_LANGUAGE,
        beam_size=_BEAM_SIZE,
    )

    transcript = " ".join(seg.text.strip() for seg in segments).strip()
    return transcript


# ── Tool ───────────────────────────────────────────────────────────


class VoiceTranscriber:
    """
    Transcribes a WAV audio file and writes transcript to
    state.raw_symptoms_text for downstream SymptomExtractor consumption.

    Contract:
        - Only tool permitted to write state.raw_symptoms_text.
        - Does NOT write state.voice_result (pipeline owns that).
        - Returns ToolError(fatal=False) on all failure modes.
        - Never raises.
    """

    TOOL_NAME = TOOL_VOICE_TRANSCRIBER

    async def run(
        self,
        state: AegisState,
    ) -> VoiceTranscriptionResult | ToolError:

        try:
            # ── Guard: path provided ───────────────────────────────
            if not state.audio_file_path:
                return ToolError(
                    tool=TOOL_VOICE_TRANSCRIBER,
                    reason="No audio file path supplied.",
                    fatal=False,
                )

            path = Path(state.audio_file_path)

            # ── Guard: file exists ─────────────────────────────────
            if not path.is_file():
                return ToolError(
                    tool=TOOL_VOICE_TRANSCRIBER,
                    reason=f"Audio file not found: {state.audio_file_path}",
                    fatal=False,
                )

            # ── Guard: must be real WAV ────────────────────────────
            path = _ensure_wav(path)
            if not _is_real_wav(path):
                logger.warning(
                    "voice_transcriber · file rejected (not a valid WAV)",
                    path=str(path),
                    session_id=state.session_id,
                )
                return ToolError(
                    tool=TOOL_VOICE_TRANSCRIBER,
                    reason=(
                        f"File is not a valid WAV (RIFF/WAVE header not found): "
                        f"{path.name}"
                    ),
                    fatal=False,
                )

            # ── Transcribe ─────────────────────────────────────────
            logger.info(
                "voice_transcriber · transcribing",
                path=str(path),
                session_id=state.session_id,
            )

            transcript = _transcribe_wav(path)

            # ── Guard: non-empty result ────────────────────────────
            if not transcript:
                logger.warning(
                    "voice_transcriber · empty transcript",
                    path=str(path),
                    session_id=state.session_id,
                )
                return ToolError(
                    tool=TOOL_VOICE_TRANSCRIBER,
                    reason="Audio file produced empty transcript.",
                    fatal=False,
                )

            # ── Success: write modality handoff field ──────────────
            # VoiceTranscriber is the only tool permitted to write
            # state.raw_symptoms_text. This converts audio input to
            # the text form that SymptomExtractor (Step 1) reads.
            state.raw_symptoms_text = transcript

            logger.info(
                "voice_transcriber · success",
                transcript_len=len(transcript),
                session_id=state.session_id,
            )

            return VoiceTranscriptionResult(transcript=transcript)

        # Replace the catch block at the end of the run() method:
        except Exception as exc:
            logger.exception(  # Changed from .error to .exception
                "voice_transcriber · unexpected error",
                session_id=state.session_id,
            )
            return ToolError(
                tool=TOOL_VOICE_TRANSCRIBER,
                reason=f"Detailed Error: {exc}",  # This will show in your UI
                fatal=False,
            )


async def transcribe(state: AegisState) -> VoiceTranscriptionResult | ToolError:
    """Canonical functional entrypoint."""
    return await VoiceTranscriber().run(state)