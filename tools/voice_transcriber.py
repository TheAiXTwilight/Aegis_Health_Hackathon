"""
tools/voice_transcriber.py — Voice transcription (Step 0).

Placeholder implementation. Real implementation uses Faster-Whisper
tiny.en INT8 without changing this module's public interface.

Changes from original:
    - WAV magic byte detection: returns ToolError for real WAV files.
    - Keeps state.raw_symptoms_text write (legitimate input handoff).
    - Removes state.voice_result assignment (pipeline owns state).
    - Uses TOOL_VOICE_TRANSCRIBER from tool_names.py.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from schemas.errors import ToolError
from schemas.state import AegisState
from schemas.voice import VoiceTranscriptionResult
from tools.tool_names import TOOL_VOICE_TRANSCRIBER


# ── WAV detection ─────────────────────────────────────────────────

_WAV_RIFF_MAGIC = b"RIFF"
_WAV_WAVE_MAGIC = b"WAVE"


def _is_real_wav(path: Path) -> bool:
    """
    Return True if file has RIFF....WAVE header indicating real WAV audio.
    Matches the same heuristic used in backend/uploads.py.
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


class VoiceTranscriber:
    """
    Transcribes audio and writes transcript to state.raw_symptoms_text.

    state.raw_symptoms_text is written here because VoiceTranscriber
    converts one input modality (audio) into another (text), which
    Step 1 (SymptomExtractor) then reads. This is the only tool
    permitted to write a non-result state field.

    Does not write state.voice_result — pipeline owns that assignment.
    """

    TOOL_NAME = TOOL_VOICE_TRANSCRIBER

    async def run(
        self,
        state: AegisState,
    ) -> VoiceTranscriptionResult | ToolError:

        try:
            if not state.audio_file_path:
                return ToolError(
                    tool=TOOL_VOICE_TRANSCRIBER,
                    reason="No audio file path supplied.",
                    fatal=False,
                )

            path = Path(state.audio_file_path)

            if not path.is_file():
                return ToolError(
                    tool=TOOL_VOICE_TRANSCRIBER,
                    reason=f"Audio file not found: {state.audio_file_path}",
                    fatal=False,
                )

            # Detect real WAV — placeholder cannot transcribe binary audio.
            if _is_real_wav(path):
                logger.warning(
                    "voice_transcriber · real WAV detected · "
                    "placeholder only supports text fixtures",
                    path=str(path),
                )
                return ToolError(
                    tool=TOOL_VOICE_TRANSCRIBER,
                    reason=(
                        "Audio transcription requires Faster-Whisper — "
                        "placeholder only supports text fixtures."
                    ),
                    fatal=False,
                )

            # Placeholder: read file as UTF-8 text (text fixtures only).
            transcript = path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).strip()

            if not transcript:
                return ToolError(
                    tool=TOOL_VOICE_TRANSCRIBER,
                    reason="Audio file produced empty transcript.",
                    fatal=False,
                )

            # Write raw_symptoms_text — legitimate input modality handoff.
            # Pipeline assigns state.voice_result from the returned result.
            state.raw_symptoms_text = transcript

            return VoiceTranscriptionResult(transcript=transcript)

        except Exception as exc:
            return ToolError(
                tool=TOOL_VOICE_TRANSCRIBER,
                reason=str(exc),
                fatal=False,
            )


async def transcribe(state: AegisState) -> VoiceTranscriptionResult | ToolError:
    """Canonical functional entrypoint."""
    return await VoiceTranscriber().run(state)