from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TextToSpeechRequest(BaseModel):
    """Input for POST /tts/speak."""

    text: str = Field(
        min_length=1,
        description="Plain or lightly-formatted text to synthesize as speech.",
    )
    job_id: Optional[str] = Field(
        default=None,
        description=(
            "If provided, checked against the background TTS cache first "
            "(see backend/tts_cache.py) — synthesis kicked off in parallel "
            "with report generation may already be ready or in progress, "
            "avoiding a redundant re-synthesis."
        ),
    )


class TextToSpeechError(BaseModel):
    """Structured error body returned when synthesis cannot run."""

    detail: str
    reason: str = Field(
        description=(
            "Machine-readable reason code: 'model_missing', 'text_too_long', "
            "or 'synthesis_failed'."
        ),
    )