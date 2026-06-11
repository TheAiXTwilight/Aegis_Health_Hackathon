from __future__ import annotations

from pydantic import BaseModel, Field


class VoiceTranscriptionResult(BaseModel):
    """
    Output of VoiceTranscriber (Step 0).

    Writes raw_symptoms_text into AegisState. No other fields required
    by the current spec.
    """

    transcript: str = Field(
        description=(
            "Transcribed symptom text. "
            "Written to AegisState.raw_symptoms_text by VoiceTranscriber."
        ),
    )

    schema_version: str = "1.0"