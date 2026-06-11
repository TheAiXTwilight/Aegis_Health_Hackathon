from __future__ import annotations

from pydantic import BaseModel, Field


class SymptomExtractionResult(BaseModel):
    """
    Output of SymptomExtractor (Step 1).

    Fields derived from the field-level truncation guarantees table in the spec.
    medical_entities and negations may be truncated; the rest are core fields.
    """

    symptoms: list[str] = Field(
        default_factory=list,
        description="Extracted symptom descriptions.",
    )

    duration: str | None = Field(
        default=None,
        description="Patient-reported duration, e.g. '3 days'. Core field — never truncated.",
    )

    severity_indicators: list[str] = Field(
        default_factory=list,
        description="Detected severity modifiers. Core field — never truncated.",
    )

    medical_entities: list[str] = Field(
        default_factory=list,
        description=(
            "Recognised medical entities (conditions, anatomy, etc.). "
            "Enrichment field — may be truncated under token budget pressure."
        ),
    )

    negations: list[str] = Field(
        default_factory=list,
        description=(
            "Negated symptom phrases. "
            "Enrichment field — may be truncated last under token budget pressure."
        ),
    )

    schema_version: str = "1.0"