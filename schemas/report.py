from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TriageReport(BaseModel):
    """
    Final output of the Aegis pipeline (Step 7 — ReportGenerator).
    Streamed to the UI.

    Pure data contract. Confidence calculation lives in tools/confidence.py
    to keep schemas free of business logic.
    """

    severity: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        description="Matches the level from SeverityResult."
    )

    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Pipeline confidence score. Logic in tools/confidence.py."
    )

    text: str = Field(
        description="The generated markdown report containing all six required sections."
    )

    citations: list[str] = Field(
        default_factory=list,
        description="Formatted MedlinePlus citations used in the Evidence section."
    )

    disclaimer: str = Field(
        description="Standard medical disclaimer appended to the report."
    )

    knowledge_base_version: str | None = Field(
        default=None,
        description="Git commit hash or version identifier of the RAG corpus."
    )

    knowledge_base_date: str | None = Field(
        default=None,
        description="Snapshot date of the RAG corpus."
    )

    schema_version: str = "1.0"