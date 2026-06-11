from __future__ import annotations

from pydantic import BaseModel, Field


class DrugInteractionResult(BaseModel):
    """
    Output of DrugInteractionChecker (Step 5).

    confidence = len(resolved) / (len(resolved) + len(unresolved))
    Zero resolved is valid data (0.0), not a ToolError.
    """

    resolved: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    interactions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    schema_version: str = "1.0"