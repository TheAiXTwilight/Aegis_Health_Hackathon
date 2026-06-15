from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class DrugInteractionSeverity(str, Enum):
    """
    Severity classification for drug interactions.

    Used by SeverityScorer to determine:
        - RULE_SEVERE_DRUG_INTERACTION
        - RULE_MODERATE_DRUG_INTERACTION
    """

    SEVERE   = "severe"
    MODERATE = "moderate"
    MINOR    = "minor"


class DrugInteraction(BaseModel):
    """
    Structured representation of a drug interaction.

    Replaces the previous list[str] representation.
    """

    drugs: list[str] = Field(
        description="The drug names involved in the interaction."
    )

    severity: DrugInteractionSeverity = Field(
        description="Clinical severity classification of the interaction."
    )

    description: str = Field(
        description="Human-readable description of the interaction."
    )


class DrugInteractionResult(BaseModel):
    """
    Output of DrugInteractionChecker (Step 5).

    confidence = len(resolved) / (len(resolved) + len(unresolved))
    Zero resolved is valid data (0.0), not a ToolError.

    NOTE:
        interactions now contains structured DrugInteraction objects
        rather than plain strings. This is a breaking schema change
        accepted as intentional — internal pre-release project,
        no persisted serialized data, schema_version stays "1.0".
    """

    resolved: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)

    interactions: list[DrugInteraction] = Field(
        default_factory=list,
        description="Structured drug interaction objects.",
    )

    warnings: list[str] = Field(default_factory=list)

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Resolution confidence: resolved / total.",
    )

    schema_version: str = "1.0"