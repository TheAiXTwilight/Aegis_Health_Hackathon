from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SeverityResult(BaseModel):
    """
    Output of SeverityScorer. Rule-based, fully deterministic.

    Responsibility split:
        SeverityScorer    — produces correct values
        SeverityResult    — guarantees no invalid instance can exist

    Invariants (enforced by model_validator below):
        1. triggered_rules is guaranteed to contain at least one rule.
        2. highest_priority_rule == triggered_rules[0]
        3. len(reasons) == len(triggered_rules)

    Tests assert on rule constants from tools.severity_scorer.ALL_RULE_CONSTANTS,
    never on reasons strings — reasons wording may change without breaking tests.
    """

    level: Literal["LOW", "MEDIUM", "HIGH"]

    confidence: float = Field(ge=0.0, le=1.0)

    triggered_rules: list[str] = Field(
        min_length=1,
        description=(
            "All rule constants that fired, in descending priority order. "
            "Always contains at least one rule constant. "
            "Machine-readable. Maps to docs/severity_rules.md."
        ),
    )

    highest_priority_rule: str = Field(
        description=(
            "The single rule that determined the severity level. "
            "Always equals triggered_rules[0]. "
            "Set by SeverityScorer at scoring time — never recomputed downstream."
        ),
    )

    reasons: list[str] = Field(
        min_length=1,
        description=(
            "Human-readable explanations, one per triggered rule, in priority order. "
            "len(reasons) == len(triggered_rules) — enforced by validator. "
            "Wording may change without breaking tests."
        ),
    )

    contributing_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Tool names whose outputs contributed to at least one triggered rule."
        ),
    )

    schema_version: str = "1.0"

    @model_validator(mode="after")
    def _check_invariants(self) -> SeverityResult:
        """
        Defense-in-depth. Validates cross-field consistency only — never repairs.
        A malformed SeverityResult cannot exist as a constructed object.

        Single-field constraints (min_length, type, range) are handled by
        Pydantic's standard validation and not duplicated here.
        """
        if self.highest_priority_rule != self.triggered_rules[0]:
            raise ValueError(
                f"highest_priority_rule ({self.highest_priority_rule!r}) "
                f"must equal triggered_rules[0] ({self.triggered_rules[0]!r})"
            )

        if len(self.reasons) != len(self.triggered_rules):
            raise ValueError(
                f"len(reasons)={len(self.reasons)} must equal "
                f"len(triggered_rules)={len(self.triggered_rules)}"
            )

        return self