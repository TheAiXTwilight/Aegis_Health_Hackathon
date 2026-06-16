"""
tools/drug_checker.py — Drug interaction checker (Step 5).

Placeholder implementation using an in-memory interaction table.
Interface is stable — replace internals with SQLite FTS5 lookup
without changing the return type or pipeline integration.

Changes from original:
    - Returns structured DrugInteraction objects with severity enum.
    - Removes internal state.drug_result assignment (pipeline owns state).
    - Uses TOOL_DRUG_INTERACTION_CHECKER from tool_names.py.
    - Deduplicates resolved medications before pair generation.
"""

from __future__ import annotations

from schemas.drugs import (
    DrugInteraction,
    DrugInteractionResult,
    DrugInteractionSeverity,
)
from schemas.errors import ToolError
from schemas.state import AegisState
from tools.tool_names import TOOL_DRUG_INTERACTION_CHECKER


# ── Known drug vocabulary ─────────────────────────────────────────

KNOWN_DRUGS: set[str] = {
    "warfarin",
    "aspirin",
    "ibuprofen",
    "metformin",
    "contrast dye",
    "digoxin",
    "amiodarone",
}


# ── Interaction table ─────────────────────────────────────────────
# Maps frozenset of drug names → DrugInteraction.
# Severity assigned based on clinical significance.
# Replace with SQLite FTS5 lookup when aegis_drugs.db is ready.

_INTERACTIONS: dict[frozenset[str], DrugInteraction] = {
    frozenset({"warfarin", "aspirin"}): DrugInteraction(
        drugs=["warfarin", "aspirin"],
        severity=DrugInteractionSeverity.SEVERE,
        description="Warfarin + Aspirin significantly increases bleeding risk.",
    ),
    frozenset({"warfarin", "ibuprofen"}): DrugInteraction(
        drugs=["warfarin", "ibuprofen"],
        severity=DrugInteractionSeverity.SEVERE,
        description="Warfarin + Ibuprofen significantly increases bleeding risk.",
    ),
    frozenset({"aspirin", "ibuprofen"}): DrugInteraction(
        drugs=["aspirin", "ibuprofen"],
        severity=DrugInteractionSeverity.MODERATE,
        description="Concurrent NSAID use increases gastrointestinal bleeding risk.",
    ),
    frozenset({"metformin", "contrast dye"}): DrugInteraction(
        drugs=["metformin", "contrast dye"],
        severity=DrugInteractionSeverity.MODERATE,
        description="Contrast dye may increase lactic acidosis risk with Metformin.",
    ),
    frozenset({"digoxin", "amiodarone"}): DrugInteraction(
        drugs=["digoxin", "amiodarone"],
        severity=DrugInteractionSeverity.SEVERE,
        description=(
            "Amiodarone increases Digoxin concentration — "
            "narrow therapeutic index."
        ),
    ),
}


class DrugInteractionChecker:
    """
    Rule-based interaction checker.
    Does not write to state — pipeline owns state mutation.
    """

    TOOL_NAME = TOOL_DRUG_INTERACTION_CHECKER

    async def run(
        self,
        state: AegisState,
    ) -> DrugInteractionResult | ToolError:

        try:
            medications = [
                drug.strip().lower()
                for drug in state.medications_raw
                if drug.strip()
            ]

            resolved_raw: list[str] = []
            unresolved:   list[str] = []

            for drug in medications:
                if drug in KNOWN_DRUGS:
                    resolved_raw.append(drug)
                else:
                    unresolved.append(drug)

            # Deduplicate resolved while preserving insertion order.
            # Prevents duplicate pair evaluations and duplicate
            # DrugInteraction objects when the same drug appears
            # more than once in the input list.
            resolved = list(dict.fromkeys(resolved_raw))

            interactions: list[DrugInteraction] = []
            warnings:     list[str]             = []

            for i in range(len(resolved)):
                for j in range(i + 1, len(resolved)):
                    pair = frozenset({resolved[i], resolved[j]})
                    if pair in _INTERACTIONS:
                        interactions.append(_INTERACTIONS[pair])

            if unresolved:
                warnings.append(
                    f"{len(unresolved)} medication(s) could not be resolved: "
                    + ", ".join(unresolved)
                )

            if interactions:
                warnings.append(
                    f"{len(interactions)} potential drug interaction(s) detected."
                )

            total      = len(resolved) + len(unresolved)
            confidence = len(resolved) / total if total > 0 else 0.0

            return DrugInteractionResult(
                resolved=resolved,
                unresolved=unresolved,
                interactions=interactions,
                warnings=warnings,
                confidence=confidence,
            )

        except Exception as exc:
            return ToolError(
                tool=TOOL_DRUG_INTERACTION_CHECKER,
                reason=str(exc),
                fatal=False,
            )


async def check(state: AegisState) -> DrugInteractionResult | ToolError:
    """Canonical functional entrypoint."""
    return await DrugInteractionChecker().run(state)