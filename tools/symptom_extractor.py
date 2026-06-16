"""
tools/symptom_extractor.py — Symptom extraction (Step 1).

Placeholder implementation using rule-based regex extraction.
Real implementation uses llama3.2:1b with 3-attempt retry/repair
without changing this module's public interface.

Changes from original:
    - Removes internal state.symptom_result assignment (pipeline owns state).
    - Uses TOOL_SYMPTOM_EXTRACTOR from tool_names.py.
"""

from __future__ import annotations

import re

from schemas.errors import ToolError
from schemas.state import AegisState
from schemas.symptom import SymptomExtractionResult
from tools.tool_names import TOOL_SYMPTOM_EXTRACTOR


_DURATION_PATTERNS = [
    r"\d+\s*month[s]?",
    r"\d+\s*week[s]?",
    r"\d+\s*day[s]?",
    r"\d+\s*hour[s]?",
]

_SEVERITY_KEYWORDS: set[str] = {
    "severe", "mild", "moderate", "intense",
    "persistent", "chronic", "acute", "sharp",
}

_NEGATION_WORDS: set[str] = {
    "no", "not", "never", "without", "denies",
}

_MEDICAL_ENTITIES: set[str] = {
    "chest", "heart", "lung", "kidney", "liver",
    "head", "abdomen", "stomach", "blood",
    "fever", "cough", "pain",
}


class SymptomExtractor:
    """
    Rule-based symptom extractor.
    Does not write to state — pipeline owns state mutation.
    """

    TOOL_NAME = TOOL_SYMPTOM_EXTRACTOR

    async def run(
        self,
        state: AegisState,
    ) -> SymptomExtractionResult | ToolError:

        try:
            text = state.raw_symptoms_text or ""

            if (
                state.voice_result is not None
                and not isinstance(state.voice_result, ToolError)
                and state.voice_result.transcript
            ):
                text = state.voice_result.transcript

            text = text.strip().lower()

            if not text:
                return ToolError(
                    tool=TOOL_SYMPTOM_EXTRACTOR,
                    reason="No symptom text available for extraction.",
                    fatal=False,
                )

            # Duration — ordered longest unit first
            # (months > weeks > days > hours)
            duration: str | None = None
            for pattern in _DURATION_PATTERNS:
                match = re.search(pattern, text)
                if match:
                    duration = match.group(0)
                    break

            severity_indicators = [
                word for word in _SEVERITY_KEYWORDS if word in text
            ]

            negations = [
                word for word in text.split() if word in _NEGATION_WORDS
            ]

            medical_entities = [
                entity for entity in _MEDICAL_ENTITIES if entity in text
            ]

            segments = re.split(r",| and |;", text)
            symptoms = [
                s.strip() for s in segments
                if s.strip() and len(s.strip()) > 2
            ]

            return SymptomExtractionResult(
                symptoms=symptoms,
                duration=duration,
                severity_indicators=severity_indicators,
                medical_entities=medical_entities,
                negations=negations,
            )

        except Exception as exc:
            return ToolError(
                tool=TOOL_SYMPTOM_EXTRACTOR,
                reason=str(exc),
                fatal=False,
            )


async def extract(state: AegisState) -> SymptomExtractionResult | ToolError:
    """Canonical functional entrypoint."""
    return await SymptomExtractor().run(state)