# tools/chat_enricher.py
"""
Optional single-sentence enrichment for chat answers.

Architecture
────────────
The deterministic answer (from _generate_answer in chat.py) is ALWAYS
the floor — this layer only adds ONE connecting sentence for cross-domain
questions where deterministic code cannot reason across findings.

Anti-hallucination controls (every one is active simultaneously):
  1. Model receives ONLY to_prompt_block() — no raw JSON, no history
  2. Model receives the base answer — cannot contradict it
  3. num_predict=80 — physically caps output at ~one sentence
  4. temperature=0.0 — fully deterministic sampling
  5. Hard 10s timeout — chat must feel fast
  6. First line only — raw.split("\\n")[0] discards any second sentence
  7. SKIP detection — model instructed to output SKIP when nothing to add
  8. Overlap check — if 70%+ of enrichment words already in base answer,
     it adds nothing and is discarded
  9. Hallucination number check — any number in the enrichment not present
     in the intel block or base answer is grounds for discard
 10. Hedge phrase filter — vague filler sentences discarded
 11. Memory gate — assert_memory_headroom() skips enrichment if model is
     under memory pressure (pipeline always wins)
 12. Cross-reference gate — only enrichable intents are even attempted
     (single-domain questions are fully answered deterministically)
 13. Data gate — skipped when report lacks cross-referenceable data

The model never sees:
  - Conversation history (no drift)
  - Raw JSON (no opportunity to invent from noise)
  - User input alone (always anchored to the intel block)

Fixes vs. earlier version:
  - _ENRICHABLE_INTENTS now includes LIFESTYLE, TREND, RECHECK,
    SPECIALIST, RANGES. These are legitimately cross-domain questions
    that can benefit from a connecting sentence tying the general
    guidance to a specific report finding (e.g. lifestyle chip getting
    "This is especially relevant given your elevated HbA1c" appended).
    The 13 anti-hallucination gates still fully apply, so if the model
    can't ground a useful sentence it silently returns the base answer.
"""
from __future__ import annotations

import re

from loguru import logger

from tools.report_analyst import ReportIntelligence


# Intents that involve cross-domain reasoning — these are the ONLY ones
# worth enriching. Single-domain questions (what is my TSH?) are fully
# answered by the deterministic handler.
#
# The lifestyle/trend/recheck/specialist/ranges intents are included
# because their deterministic answers are grounded in report findings
# but written as generic guidance — a one-sentence enrichment can tie
# that guidance back to a specific abnormal lab or symptom for extra
# relevance. All 13 anti-hallucination gates still apply, so if the
# model can't ground a useful sentence the base answer is returned
# unchanged.
_ENRICHABLE_INTENTS = {
    "SEVERITY_REASON",
    "CONDITION_STATUS",
    "NEXT_STEPS",
    "CARDIAC",
    "RESPIRATORY",
    "NEURO",
    "SYMPTOM_GENERAL",
    "LIFESTYLE",
    "TREND",
    "RECHECK",
    "SPECIALIST",
    "RANGES",
}

_ENRICHMENT_PROMPT = """\
You are a clinical report explainer with one strictly limited job.

RULES — follow ALL of them without exception:
1. Output exactly ONE sentence (maximum 30 words).
2. Only use facts from REPORT DATA below — no outside knowledge.
3. Do not repeat anything already in BASE ANSWER.
4. Do not invent any number, medication, diagnosis, or finding.
5. Do not give dosage advice or treatment recommendations.
6. If you cannot add anything genuinely new from the report data, output: SKIP
7. Output nothing except the one sentence or the word SKIP.

BASE ANSWER (correct — do not contradict or repeat):
{base_answer}

REPORT DATA (your only allowed source):
{intel_block}

QUESTION:
{question}

ONE ADDITIONAL SENTENCE (or SKIP):"""

_HEDGE_PHRASES = (
    "it is important to", "please consult", "i recommend",
    "you should always", "make sure to", "remember to",
    "it is worth noting", "as mentioned", "as stated",
    "as noted above", "in summary", "to summarize",
)


def _extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+\.?\d*", text))


def _contains_hallucinated_number(
    enrichment: str,
    intel_block: str,
    base_answer: str,
) -> bool:
    allowed = _extract_numbers(intel_block) | _extract_numbers(base_answer)
    invented = _extract_numbers(enrichment) - allowed
    if invented:
        logger.warning(
            "chat_enricher · discarded enrichment — invented numbers",
            invented=invented,
            enrichment=enrichment[:120],
        )
        return True
    return False


def _is_useful(addition: str, base_answer: str, intel_block: str) -> bool:
    cleaned = addition.strip().strip('"').strip("'")

    if not cleaned or len(cleaned) < 15:
        return False
    if cleaned.upper().startswith("SKIP"):
        return False

    base_words = set(re.findall(r"[a-z]+", base_answer.lower()))
    addition_words = re.findall(r"[a-z]+", cleaned.lower())
    if addition_words:
        overlap = sum(1 for w in addition_words if w in base_words) / len(addition_words)
        if overlap > 0.70:
            logger.debug("chat_enricher · discarded — too similar to base", overlap=round(overlap, 2))
            return False

    if _contains_hallucinated_number(cleaned, intel_block, base_answer):
        return False

    if any(phrase in cleaned.lower() for phrase in _HEDGE_PHRASES):
        logger.debug("chat_enricher · discarded — hedge phrase", text=cleaned[:80])
        return False

    return True


async def enrich_answer(
    *,
    base_answer: str,
    intel: ReportIntelligence,
    question: str,
    intents: list[str],
    model_registry: object,
) -> str:
    """
    Attempt to enrich a deterministic answer with one grounded sentence.

    Returns base_answer unchanged if:
      - No enrichable intent is present
      - Report lacks cross-referenceable data
      - Model is busy / memory pressure
      - Model outputs SKIP or fails validation
      - Any exception occurs

    The base_answer is ALWAYS correct and ALWAYS shown.
    Enrichment is strictly additive — never replaces.
    """
    # Gate 1: only enrichable cross-domain intents
    if not any(intent in _ENRICHABLE_INTENTS for intent in intents):
        return base_answer

    # Gate 2: only when the report has data worth cross-referencing
    has_cross_ref = (
        (bool(intel.abnormal_labs) and bool(intel.all_symptoms()))
        or bool(intel.drug_interactions)
        or (bool(intel.severity_reasons) and bool(intel.abnormal_labs))
        or (bool(intel.xray_findings) and bool(intel.all_symptoms()))
    )
    if not has_cross_ref:
        return base_answer

    intel_block = intel.to_prompt_block()

    try:
        # Gate 3: memory check — pipeline always wins
        model_registry.assert_memory_headroom("chat_enrichment")  # type: ignore[attr-defined]

        prompt = _ENRICHMENT_PROMPT.format(
            base_answer=base_answer,
            intel_block=intel_block,
            question=question,
        )

        data = await model_registry.ollama_generate(  # type: ignore[attr-defined]
            prompt,
            num_predict=80,      # ~one sentence — hard physical cap
            num_ctx=768,         # tiny — only intel block fits
            temperature=0.0,     # fully deterministic
            stream=False,
            timeout=10.0,        # hard cap — chat must feel fast
        )

        raw = str(data.get("response") or "").strip() if isinstance(data, dict) else ""
        first_line = raw.split("\n")[0].strip()

        if _is_useful(first_line, base_answer, intel_block):
            logger.info(
                "chat_enricher · answer enriched",
                job_id=intel.job_id,
                intents=intents,
                chars_added=len(first_line),
            )
            return base_answer + " " + first_line

        logger.debug(
            "chat_enricher · enrichment skipped — validation failed",
            job_id=intel.job_id,
            raw=raw[:100],
        )
        return base_answer

    except Exception as exc:
        logger.debug(
            "chat_enricher · enrichment skipped — model unavailable or error",
            job_id=intel.job_id,
            reason=str(exc),
        )
        return base_answer