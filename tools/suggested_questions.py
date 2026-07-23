# tools/suggested_questions.py
"""
Intelligent suggested-question generator.

Replaces the old keyword-rule table with a system that:
  1. Builds a ReportIntelligence from the context dict.
  2. Generates candidate questions from ACTUAL report findings —
     lab values with real numbers, specific symptoms, specific drugs,
     specific imaging findings.
  3. Scores each candidate against conversation history so already-
     answered topics don't re-surface.
  4. Skips any candidate whose text matches a question the user has
     already asked, so no chip re-appears verbatim after being used.
  5. Returns the highest-scoring, non-redundant questions capped at 4.

Every question references something that literally exists in the report.
No question is fabricated from a static rule table.

All symptoms flow through ReportIntelligence which cleans long free-text
blobs into short atomic labels — so a chip can never read like
"Could my i have been feeling very tired for the past week..." again.

Fixes vs. earlier version:
  - _lab_candidates() builds the value-with-unit string safely and
    never prints "None" as a unit (which produced "TSH (6.05None)"
    in the earlier chips).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from tools.report_analyst import (
    ReportIntelligence,
    LabFinding,
    DrugInteraction,
    XrayFinding,
    is_urgent,
    is_moderate,
    display_symptom,
)

# Only these atomic, well-understood symptom names are allowed to
# generate a "general" bucket chip — this bucket is where loose
# extraction noise tends to land, so we don't want to fabricate a
# question about something that isn't a clean, real symptom.
_KNOWN_ATOMIC_GENERAL = {
    "fever", "fatigue", "cough", "nausea", "vomiting", "diarrhea",
    "swelling", "rash", "insomnia", "weight loss", "weight gain",
    "sore throat",
}


@dataclass
class _Candidate:
    text: str
    score: float
    topic_key: str   # one question shown per topic key


# ── History fingerprint ───────────────────────────────────────────

def _fingerprint(history: list[dict]) -> set[str]:
    """Token-level set of everything said in the conversation."""
    tokens: set[str] = set()
    for item in history:
        words = re.findall(r"[a-z0-9]+", str(item.get("content") or "").lower())
        tokens.update(words)
    return tokens


def _covered(fp: set[str], *keywords: str) -> bool:
    """True if ALL keywords appear in the fingerprint."""
    for kw in keywords:
        parts = re.findall(r"[a-z0-9]+", kw.lower())
        if not all(p in fp for p in parts):
            return False
    return True


def _any_covered(fp: set[str], *keywords: str) -> bool:
    return any(_covered(fp, kw) for kw in keywords)


# ── Candidate generators ──────────────────────────────────────────

def _severity_candidates(intel: ReportIntelligence, fp: set[str]) -> list[_Candidate]:
    out: list[_Candidate] = []

    if not _any_covered(fp, "severity", "urgent", "serious", "risk"):
        if is_urgent(intel.severity):
            reason = intel.severity_reasons[0] if intel.severity_reasons else "multiple risk factors"
            out.append(_Candidate(
                text=f"This is rated {intel.severity} due to {reason} — what should I do immediately?",
                score=6.0,
                topic_key="severity_urgent",
            ))
        elif is_moderate(intel.severity):
            out.append(_Candidate(
                text=f"My report is {intel.severity} severity — how quickly should I see a doctor?",
                score=4.0,
                topic_key="severity_moderate",
            ))
        else:
            out.append(_Candidate(
                text=f"My report is {intel.severity} — is there anything I should still watch for?",
                score=2.5,
                topic_key="severity_low",
            ))

    if intel.severity_reasons and not _any_covered(fp, "reason", "driver", "factor", "critical factor"):
        reason = intel.severity_reasons[0]
        out.append(_Candidate(
            text=f"The report flags '{reason[:60]}' — can you explain that?",
            score=4.5 if is_urgent(intel.severity) else 3.0,
            topic_key="severity_reason",
        ))

    return out


def _lab_candidates(intel: ReportIntelligence, fp: set[str]) -> list[_Candidate]:
    """
    Build one chip per abnormal lab. Never renders "None" as a unit
    even when the parser stored the literal string "None" instead of
    the Python None singleton.
    """
    out: list[_Candidate] = []

    for i, lab in enumerate(intel.abnormal_labs[:5]):
        name = lab.name.strip()

        # Safe value display — no "None" leak
        val_str = ""
        if lab.value is not None and lab.value != "":
            unit_ok = lab.unit and lab.unit.lower() not in ("none", "null")
            if unit_ok:
                val_str = f" ({lab.value}{lab.unit})"
            else:
                val_str = f" ({lab.value})"

        direction = (
            "elevated" if lab.direction == "high"
            else ("low" if lab.direction == "low" else "abnormal")
        )
        key = f"lab_{re.sub(r'[^a-z0-9]', '', name.lower())}"

        if _any_covered(fp, name.lower()):
            continue

        out.append(_Candidate(
            text=f"My {name}{val_str} is {direction} — what does that mean?",
            score=4.0 - i * 0.3,
            topic_key=key,
        ))

    return out


def _symptom_candidates(intel: ReportIntelligence, fp: set[str]) -> list[_Candidate]:
    out: list[_Candidate] = []

    for symptom in intel.symptom_groups.cardiac[:2]:
        disp = display_symptom(symptom)
        key = f"symptom_cardiac_{symptom[:12]}"
        if _any_covered(fp, symptom.lower()):
            continue
        out.append(_Candidate(
            text=f"Could my {disp} be a sign of something serious?",
            score=5.0,
            topic_key=key,
        ))

    for symptom in intel.symptom_groups.respiratory[:2]:
        disp = display_symptom(symptom)
        key = f"symptom_resp_{symptom[:12]}"
        if _any_covered(fp, symptom.lower()):
            continue
        out.append(_Candidate(
            text=f"Is my {disp} linked to the {intel.severity} severity rating?",
            score=4.5,
            topic_key=key,
        ))

    for symptom in intel.symptom_groups.neurological[:2]:
        disp = display_symptom(symptom)
        key = f"symptom_neuro_{symptom[:12]}"
        if _any_covered(fp, symptom.lower()):
            continue
        out.append(_Candidate(
            text=f"What could be causing my {disp}?",
            score=4.2,
            topic_key=key,
        ))

    for symptom in intel.symptom_groups.metabolic[:1]:
        disp = display_symptom(symptom)
        key = f"symptom_meta_{symptom[:12]}"
        if _any_covered(fp, symptom.lower()):
            continue
        out.append(_Candidate(
            text=f"What does my {disp} finding suggest?",
            score=3.8,
            topic_key=key,
        ))

    # General-bucket symptoms are the noisiest category (often loose
    # extraction artifacts), so only surface a chip when the symptom is
    # a real, known atomic symptom — never fabricate/guess a chip for
    # something that isn't actually a clean recorded finding.
    for symptom in intel.symptom_groups.general[:2]:
        disp = display_symptom(symptom)
        # Containment, not exact match: extraction may return "mild fever"
        # or "unintentional weight loss" rather than the bare atomic term.
        # A real recorded symptom should still surface a chip as long as
        # a known atomic symptom is clearly present in it; this only
        # blocks names that don't contain any recognized symptom at all
        # (the actual noise case, e.g. leftover admin/lab fragments).
        if not any(atom in disp.lower() for atom in _KNOWN_ATOMIC_GENERAL):
            continue
        key = f"symptom_gen_{symptom[:12]}"
        if _any_covered(fp, symptom.lower()):
            continue
        out.append(_Candidate(
            text=f"What could be causing my {disp}?",
            score=3.0,
            topic_key=key,
        ))

    return out


def _medication_candidates(intel: ReportIntelligence, fp: set[str]) -> list[_Candidate]:
    out: list[_Candidate] = []

    if intel.drug_interactions and not _any_covered(fp, "interaction", "drug interaction"):
        top = intel.top_interaction()
        if top:
            names = " and ".join(top.drugs[:2]) if top.drugs else "my medications"
            sev = f"[{top.severity}] " if top.severity else ""
            out.append(_Candidate(
                text=f"What should I do about the {sev}interaction between {names}?",
                score=5.5,
                topic_key="drug_interaction",
            ))

    if intel.drug_warnings and not _any_covered(fp, "warning", "drug warning"):
        out.append(_Candidate(
            text=f"What does the drug warning ({intel.drug_warnings[0][:60]}) mean for me?",
            score=4.8,
            topic_key="drug_warning",
        ))

    if intel.medications and not _any_covered(fp, "medication", "medicine", "dose"):
        if len(intel.medications) == 1:
            out.append(_Candidate(
                text=f"Is {intel.medications[0]} appropriate for my current condition?",
                score=3.5,
                topic_key="medication_fit",
            ))
        else:
            names = ", ".join(intel.medications[:2])
            out.append(_Candidate(
                text=f"Could any of my medications ({names}) be contributing to these symptoms?",
                score=3.5,
                topic_key="medication_fit",
            ))

    return out


def _imaging_candidates(intel: ReportIntelligence, fp: set[str]) -> list[_Candidate]:
    out: list[_Candidate] = []

    for i, finding in enumerate(intel.xray_findings[:2]):
        if _any_covered(fp, "xray", "x-ray", "imaging", finding.text.lower()[:12]):
            continue
        score = 5.0 if finding.is_significant else 3.2
        out.append(_Candidate(
            text=f"What does '{finding.text[:70]}' in the X-ray mean for my treatment?",
            score=score - i * 0.3,
            topic_key=f"xray_{i}",
        ))

    return out


def _followup_candidates(
    intel: ReportIntelligence,
    fp: set[str],
    used_turns: int,
) -> list[_Candidate]:
    out: list[_Candidate] = []

    if used_turns >= 3 and not _any_covered(fp, "next step", "what should", "follow up"):
        action = "immediately" if is_urgent(intel.severity) else "next"
        out.append(_Candidate(
            text=f"What should I do {action} based on this report?",
            score=3.5 + used_turns * 0.3,
            topic_key="next_steps",
        ))

    if intel.symptom_duration and not _any_covered(fp, "duration", "long", "how long"):
        out.append(_Candidate(
            text=f"My symptoms have lasted {intel.symptom_duration} — is that too long to wait?",
            score=3.8,
            topic_key="duration",
        ))

    return out


def _fallback_candidates(
    intel: ReportIntelligence,
    fp: set[str],
    severity: str,
) -> list[_Candidate]:
    """
    Generic-but-grounded fallback candidates, used only once the
    specific generators above have nothing left to offer (every lab,
    symptom, medication, and imaging finding already covered).

    Previously this was a single hardcoded question
    ("What are the main concerns in my {severity} report?"), which
    meant that once real candidates ran out — easy to happen on a
    short 2-3 value report — the SAME chip kept reappearing turn
    after turn with no way to dismiss it. This pool gives several
    distinct, still report-grounded options and runs through the
    same _covered()/_already_asked() filtering as every other
    candidate, so each one is offered once and then retired.
    """
    out: list[_Candidate] = []

    if not _any_covered(fp, "main concern", "concerns"):
        out.append(_Candidate(
            text=f"What are the main concerns in my {severity} report?",
            score=1.0,
            topic_key="fallback_main_concerns",
        ))

    if not _any_covered(fp, "confidence", "how sure", "how confident"):
        out.append(_Candidate(
            text="How confident is this assessment, and why?",
            score=0.9,
            topic_key="fallback_confidence",
        ))

    if not _any_covered(fp, "trend", "improving", "worsening", "changed"):
        out.append(_Candidate(
            text="How has my health trended across my reports?",
            score=0.85,
            topic_key="fallback_trend",
        ))

    if not _any_covered(fp, "doctor", "specialist", "clinician", "physician"):
        out.append(_Candidate(
            text="Should I see a doctor about any of this, and which kind?",
            score=0.8,
            topic_key="fallback_specialist",
        ))

    if not _any_covered(fp, "recheck", "retest", "repeat test", "monitor"):
        out.append(_Candidate(
            text="Should any of these values be rechecked, and when?",
            score=0.75,
            topic_key="fallback_recheck",
        ))

    if not _any_covered(fp, "lifestyle", "diet", "exercise", "prevent"):
        out.append(_Candidate(
            text="Are there lifestyle changes that could help with this?",
            score=0.7,
            topic_key="fallback_lifestyle",
        ))

    if not _any_covered(fp, "normal range", "reference range", "what is normal"):
        out.append(_Candidate(
            text="What do the normal reference ranges mean for my results?",
            score=0.65,
            topic_key="fallback_ranges",
        ))

    return out


# ── Public API ────────────────────────────────────────────────────

def build_suggested_questions(
    severity: str,
    symptoms: list[str],
    *,
    context: dict[str, Any] | None = None,
    history: list[dict] | None = None,
    max_questions: int = 4,
) -> list[str]:
    """
    Build a ranked list of suggested follow-up questions grounded in
    actual report findings.

    Args:
        severity:      Report severity string (e.g. "HIGH").
        symptoms:      Symptom list — used as fallback when context is absent.
        context:       Full context dict from _build_report_context().
        history:       Full conversation history list.
        max_questions: Cap on returned questions (default 4).

    Returns:
        Question strings, highest-relevance first, no duplicates.
        Every question references something from the actual report.
        Any question the user has already asked verbatim is skipped
        so used chips do not re-appear.
    """
    context = context or {}
    history = history or []

    if symptoms and not context.get("extracted_symptoms"):
        context = {**context, "extracted_symptoms": symptoms}

    intel = ReportIntelligence.from_context(context)
    fp = _fingerprint(history)
    used_turns = sum(1 for m in history if m.get("role") == "user")

    # Build a set of already-asked question fingerprints so any suggestion
    # that matches a user question already asked is skipped. This is what
    # prevents a chip like "What are the main concerns in my HIGH report?"
    # from sitting there after the user has already clicked it.
    #
    # Matches on BOTH exact normalized text AND a "core words" overlap
    # (stopwords stripped) so a candidate that differs only by trivial
    # rephrasing is still recognised as already asked. Exact-match alone
    # was too fragile: a candidate regenerated on the next call with
    # slightly different wording could slip past and resurface a
    # question the user just sent.
    _STOPWORDS = {
        "a", "an", "the", "is", "are", "my", "me", "i", "to", "for",
        "of", "in", "on", "that", "this", "does", "do", "did", "what",
        "why", "how", "can", "could", "should", "would", "it", "be",
    }

    def _core_words(text):
        words = re.findall(r"[a-z0-9]+", text.lower())
        return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 2)

    asked_normalized: set[str] = set()
    asked_core = []
    for item in history:
        if item.get("role") == "user":
            raw = str(item.get("content") or "")
            text = re.sub(r"[^a-z0-9 ]", "", raw.lower())
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                asked_normalized.add(text)
                core = _core_words(raw)
                if core:
                    asked_core.append(core)

    def _already_asked(candidate_text):
        norm = re.sub(r"[^a-z0-9 ]", "", candidate_text.lower())
        norm = re.sub(r"\s+", " ", norm).strip()
        if norm in asked_normalized:
            return True
        cand_core = _core_words(candidate_text)
        if not cand_core:
            return False
        for prior_core in asked_core:
            if not prior_core:
                continue
            overlap = len(cand_core & prior_core)
            smaller = min(len(cand_core), len(prior_core))
            if smaller and overlap / smaller >= 0.8:
                return True
        return False

    all_candidates: list[_Candidate] = []
    all_candidates.extend(_severity_candidates(intel, fp))
    all_candidates.extend(_lab_candidates(intel, fp))
    all_candidates.extend(_symptom_candidates(intel, fp))
    all_candidates.extend(_medication_candidates(intel, fp))
    all_candidates.extend(_imaging_candidates(intel, fp))
    all_candidates.extend(_followup_candidates(intel, fp, used_turns))
    all_candidates.extend(_fallback_candidates(intel, fp, severity))

    all_candidates.sort(key=lambda c: c.score, reverse=True)

    seen_keys: set[str] = set()
    seen_texts: set[str] = set()
    result: list[str] = []

    for candidate in all_candidates:
        if candidate.topic_key in seen_keys:
            continue
        norm = re.sub(r"[^a-z0-9 ]", "", candidate.text.lower())
        norm = re.sub(r"\s+", " ", norm).strip()
        if norm in seen_texts:
            continue
        # Skip if the user already asked this question (exact match or
        # near-duplicate by substantive word overlap) — that chip should
        # not re-appear after being clicked.
        if _already_asked(candidate.text):
            continue
        seen_keys.add(candidate.topic_key)
        seen_texts.add(norm)
        result.append(candidate.text)
        if len(result) >= max_questions:
            break

    # Note: `result` can legitimately be empty once every specific
    # candidate AND every fallback candidate has been covered or
    # already asked — that's correct behaviour for a long, thorough
    # conversation rather than something to paper over with a repeat
    # of a question already answered.
    return result